"""
Dataset, loss, and calibration-metric support for Model A (docs/jev-replication.md Phase 1) --
see model_core/architectures/jev_bert.py for the model itself. Kept as its own dedicated module
(not wired into this package's get_trainer_fns/get_default_criterion dispatch, and
architectures/jev_bert.py's JevBertModel is not wired into architectures.build_model) for the
same reason arima_trainer.py isn't either: the input/label/loss shape here (variable-length
tokenized text, multiple typed heads, mixed CE/BCE losses) has nothing in common with the
OHLCWindowDataset + single-criterion supervised loop every other architecture shares, so forcing
it through that dispatch would mean special-casing the shared path instead of just having its own
-- see celery_worker.py's TYPED_DECISION_ARCHITECTURES branch (mirrors ARIMA_ARCHITECTURES).

TypedDecisionDataset assumes every example shares the same fixed question set (Phase 1) --
Phase 3/4's dynamic per-example questions need a different collate/label story (variable head
count per batch) and aren't handled here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

TYPED_DECISION_ARCHITECTURES = ("jev_bert_v1",)


def _question_specs_from_record(questions: dict) -> dict[str, dict]:
    specs = {}
    for qid, q in questions.items():
        qtype = q["type"]
        if qtype == "noul":
            specs[qid] = {"type": "noul"}
        elif qtype == "choice":
            specs[qid] = {"type": "choice", "options": list(q["criteria"].keys())}
        elif qtype == "score":
            specs[qid] = {"type": "score", "options": list(q["criteria"])}
        else:
            raise ValueError(f"unknown question type {qtype!r} for question {qid!r}")
    return specs


class TypedDecisionDataset(Dataset):
    def __init__(
        self, artifact_path: Path, tokenizer, max_length: int = 512,
        val_split: float = 0.2, split_seed: int = 42,
    ):
        records = []
        with open(artifact_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"{artifact_path} contains no examples")

        # Fixed-question-set assumption: derive the schema once, from the first record, and
        # require every other record to match it exactly -- a mismatch means the dataset wasn't
        # actually generated with a fixed question set, which this class can't handle correctly
        # (see module docstring).
        self.question_specs = _question_specs_from_record(records[0]["questions"])
        self.question_ids = list(self.question_specs.keys())
        for i, rec in enumerate(records):
            if _question_specs_from_record(rec["questions"]) != self.question_specs:
                raise ValueError(
                    f"record {i} has a different question schema than record 0 -- "
                    "TypedDecisionDataset requires a fixed question set (Phase 1); a dynamic "
                    "one (Phase 3/4) needs a different dataset class"
                )

        marker_tokens = [f"[Q_{qid}]" for qid in self.question_ids]
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})
        if num_added != len(marker_tokens):
            # Only a real problem the first time (a re-used tokenizer across multiple dataset
            # instances for the same question set would legitimately add 0 new tokens the second
            # time) -- but silently accepting a partial add would corrupt position tracking below,
            # so verify every marker actually made it into the vocab regardless of *why*.
            missing = [t for t in marker_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
            if missing:
                raise ValueError(f"tokenizer failed to register marker tokens: {missing}")
        self.marker_ids = {
            qid: tokenizer.convert_tokens_to_ids(f"[Q_{qid}]") for qid in self.question_ids
        }
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = [self._encode(rec) for rec in records]

        rng = random.Random(split_seed)
        indices = list(range(len(self.examples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split)) if len(indices) > 1 else 0
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]

    def _encode(self, rec: dict) -> dict:
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        state_text = json.dumps(rec["state"]) if isinstance(rec["state"], dict) else str(rec["state"])
        state_ids = self.tokenizer.encode(state_text, add_special_tokens=False)

        # Reserve room for [CLS], the state/question separator [SEP], one marker id per question,
        # each question's own instruction tokens, and the trailing [SEP] -- truncate the STATE
        # (not the questions, which are short and fixed) if the total would exceed max_length.
        question_pieces = []
        for qid in self.question_ids:
            q = rec["questions"][qid]
            instr_ids = self.tokenizer.encode(q["instructions"], add_special_tokens=False)
            question_pieces.append((self.marker_ids[qid], instr_ids))
        fixed_len = 3 + len(self.question_ids) + sum(len(ids) for _, ids in question_pieces)  # CLS + 2*SEP + markers
        max_state_len = max(1, self.max_length - fixed_len)
        state_ids = state_ids[:max_state_len]

        input_ids = [cls_id] + state_ids + [sep_id]
        q_positions: dict[str, int] = {}
        for qid, (marker_id, instr_ids) in zip(self.question_ids, question_pieces):
            q_positions[qid] = len(input_ids)
            input_ids.append(marker_id)
            input_ids.extend(instr_ids)
        input_ids.append(sep_id)
        input_ids = input_ids[: self.max_length]

        targets: dict[str, float | int] = {}
        for qid, spec in self.question_specs.items():
            answer = rec["answers"][qid]
            if spec["type"] == "noul":
                targets[qid] = float(answer["noul"])
            else:
                targets[qid] = spec["options"].index(answer["choice"])

        return {"input_ids": input_ids, "q_positions": q_positions, "targets": targets}

    @property
    def model_question_specs(self) -> dict[str, dict]:
        """JevBertModel's expected shape ({"type", "n_options"}), derived from this dataset's own
        ({"type", "options"}) -- kept as one conversion here rather than duplicated at every call
        site that builds a JevBertModel from a TypedDecisionDataset."""
        specs = {}
        for qid, spec in self.question_specs.items():
            if spec["type"] == "noul":
                specs[qid] = {"type": "noul"}
            else:
                specs[qid] = {"type": spec["type"], "n_options": len(spec["options"])}
        return specs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_typed_decisions(batch: list[dict], pad_token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        seq = ex["input_ids"]
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1

    question_ids = batch[0]["q_positions"].keys()
    q_positions = {
        qid: torch.tensor([ex["q_positions"][qid] for ex in batch], dtype=torch.long) for qid in question_ids
    }
    targets = {
        qid: torch.tensor([ex["targets"][qid] for ex in batch]) for qid in question_ids
    }
    return input_ids.to(device), attention_mask.to(device), q_positions, targets


def compute_losses(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                    question_specs: dict[str, dict], device: str) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum of per-question losses (BCE for noul, CE for choice/score) -- the combined scalar is
    what TrainingRunMetric's train_loss/val_loss track; the per-question breakdown is returned
    alongside for logging, not persisted per-epoch (see celery_worker.py's
    _run_typed_decision_training -- only the final calibration report goes to ModelValidation)."""
    total = torch.zeros((), device=device)
    per_question: dict[str, float] = {}
    for qid, spec in question_specs.items():
        logits = outputs[qid]
        target = targets[qid].to(device)
        if spec["type"] == "noul":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), target.float())
        else:
            loss = torch.nn.functional.cross_entropy(logits, target.long())
        per_question[qid] = loss.item()
        total = total + loss
    return total, per_question


def score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """The continuous "score" position Jev's docs describe: the softmax-probability-weighted
    expectation over level indices (see model_core/architectures/jev_bert.py's docstring) --
    used for Score questions; Choice questions only need the argmax + probabilities, not this."""
    probs = torch.softmax(logits, dim=-1)
    levels = torch.arange(logits.shape[-1], device=logits.device, dtype=probs.dtype)
    return (probs * levels).sum(dim=-1)


def _expected_calibration_error(confidences: list[float], correctness: list[float], n_bins: int = 10) -> float:
    """Standard binned ECE: |mean(confidence) - mean(correctness)| per bin, weighted by bin
    size. "correctness" is 1.0/0.0 for Choice/Score (did the argmax match the label) or the
    actual 0/1 label for Noul (there, "confidence" already *is* the predicted probability of the
    positive outcome, so this reduces to the textbook binary-calibration definition Jev's own
    docs give: "Outcomes assigned a probability of 0.2 should occur about 20% of the time")."""
    if not confidences:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, correctness):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    n = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        mean_acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(mean_conf - mean_acc)
    return ece


def _brier_score(confidences: list[float], correctness: list[float]) -> float:
    if not confidences:
        return float("nan")
    return sum((c - y) ** 2 for c, y in zip(confidences, correctness)) / len(confidences)


@torch.no_grad()
def compute_calibration_metrics(model, dataset: TypedDecisionDataset, indices: list[int],
                                 batch_size: int, device: str) -> dict:
    """Runs the model over *indices* (typically the val split) and returns, per question id:
    accuracy (choice/score only), ECE, and Brier score -- see docs/jev-replication.md's Phase 0
    spec for why these specific metrics (they're exactly what Jev's own RLCD training objective
    targets, per that doc's "Training objective (RLCD)" section). Written to a ModelValidation
    row by the caller (celery_worker.py's _run_typed_decision_training), not by this function.
    """
    model.eval()
    per_question: dict[str, dict] = {qid: {"confidences": [], "correctness": []} for qid in dataset.question_ids}

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = [dataset[i] for i in batch_indices]
        input_ids, attention_mask, q_positions, targets = collate_typed_decisions(
            batch, dataset.tokenizer.pad_token_id, device
        )
        outputs = model(input_ids, attention_mask, q_positions)
        for qid, spec in dataset.question_specs.items():
            logits = outputs[qid]
            target = targets[qid].to(device)
            if spec["type"] == "noul":
                probs = torch.sigmoid(logits.squeeze(-1))
                per_question[qid]["confidences"].extend(probs.tolist())
                per_question[qid]["correctness"].extend(target.float().tolist())
            else:
                probs = torch.softmax(logits, dim=-1)
                max_probs, preds = probs.max(dim=-1)
                per_question[qid]["confidences"].extend(max_probs.tolist())
                per_question[qid]["correctness"].extend((preds == target).float().tolist())

    result = {}
    for qid, spec in dataset.question_specs.items():
        confs = per_question[qid]["confidences"]
        correct = per_question[qid]["correctness"]
        entry = {
            "ece": _expected_calibration_error(confs, correct),
            "brier": _brier_score(confs, correct),
            "n": len(confs),
        }
        if spec["type"] != "noul":
            entry["accuracy"] = sum(correct) / len(correct) if correct else float("nan")
        result[qid] = entry
    return result
