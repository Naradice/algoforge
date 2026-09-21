"""
Dataset, embedding-precompute, and calibration/latency support for Model B
(model_core/architectures/jev_cross_attn.py -- docs/jev-replication.md's Phase 2 follow-up).

Unlike typed_decision.py's TypedDecisionDataset (state + all N questions concatenated into one
sequence, with `[Q_<id>]` marker tokens), this dataset keeps STATE and QUESTIONS separate --
there's no shared sequence for a marker token to live in, since each question is evaluated via its
own independent cross-attention call against the state's encoder output (see jev_cross_attn.py's
module docstring for why). Reuses typed_decision.py's loss/calibration MATH (compute_losses,
score_from_logits, flatten_calibration_metrics, and the private ECE/Brier/correlation helpers --
all pure functions of {question_id: logits}/{question_id: targets}, with no dependency on which
model produced the logits) rather than duplicating it; only the "run the model over a batch"
orchestration differs, because JevCrossAttnModel's forward signature differs from JevBertModel's.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .typed_decision import (
    _brier_score, _correlation, _expected_calibration_error, _question_specs_from_record,
    compute_losses,
)

CROSS_ATTN_ARCHITECTURES = ("jev_cross_attn_v1",)


class CrossAttnTypedDecisionDataset(Dataset):
    def __init__(self, artifact_path: Path, tokenizer, max_state_length: int = 512,
                 val_split: float = 0.2, split_seed: int = 42):
        records = []
        with open(artifact_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise ValueError(f"{artifact_path} contains no examples")

        # Same fixed-question-set requirement as TypedDecisionDataset -- see that class's
        # docstring for why a mismatch is rejected rather than silently handled.
        self.question_specs = _question_specs_from_record(records[0]["questions"])
        self.question_ids = list(self.question_specs.keys())
        for i, rec in enumerate(records):
            if _question_specs_from_record(rec["questions"]) != self.question_specs:
                raise ValueError(
                    f"record {i} has a different question schema than record 0 -- "
                    "CrossAttnTypedDecisionDataset requires a fixed question set"
                )
        # The question TEXT itself is identical across every record (fixed set) -- kept once here
        # (not per-example) so compute_question_embeddings only ever tokenizes/embeds each
        # question a single time for the whole dataset, matching this design's "questions are
        # cheap, computed once" premise (see jev_cross_attn.py's module docstring).
        self.question_instructions = {qid: q["instructions"] for qid, q in records[0]["questions"].items()}

        self.tokenizer = tokenizer
        self.max_state_length = max_state_length
        self.examples = [self._encode(rec) for rec in records]

        rng = random.Random(split_seed)
        indices = list(range(len(self.examples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split)) if len(indices) > 1 else 0
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]

    def _encode(self, rec: dict) -> dict:
        state_text = json.dumps(rec["state"]) if isinstance(rec["state"], dict) else str(rec["state"])
        state_input_ids = self.tokenizer.encode(
            state_text, add_special_tokens=True, truncation=True, max_length=self.max_state_length
        )
        targets: dict[str, float | int] = {}
        for qid, spec in self.question_specs.items():
            answer = rec["answers"][qid]
            if spec["type"] == "noul":
                targets[qid] = float(answer["noul"])
            else:
                targets[qid] = spec["options"].index(answer["choice"])
        return {"state_input_ids": state_input_ids, "targets": targets}

    @property
    def model_question_specs(self) -> dict[str, dict]:
        """Same conversion as TypedDecisionDataset's -- see that class's docstring."""
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


def collate_cross_attn(batch: list[dict], pad_token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    max_len = max(len(ex["state_input_ids"]) for ex in batch)
    state_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    state_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        seq = ex["state_input_ids"]
        state_input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        state_attention_mask[i, : len(seq)] = 1
    question_ids = batch[0]["targets"].keys()
    targets = {qid: torch.tensor([ex["targets"][qid] for ex in batch]) for qid in question_ids}
    return state_input_ids.to(device), state_attention_mask.to(device), targets


@torch.no_grad()
def compute_question_embeddings(model, tokenizer, question_instructions: dict[str, str], device: str) -> dict[str, torch.Tensor]:
    """Runs each fixed question's instruction text through the state encoder's own embedding
    layer (word + position + token-type, BERT's usual input-embedding step -- no self-attention)
    ONCE, producing the fixed query sequence jev_cross_attn.py's forward()/forward_batched() take
    as `question_embeds_by_id`. Deliberately no-grad / computed once per training run (not once
    per batch) -- this IS the "questions are cheap and reusable" half of this design's premise;
    only the state side is re-encoded per batch. A model wanting to fine-tune question embeddings
    end-to-end would need to move this inside the training loop instead -- not done here since
    that would undercut exactly the property being tested.
    """
    out = {}
    for qid, instructions in question_instructions.items():
        ids = tokenizer.encode(instructions, add_special_tokens=True, return_tensors="pt").to(device)
        out[qid] = model.state_encoder.embeddings(input_ids=ids).squeeze(0)
    return out


@torch.no_grad()
def compute_calibration_metrics(model, dataset: CrossAttnTypedDecisionDataset, question_embeds: dict,
                                 indices: list[int], batch_size: int, device: str) -> dict:
    """Model-B equivalent of typed_decision.py's compute_calibration_metrics -- same metric
    definitions (imported, not reimplemented), different orchestration (JevCrossAttnModel's
    forward signature)."""
    model.eval()
    per_question: dict[str, dict] = {qid: {"confidences": [], "correctness": []} for qid in dataset.question_ids}

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start: start + batch_size]
        batch = [dataset[i] for i in batch_indices]
        state_input_ids, state_attention_mask, targets = collate_cross_attn(batch, dataset.tokenizer.pad_token_id, device)
        outputs = model(state_input_ids, state_attention_mask, question_embeds)
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
        else:
            entry["correlation"] = _correlation(confs, correct)
        result[qid] = entry
    return result


@torch.no_grad()
def measure_cross_attn_latency(model, dataset: CrossAttnTypedDecisionDataset, question_embeds: dict,
                                indices: list[int], device: str, n_calls: int = 20, n_warmup: int = 3) -> dict:
    """Phase 2 follow-up's actual test: measures single-call latency for state encoding alone,
    for the full forward_batched() call (all questions batched in one cross-attention call), and
    for forward() (the unbatched per-question loop) -- so a direct N-scaling comparison against
    typed_decision.py's measure_inference_latency (Model A) can isolate exactly which part of the
    cost grows with N and how each design's *total* latency responds to it. See
    jev_cross_attn.py's module docstring for why any parallelism benefit here is CPU-level only
    (this environment has no GPU -- docs/jev-replication.md's Phase 2 Status section), not the GPU
    batch-parallelism Jev's own docs are presumably describing.
    """
    model.eval()
    sample_indices = (indices * ((n_calls + n_warmup) // max(len(indices), 1) + 1))[: n_calls + n_warmup]

    def _timed(fn) -> list[float]:
        durations = []
        for i, idx in enumerate(sample_indices):
            batch = [dataset[idx]]
            state_input_ids, state_attention_mask, _ = collate_cross_attn(batch, dataset.tokenizer.pad_token_id, device)
            start = time.perf_counter()
            fn(state_input_ids, state_attention_mask)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if i >= n_warmup:
                durations.append(elapsed_ms)
        durations.sort()
        return durations

    state_only = _timed(lambda ids, mask: model.encode_state(ids, mask))
    batched_all_q = _timed(lambda ids, mask: model.forward_batched(ids, mask, question_embeds))
    unbatched_loop = _timed(lambda ids, mask: model(ids, mask, question_embeds))

    def _stats(durations: list[float]) -> dict:
        n = len(durations)
        return {
            "n_calls": n,
            "latency_ms_mean": sum(durations) / n if n else float("nan"),
            "latency_ms_median": durations[n // 2] if n else float("nan"),
        }

    return {
        "n_questions": len(dataset.question_ids),
        "state_encode_only": _stats(state_only),
        "batched_all_questions": _stats(batched_all_q),
        "unbatched_per_question_loop": _stats(unbatched_loop),
    }


def flatten_cross_attn_metrics(per_question: dict, latency: dict) -> dict:
    """Same flattening convention as typed_decision.py's flatten_calibration_metrics (study_
    manager's Evaluate does a flat metrics.get(name) lookup -- see that function's docstring),
    plus this architecture's three-way latency breakdown under distinctly-named flat keys so a
    Research Brief can reference e.g. "latency_ms_mean_batched" without colliding with Model A's
    plain "latency_ms_mean" if both ever appear in the same comparison."""
    flat: dict = {"calibration": per_question, "latency_breakdown": latency}
    eces, briers = [], []
    for qid, entry in per_question.items():
        for key in ("accuracy", "ece", "brier", "correlation"):
            if key in entry:
                flat[f"{key}_{qid}"] = entry[key]
        eces.append(entry["ece"])
        briers.append(entry["brier"])
    flat["expected_calibration_error"] = sum(eces) / len(eces) if eces else float("nan")
    flat["brier_score"] = sum(briers) / len(briers) if briers else float("nan")
    flat["ece"] = flat["expected_calibration_error"]
    flat["brier"] = flat["brier_score"]
    flat["n_questions"] = latency["n_questions"]
    flat["latency_ms_mean_state_only"] = latency["state_encode_only"]["latency_ms_mean"]
    flat["latency_ms_mean_batched"] = latency["batched_all_questions"]["latency_ms_mean"]
    flat["latency_ms_mean_unbatched"] = latency["unbatched_per_question_loop"]["latency_ms_mean"]
    # Alias matching Model A's own flat key name -- the single number most directly comparable
    # across the two architectures for the same N (the batched path is this design's actual
    # claim, so it -- not the unbatched loop -- is what "latency_ms_mean" means here).
    flat["latency_ms_mean"] = flat["latency_ms_mean_batched"]
    return flat
