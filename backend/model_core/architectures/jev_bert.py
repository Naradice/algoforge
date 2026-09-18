"""
Model A (docs/jev-replication.md Phase 1): a pretrained BERT encoder plus one typed decision
head per fixed question, each head reading the hidden state at that question's own marker token
-- not the shared [CLS] token -- so the "parallel decisions in one shared-state sequence" design
Jev's docs describe (docs/jev-replication.md's Phase 0 spec) is actually exercised, not just a
single-question classifier wearing multiple heads.

Input layout, one sequence per example (built by
model_core/trainers/typed_decision.py:TypedDecisionDataset):

    [CLS] <state text> [SEP] [Q_<id1>] <question 1 instructions> [Q_<id2>] <question 2 instructions> ... [SEP]

Each `[Q_<id>]` is a real vocabulary token (added via tokenizer.add_special_tokens +
resize_token_embeddings, not reused from BERT's existing special tokens) whose position in the
sequence is recorded per example; this module gathers the encoder's output hidden state at that
exact position and feeds it to the matching head. This is deliberately the "special token"
design (docs/jev-replication.md's own plan calls it "B1"), not the [CLS]-only design a plain
classification model would use -- Phase 3 compares this against alternatives (learned query
tokens, cross-attention), not this file.

Head shapes, one per question id, chosen from that question's type (see
data/collectors/llm_typed_decisions.py's QUESTIONS for what "type"/"criteria" mean):
    noul:   Linear(hidden, 1)         -- a logit; sigmoid at inference gives the 0-1 probability
    choice: Linear(hidden, n_options) -- a logit per option key, in criteria's key order
    score:  Linear(hidden, n_levels)  -- a logit per level, in criteria's list order; the
            continuous "score" position Jev's docs describe is the softmax-probability-weighted
            expectation over level indices, computed by the caller (see typed_decision.py's
            score_from_logits), not by this module -- this module only produces logits for
            whichever loss (CE for choice/score, BCE for noul) trains them.

Fixed-question-set only (Phase 1). A model whose head set is read from each example's own
question list instead of fixed at construction time is Phase 3/4's job, not this one's.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JevBertModel(nn.Module):
    def __init__(
        self,
        question_specs: dict[str, dict],
        pretrained_name: str = "bert-base-uncased",
        vocab_size: int | None = None,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """*question_specs*: {question_id: {"type": "noul"|"choice"|"score", "n_options": int}}
        ("n_options" ignored for "noul"). *vocab_size* must be passed as
        tokenizer_len (after add_special_tokens) if it differs from the pretrained checkpoint's
        own vocab -- the caller (typed_decision.py) is responsible for resizing consistently;
        this module just builds embeddings at whatever size it's told.
        """
        super().__init__()
        from transformers import AutoModel

        self.question_ids = list(question_specs.keys())
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        if vocab_size is not None and vocab_size != self.encoder.config.vocab_size:
            self.encoder.resize_token_embeddings(vocab_size)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict()
        for qid, spec in question_specs.items():
            out_dim = 1 if spec["type"] == "noul" else int(spec["n_options"])
            self.heads[qid] = nn.Linear(hidden, out_dim)
        self.to(device)
        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                q_positions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """*q_positions*: {question_id: LongTensor[batch]} -- the index (within each example's
        own sequence) of that question's marker token, gathered from
        TypedDecisionDataset.__getitem__. Returns {question_id: logits[batch, out_dim]}."""
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size = input_ids.shape[0]
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        outputs: dict[str, torch.Tensor] = {}
        for qid in self.question_ids:
            pos = q_positions[qid].to(hidden_states.device)
            token_hidden = hidden_states[batch_idx, pos]  # [batch, hidden]
            outputs[qid] = self.heads[qid](self.dropout(token_hidden))
        return outputs

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
