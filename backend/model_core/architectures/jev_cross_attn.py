"""
Model B (docs/jev-replication.md Phase 2 follow-up): tests whether DECOUPLING state encoding from
question evaluation -- rather than JevBertModel's (Phase 1, "Model A") single concatenated
sequence -- lets a BERT-family stand-in actually deliver Jev's "Speculative Fan-Out" claim
("adding more questions to a call typically doesn't add any latency to the response").

Model A's design (state + all N questions in one sequence, self-attention over the whole thing)
necessarily couples every question's cost to every other question's presence -- Phase 2 measured
this directly: latency roughly doubled when N went 3 -> 6. This module's design instead:

  1. Encodes the STATE once (`state_encoder`, a plain pretrained BERT) -- this cost depends only
     on state length, never on N.
  2. Treats each question's own text as a small, FIXED query sequence (fixed because this is still
     a Phase 1/2-style fixed question set -- the question text doesn't vary per example, so its
     token embeddings are computed once at dataset-build time, not per forward call; see
     model_core/trainers/cross_attn_typed_decision.py). Each question cross-attends (queries
     against the state's own hidden states as keys/values) to produce one pooled vector per
     question, fed to that question's own typed-decision head -- structurally identical head
     shapes to jev_bert.py's JevBertModel (Linear(hidden,1) for noul, Linear(hidden,n_options) for
     choice/score).
  3. Question 1..N's cross-attention calls are independent of each other given the shared state
     hidden states -- so they can be BATCHED into one call (`forward`'s `question_ids=None` path
     stacks every question as an extra batch dimension against the same state, one
     nn.MultiheadAttention call handling all of them) instead of N sequential ones. This is the
     concrete mechanism Jev's docs gesture at ("evaluated in parallel") and Model A's single-
     sequence self-attention structurally cannot do (every token depends on every other token in
     one attention computation, so there is no way to "skip" the coupling between questions).

Total compute still grows with N (there is no free lunch -- see docs/jev-replication.md's Phase 2
conclusion on this point); what changes is that the *added* cost per extra question is one cheap,
independent cross-attention pass rather than a share of a bigger self-attention matrix that every
existing question was already paying into. Whether that difference shows up as measurably flatter
wall-clock latency (this environment is CPU-only -- see Phase 2's Status section on this machine's
resource constraints, so any parallelism benefit here is CPU SIMD/multi-core-level, not the GPU
batch-parallelism Jev's own docs are presumably describing) is exactly what
measure_cross_attn_latency (cross_attn_typed_decision.py) tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class JevCrossAttnModel(nn.Module):
    def __init__(
        self,
        question_specs: dict[str, dict],
        pretrained_name: str = "bert-base-uncased",
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """*question_specs*: {question_id: {"type": "noul"|"choice"|"score", "n_options": int}}
        -- identical shape to jev_bert.JevBertModel's, so the same
        TypedDecisionDataset.model_question_specs property feeds both. Unlike JevBertModel, this
        model's vocabulary is NOT resized (no `[Q_<id>]` marker tokens -- there's no shared
        sequence for a marker to live in), so `pretrained_name`'s own tokenizer/vocab is used
        as-is.
        """
        super().__init__()
        from transformers import AutoModel

        self.question_ids = list(question_specs.keys())
        self.state_encoder = AutoModel.from_pretrained(pretrained_name)
        hidden = self.state_encoder.config.hidden_size
        # batch_first=True: state_hidden/query shapes are [batch, seq, hidden] throughout this
        # module, matching every other tensor here -- avoids the transpose bugs that come from
        # mixing MultiheadAttention's legacy [seq, batch, hidden] default with the rest of the
        # typed-decision pipeline's [batch, seq, ...] convention.
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict()
        for qid, spec in question_specs.items():
            out_dim = 1 if spec["type"] == "noul" else int(spec["n_options"])
            self.heads[qid] = nn.Linear(hidden, out_dim)
        self.to(device)
        self.device = device

    def encode_state(self, state_input_ids: torch.Tensor, state_attention_mask: torch.Tensor) -> torch.Tensor:
        """The expensive, N-independent step -- call once per batch, reuse the result for every
        question (see forward() and measure_cross_attn_latency, which time this separately from
        the per-question cross-attention step for exactly this reason)."""
        return self.state_encoder(input_ids=state_input_ids, attention_mask=state_attention_mask).last_hidden_state

    def evaluate_question(
        self, qid: str, state_hidden: torch.Tensor, state_key_padding_mask: torch.Tensor,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """One question's cost, independent of every other question -- *question_embeds*
        [q_len, hidden] is that question's own fixed, precomputed token embedding sequence
        (see cross_attn_typed_decision.py's CrossAttnTypedDecisionDataset), broadcast to the
        batch. Returns logits [batch, out_dim]."""
        batch_size = state_hidden.shape[0]
        query = question_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(query, state_hidden, state_hidden, key_padding_mask=state_key_padding_mask)
        pooled = attn_out.mean(dim=1)  # mean-pool over the (short, fixed) question length
        return self.heads[qid](self.dropout(pooled))

    def forward_batched(
        self, state_input_ids: torch.Tensor, state_attention_mask: torch.Tensor,
        question_embeds_by_id: dict[str, torch.Tensor], question_ids: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluates every question in *question_ids* (default: all of them) as ONE batched
        cross-attention call -- state_hidden and each question's key_padding_mask are repeated
        across a [batch * n_questions] pseudo-batch dimension instead of looping. This is the
        literal "speculative fan-out" mechanism this module exists to test (see module docstring)
        -- forward() (the plain, one-call-per-question loop) is kept as a simpler reference
        implementation and for correctness testing, not for the latency comparison itself.
        """
        qids = question_ids if question_ids is not None else self.question_ids
        state_hidden = self.encode_state(state_input_ids, state_attention_mask)
        batch_size, state_len, hidden = state_hidden.shape
        n_q = len(qids)
        key_padding_mask = ~state_attention_mask.bool()

        # Stack questions as an extra batch dimension -- [n_q, batch, ...] flattened to
        # [n_q * batch, ...] so nn.MultiheadAttention sees one big batch, one call, one matmul.
        state_rep = state_hidden.unsqueeze(0).expand(n_q, -1, -1, -1).reshape(n_q * batch_size, state_len, hidden)
        mask_rep = key_padding_mask.unsqueeze(0).expand(n_q, -1, -1).reshape(n_q * batch_size, state_len)
        # Different questions have different (fixed) token lengths, so batching them together
        # needs query-side padding too -- q_len_mask tracks which query positions are real vs
        # padding, so the mean-pool below only averages over real positions (padded positions get
        # a defined attention output too, since they're valid queries into a real key/value set,
        # but including them in the pool would dilute shorter questions' pooled vector with
        # padding-position output -- confirmed live: without this mask, forward_batched's output
        # diverged from forward()'s unbatched loop by up to ~0.003, small but real).
        max_q_len = max(question_embeds_by_id[q].shape[0] for q in qids)
        query = state_hidden.new_zeros((n_q, batch_size, max_q_len, hidden))
        q_len_mask = state_hidden.new_zeros((n_q, max_q_len))
        for i, qid in enumerate(qids):
            qe = question_embeds_by_id[qid]
            query[i, :, : qe.shape[0], :] = qe.unsqueeze(0)
            q_len_mask[i, : qe.shape[0]] = 1.0
        query = query.reshape(n_q * batch_size, max_q_len, hidden)
        q_len_mask = q_len_mask.unsqueeze(1).expand(n_q, batch_size, max_q_len).reshape(n_q * batch_size, max_q_len, 1)

        attn_out, _ = self.cross_attn(query, state_rep, state_rep, key_padding_mask=mask_rep)
        pooled = (attn_out * q_len_mask).sum(dim=1) / q_len_mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled.reshape(n_q, batch_size, hidden)

        return {qid: self.heads[qid](self.dropout(pooled[i])) for i, qid in enumerate(qids)}

    def forward(
        self, state_input_ids: torch.Tensor, state_attention_mask: torch.Tensor,
        question_embeds_by_id: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reference (unbatched, one cross-attention call per question) path -- used for training
        (where the per-question loop's clarity/debuggability matters more than the last bit of
        speed) and as a correctness baseline for forward_batched. Returns {question_id: logits}."""
        state_hidden = self.encode_state(state_input_ids, state_attention_mask)
        key_padding_mask = ~state_attention_mask.bool()
        return {
            qid: self.evaluate_question(qid, state_hidden, key_padding_mask, question_embeds_by_id[qid])
            for qid in self.question_ids
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
