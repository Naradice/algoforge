"""
LLM-synthesized typed-decision collector — for the Jev replication investigation (see
docs/jev-replication.md). Generates a synthetic dataset of (state, questions, gold answers)
examples in TypeSafe/Jev's own request/response shape (see docs/jev-replication.md's Phase 0
spec), via an LLM, since there is no access to Jev's real training data or weights -- only its
public API contract. Unlike every other collector, this one's "ground truth" is itself an LLM's
output, not observed/downloaded real-world data -- the model trained on it (model_core's
jev_bert architecture) can only ever be evaluated against how well it reproduces the LABELING
LLM's judgments, not Jev's. That's a real, known ceiling on this whole investigation's external
validity, not a bug in this collector.

Datasource config shape (stored in datasources.config):
    {
        "n_examples": 500,        # total examples to generate
        "batch_size": 10,         # examples requested per LLM call (cost/latency vs. reliability
                                   # of a single JSON response holding many examples)
        "llm_model": "gpt-4o-mini",
        "seed": 42,               # only affects which scenario prompts are sampled, not the LLM's
                                   # own sampling (temperature is fixed below, not exposed here)
    }

Fixed question set (Phase 1 -- see docs/jev-replication.md; "dynamic questions" is Phase 3/4, not
this collector). Chosen to mirror TypeSafe's own worked example domain (a customer-support
ticket triage) as closely as possible, one question per primitive type:
    - refund_requested (noul):  does the ticket ask for a refund?
    - urgency (score):          low < medium < high < critical
    - category (choice):        billing | shipping | technical | other

Output: one JSON object per line (JSONL), each shaped exactly like a TypeSafe request+response
pair -- "questions" keyed by id with type/instructions/criteria, "answers" keyed by the same ids
-- so model_core's dataset loader needs no separate schema and this file is already in the shape
Phase 3/4's dynamic (per-example) questions will need, even though every example uses the same
fixed three here.

Uses algoforge's existing Gemini wiring (GOOGLE_API_KEY/GEMINI_API_KEY, google-genai package --
see strategy/engine/llm_condition.py for the same pattern elsewhere in this codebase) rather than
introducing a second LLM provider/credential just for this collector.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("llm_typed_decisions_collector")

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

# Fixed for Phase 1 -- see module docstring. "criteria" here is exactly what a real TypeSafe
# question's "criteria" field would hold (an ordered list for Score, a description map for
# Choice), so QUESTIONS below can be embedded verbatim into every generated example.
QUESTIONS: dict[str, dict] = {
    "refund_requested": {
        "type": "noul",
        "instructions": "Does `ticket_message` request a refund?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is `ticket_message`?",
        "criteria": ["low", "medium", "high", "critical"],
    },
    "category": {
        "type": "choice",
        "instructions": "What category best fits `ticket_message`?",
        "criteria": {
            "billing": "Payment, charges, invoices, or refunds",
            "shipping": "Delivery, tracking, or package condition",
            "technical": "App or website bugs, errors, or outages",
            "other": "Anything not covered by the above",
        },
    },
}

# Sampled (not exhaustive) per generation batch purely to nudge the labeling LLM toward a mix of
# scenarios instead of drifting into one repeated pattern across a large n_examples -- the LLM
# still invents the actual ticket text and gold labels, this is just a topic hint.
_SCENARIO_HINTS = [
    "a duplicate charge on a credit card",
    "a package that arrived damaged",
    "a package that never arrived",
    "difficulty logging into an account",
    "a website checkout error",
    "a subscription the customer wants to cancel",
    "a product that doesn't match its description",
    "a delayed delivery with no tracking updates",
    "an app crash losing unsaved work",
    "a question about how a feature works (no complaint)",
    "a compliment with an incidental minor question",
    "a request to change a shipping address after ordering",
    "a coupon code that didn't apply at checkout",
    "a request for an invoice/receipt copy",
    "a report of unauthorized account access",
]

_SYSTEM_PROMPT = """You generate synthetic customer-support tickets and their gold-standard \
triage labels, for training a small model to reproduce typed-decision judgments. For each \
scenario hint given, invent a short, realistic customer support message (2-5 sentences, \
first-person from the customer) and label it accurately and consistently according to these \
three questions:

- refund_requested: true if the message is asking for a refund (explicitly or clearly implied), \
false otherwise.
- urgency: exactly one of "low", "medium", "high", "critical" -- how urgently this ticket needs a \
human response.
- category: exactly one of "billing", "shipping", "technical", "other".

Vary tone, length, and phrasing across examples. Some tickets should be ambiguous or borderline \
(e.g. urgency between two levels, or a category that could plausibly be two things) -- do not \
make every example a clear-cut case, real support tickets aren't. Respond with a JSON object: \
{"examples": [{"ticket_message": str, "refund_requested": bool, "urgency": str, "category": str}, ...]} \
with exactly as many entries as scenario hints given, in the same order."""


@dataclass
class CollectResult:
    artifact_path: str   # relative to ARTIFACT_STORE
    row_count: int
    from_ts: datetime
    to_ts: datetime


def _generate_batch(client, model: str, hints: list[str]) -> list[dict]:
    from google.genai import types

    user_prompt = "Scenario hints:\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(hints))
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=1.0,
    )
    response = client.models.generate_content(model=model, contents=user_prompt, config=config)
    content = response.text
    try:
        parsed = json.loads(content)
        examples = parsed["examples"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"LLM response wasn't the expected {{'examples': [...]}} shape: {content!r}") from exc
    if len(examples) != len(hints):
        logger.warning(f"requested {len(hints)} examples, LLM returned {len(examples)} -- using what came back")
    return examples


def _to_record(raw: dict) -> dict:
    """Converts one LLM-generated {ticket_message, refund_requested, urgency, category} object
    into the TypeSafe-shaped training record (state + questions + answers) described in this
    module's docstring. Raises KeyError/ValueError on a malformed LLM output rather than silently
    substituting a default -- a bad label should fail loudly during collection, not get baked
    into the dataset as if it were valid."""
    ticket_message = raw["ticket_message"]
    urgency = raw["urgency"]
    category = raw["category"]
    if urgency not in QUESTIONS["urgency"]["criteria"]:
        raise ValueError(f"LLM produced unknown urgency level {urgency!r}")
    if category not in QUESTIONS["category"]["criteria"]:
        raise ValueError(f"LLM produced unknown category {category!r}")
    return {
        "state": {"ticket_message": ticket_message},
        "questions": QUESTIONS,
        "answers": {
            "refund_requested": {"noul": bool(raw["refund_requested"])},
            "urgency": {"choice": urgency},
            "category": {"choice": category},
        },
    }


def collect(datasource_id: int, config: dict) -> CollectResult:
    from google import genai

    n_examples = int(config.get("n_examples", 500))
    batch_size = int(config.get("batch_size", 10))
    # Not os.getenv("LLM_MODEL", ...)'s "gemini-2.0-flash" default -- confirmed live (2026-09-18)
    # that model was retired ("no longer available... use models/gemini-3.6-flash"), so this
    # collector pins its own known-working default instead of inheriting a stale one; other
    # LLM_MODEL call sites (strategy/investigation.py, chat_agent.py, llm_condition.py) still
    # default to the retired name and will fail the same way until updated separately.
    llm_model = config.get("llm_model", os.getenv("LLM_MODEL", "gemini-3.6-flash"))
    seed = int(config.get("seed", 42))

    if n_examples < 1:
        raise ValueError("n_examples must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set -- required for the llm_typed_decisions collector")
    client = genai.Client(api_key=api_key)

    rng = random.Random(seed)
    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/typed_decisions.jsonl"
    artifact_abs = ARTIFACT_STORE / artifact_rel

    row_count = 0
    with open(artifact_abs, "w", encoding="utf-8") as f:
        remaining = n_examples
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            hints = [rng.choice(_SCENARIO_HINTS) for _ in range(this_batch)]
            raw_examples = _generate_batch(client, llm_model, hints)
            for raw in raw_examples:
                try:
                    record = _to_record(raw)
                except (KeyError, ValueError) as exc:
                    logger.warning(f"skipping malformed generated example: {exc}")
                    continue
                f.write(json.dumps(record) + "\n")
                row_count += 1
            remaining -= this_batch
            logger.info(f"llm_typed_decisions collector: generated {row_count}/{n_examples} examples so far")

    if row_count == 0:
        raise ValueError("collector produced zero usable examples -- every LLM response was malformed")

    from data.artifact_store import upload as _upload
    _upload(artifact_abs)

    now = datetime.now(timezone.utc)
    return CollectResult(artifact_path=artifact_rel, row_count=row_count, from_ts=now, to_ts=now)
