"""
classifier.py — LLM-based intent classifier (the Part 2 replacement for the
Part 1 TF-IDF baseline, evaluated on the exact same golden set for a fair
comparison).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

from agent.config import INTENT_LIST, WEAK_LABELED_PATH, GOLDEN_PATH
from agent.prompts import classifier_system_prompt, classifier_user_prompt, build_few_shot_block


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    reasoning: str
    raw_response: str
    parse_ok: bool


def _extract_json(text: str) -> dict:
    """LLMs occasionally wrap JSON in prose or fences despite instructions
    not to. Try straight parse first, then fall back to extracting the
    first {...} block."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model response: {text[:200]!r}")


@lru_cache(maxsize=1)
def _few_shot_examples() -> dict:
    """One short, real (weak-labeled) example per intent, excluding every
    golden-set id, for classifier calibration. Picked deterministically
    (shortest example >= 25 chars per class) rather than hand-curated, so
    this stays reproducible if the underlying data changes."""
    golden_ids = set(pd.read_csv(GOLDEN_PATH)["customer_tweet_id"])
    df = pd.read_csv(WEAK_LABELED_PATH)
    df = df[~df["customer_tweet_id"].isin(golden_ids)].copy()
    df["text_len"] = df["customer_text"].astype(str).str.len()
    df = df[df["text_len"] >= 25]

    examples = {}
    for intent in INTENT_LIST:
        subset = df[df["weak_intent"] == intent].sort_values("text_len")
        if len(subset):
            examples[intent] = subset.iloc[0]["customer_text"]
    return examples


def classify(client, customer_text: str) -> ClassificationResult:
    few_shot_block = build_few_shot_block(_few_shot_examples())
    system = classifier_system_prompt(few_shot_block)
    user = classifier_user_prompt(customer_text)

    response = client.complete(system=system, user=user)
    try:
        parsed = _extract_json(response.text)
        intent = parsed.get("intent", "").strip()
        if intent not in INTENT_LIST:
            # Model returned something outside the taxonomy -- fail safe
            # to the fallback intent rather than crash the pipeline, but
            # flag it so the harness can count these as classifier errors.
            return ClassificationResult(
                intent="general_chitchat_ack",
                confidence=0.0,
                reasoning=f"[PARSE FALLBACK] model returned out-of-taxonomy intent: {intent!r}",
                raw_response=response.text,
                parse_ok=False,
            )
        return ClassificationResult(
            intent=intent,
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=str(parsed.get("reasoning", "")),
            raw_response=response.text,
            parse_ok=True,
        )
    except (ValueError, TypeError) as e:
        return ClassificationResult(
            intent="general_chitchat_ack",
            confidence=0.0,
            reasoning=f"[PARSE FALLBACK] {e}",
            raw_response=response.text,
            parse_ok=False,
        )
