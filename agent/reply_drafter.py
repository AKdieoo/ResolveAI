"""
reply_drafter.py — drafts a reply grounded in real historical resolutions
retrieved by agent/retriever.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List

from agent.prompts import drafter_system_prompt, drafter_user_prompt, build_grounding_block
from agent.retriever import GroundingExample


@dataclass
class DraftResult:
    reply: str
    used_grounding: bool
    raw_response: str
    parse_ok: bool


def _extract_json(text: str) -> dict:
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


def draft_reply(
    client,
    customer_text: str,
    intent: str,
    escalate: bool,
    grounding_examples: List[GroundingExample],
) -> DraftResult:
    grounding_block = build_grounding_block(grounding_examples)
    system = drafter_system_prompt()
    user = drafter_user_prompt(customer_text, intent, escalate, grounding_block)
    response = client.complete(system=system, user=user)

    try:
        parsed = _extract_json(response.text)
        reply = str(parsed.get("reply", "")).strip()
        if not reply:
            raise ValueError("empty reply string")
        return DraftResult(
            reply=reply,
            used_grounding=bool(parsed.get("used_grounding", False)),
            raw_response=response.text,
            parse_ok=True,
        )
    except (ValueError, TypeError) as e:
        return DraftResult(
            reply=(
                "Thanks for reaching out — could you DM us your account's "
                "email address so we can take a closer look? [FALLBACK "
                "REPLY: drafter response failed to parse]"
            ),
            used_grounding=False,
            raw_response=response.text,
            parse_ok=False,
        )
