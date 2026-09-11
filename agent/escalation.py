"""
escalation.py — decides auto-handle vs. escalate-to-human, WITH a reason
string (per the assignment spec: "decide whether the message should be
auto-handled or escalated to a human — with a stated reason").

Design (see docs/decision_log.md #11): this is an LLM call, not a hardcoded
rule, deliberately given the EXACT policy text used to hand-label the
golden set (agent/config.py:ESCALATION_POLICY). The point of Part 2's
escalation module is to test whether an LLM can apply the same written
policy a human applied — not to test whether it can guess an unstated
policy. If the LLM can't hit reasonable agreement against a policy it was
literally given, no amount of rule engineering will fix that; if it can,
that's a meaningful result to report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent.prompts import escalation_system_prompt, escalation_user_prompt


@dataclass
class EscalationResult:
    escalate: bool
    reason: str
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


def decide_escalation(client, customer_text: str, intent: str) -> EscalationResult:
    system = escalation_system_prompt()
    user = escalation_user_prompt(customer_text, intent)
    response = client.complete(system=system, user=user)

    try:
        parsed = _extract_json(response.text)
        escalate = bool(parsed.get("escalate", False))
        reason = str(parsed.get("reason", "")).strip()
        if not reason:
            raise ValueError("empty reason string")
        return EscalationResult(
            escalate=escalate, reason=reason, raw_response=response.text, parse_ok=True
        )
    except (ValueError, TypeError) as e:
        # Fail SAFE: an unparseable escalation decision defaults to
        # escalate=True. Silently defaulting to auto-handle on a parse
        # failure would be the wrong failure mode for a support agent --
        # better to over-route to a human than to auto-handle blind.
        return EscalationResult(
            escalate=True,
            reason=f"[PARSE FALLBACK - defaulted to escalate for safety] {e}",
            raw_response=response.text,
            parse_ok=False,
        )
