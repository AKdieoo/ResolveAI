"""
llm_judge.py — LLM-as-judge rubric scorer.

Scores a drafted reply on 5 dimensions (groundedness, correctness, tone,
actionability, safety) + an overall score, per agent/prompts.py:judge_system_prompt.

Uses the SAME model as the agent (agent/config.py:MODEL) by default. This
is a known limitation, not an oversight — see docs/decision_log.md #14 and
the report's "misleading headline number" section: a same-family judge can
share blind spots with the model it's grading. The harness supports passing
a different judge client/model if you have access to one, and the report
recommends that as a next step.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent.prompts import judge_system_prompt, judge_user_prompt, build_grounding_block
from agent.retriever import GroundingExample


@dataclass
class JudgeResult:
    groundedness: int
    correctness: int
    tone: int
    actionability: int
    safety: int
    overall: int
    rationale: str
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


def judge_reply(
    client,
    customer_text: str,
    intent: str,
    escalate: bool,
    grounding_examples,
    drafted_reply: str,
) -> JudgeResult:
    grounding_block = build_grounding_block(grounding_examples)
    system = judge_system_prompt()
    user = judge_user_prompt(customer_text, intent, escalate, grounding_block, drafted_reply)
    response = client.complete(system=system, user=user)

    try:
        parsed = _extract_json(response.text)
        scores = {
            k: int(parsed[k])
            for k in ["groundedness", "correctness", "tone", "actionability", "safety", "overall"]
        }
        for k, v in scores.items():
            if not (1 <= v <= 5):
                raise ValueError(f"score {k}={v} out of range 1-5")
        return JudgeResult(
            **scores,
            rationale=str(parsed.get("rationale", "")),
            raw_response=response.text,
            parse_ok=True,
        )
    except (ValueError, TypeError, KeyError) as e:
        # A judge score that fails to parse must NOT silently count as a
        # passing score anywhere downstream -- harness.py excludes
        # parse_ok=False rows from averaged judge metrics and reports the
        # exclusion count explicitly.
        return JudgeResult(
            groundedness=0, correctness=0, tone=0, actionability=0, safety=0, overall=0,
            rationale=f"[PARSE FAILURE - excluded from aggregate judge metrics] {e}",
            raw_response=response.text,
            parse_ok=False,
        )
