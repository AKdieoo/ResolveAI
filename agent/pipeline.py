"""
pipeline.py — wires classifier -> retriever -> escalation -> drafter into
one agent.run(customer_text) call. This IS the deliverable described in
the assignment: "build an AI support agent that can classify, draft a
grounded reply, and decide auto-handle vs escalate with a reason."

Order of operations matters and is deliberate (decision log #13):
1. Classify intent first.
2. Decide escalation SECOND, using the classified intent (not raw text
   alone) — the escalation policy is partly intent-conditioned.
3. Retrieve grounding examples, intent-filtered using the CLASSIFIED
   intent (not the unknown true intent — this is the realistic setting).
4. Draft the reply last, given intent + escalation decision + grounding,
   so an escalate=true case gets a reply whose only job is the DM handoff
   (see drafter_system_prompt), not a wrong attempt to solve it in-tweet.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from agent.config import GOLDEN_PATH, RESULTS_DIR
from agent.llm_client import get_client
from agent.retriever import Retriever
from agent.classifier import classify, ClassificationResult
from agent.escalation import decide_escalation, EscalationResult
from agent.reply_drafter import draft_reply, DraftResult


@dataclass
class AgentResult:
    customer_text: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    intent_parse_ok: bool
    escalate: bool
    escalate_reason: str
    escalate_parse_ok: bool
    reply: str
    reply_used_grounding: bool
    reply_parse_ok: bool
    n_grounding_examples: int
    grounding_ids: str  # comma-separated customer_tweet_ids, for audit


class SpotifySupportAgent:
    def __init__(self, client=None, mock: bool = False, exclude_ids: Optional[set] = None):
        self.client = client or get_client(mock=mock)
        self.retriever = Retriever(exclude_ids=exclude_ids)

    def run(self, customer_text: str, k_grounding: int = 3) -> AgentResult:
        cls: ClassificationResult = classify(self.client, customer_text)
        esc: EscalationResult = decide_escalation(self.client, customer_text, cls.intent)
        grounding = self.retriever.retrieve(
            customer_text, k=k_grounding, intent_filter=cls.intent
        )
        draft: DraftResult = draft_reply(
            self.client, customer_text, cls.intent, esc.escalate, grounding
        )

        return AgentResult(
            customer_text=customer_text,
            intent=cls.intent,
            intent_confidence=cls.confidence,
            intent_reasoning=cls.reasoning,
            intent_parse_ok=cls.parse_ok,
            escalate=esc.escalate,
            escalate_reason=esc.reason,
            escalate_parse_ok=esc.parse_ok,
            reply=draft.reply,
            reply_used_grounding=draft.used_grounding,
            reply_parse_ok=draft.parse_ok,
            n_grounding_examples=len(grounding),
            grounding_ids=",".join(str(g.customer_tweet_id) for g in grounding),
        )


def main():
    parser = argparse.ArgumentParser(description="Run the Spotify support agent on one message")
    parser.add_argument("--text", type=str, help="Customer message to run the agent on")
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient (no API key / no network needed)")
    parser.add_argument(
        "--golden-row", type=int, default=None,
        help="Instead of --text, run on row N (0-indexed) of the golden set, for a quick spot-check",
    )
    args = parser.parse_args()

    if args.golden_row is not None:
        golden = pd.read_csv(GOLDEN_PATH)
        row = golden.iloc[args.golden_row]
        text = row["customer_text"]
        print(f"[golden row {args.golden_row}] true_intent={row['true_intent']} escalate={row['escalate']}")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide --text or --golden-row")

    golden_ids = set(pd.read_csv(GOLDEN_PATH)["customer_tweet_id"])
    agent = SpotifySupportAgent(mock=args.mock, exclude_ids=golden_ids)
    result = agent.run(text)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
