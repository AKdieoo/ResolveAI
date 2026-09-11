"""
llm_client.py — thin wrapper around the Anthropic API, plus a deterministic
MockLLMClient used ONLY for offline pipeline smoke-testing.

Why a mock client exists (read this before assuming the "mock" numbers
anywhere in results/ are real quality numbers — THEY ARE NOT):
This build environment has no internet access, so I (the model that wrote
this repo) cannot execute real Anthropic API calls myself. Rather than ship
untested plumbing, MockLLMClient lets the full pipeline (classify ->
retrieve -> draft -> escalate -> judge) run end-to-end with a rule-based
stand-in for the LLM, so I could verify the code actually runs, the JSON
parsing handles real-shaped output, and the harness produces well-formed
result files. Every place a mock-derived number appears, it is labeled
MOCK in the filename and in the results JSON. Real quality numbers require
running this with a real ANTHROPIC_API_KEY — see README.md.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Optional
from dataclasses import dataclass

from agent.config import MODEL, INTENT_LIST


@dataclass
class LLMResponse:
    text: str
    model: str
    mock: bool = False


class AnthropicLLMClient:
    """Real client. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = MODEL, max_tokens: int = 1024, temperature: float = 0.0):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required to run the real agent. "
                "Install with: pip install anthropic"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running the "
                "agent, e.g.: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(text=text, model=self.model, mock=False)


class GeminiLLMClient:
    """Real client using Google's free-tier Gemini API (Flash-class models).

    Added as an alternative to AnthropicLLMClient so this project can run
    on Google AI Studio's free tier (no credit card) instead of a paid
    Anthropic API key -- see docs/decision_log.md #16. The prompt
    contracts in agent/prompts.py are provider-agnostic (system + user
    string in, JSON string out), so swapping providers doesn't touch
    classifier.py / escalation.py / reply_drafter.py / eval/llm_judge.py
    at all -- only this file changes.

    IMPORTANT: use a non-"Preview" Flash model (e.g. gemini-2.5-flash or
    gemini-2.0-flash). Preview/Pro model names require billing enabled on
    the Google Cloud project even if your API key itself was free to
    create.
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' package is required to run the agent "
                "against Gemini. Install with: pip install google-genai"
            ) from e

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Export it "
                "before running the agent, e.g. in PowerShell: "
                '$env:GOOGLE_API_KEY = "AIza..."'
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": self.temperature,
            },
        )
        text = resp.text or ""
        return LLMResponse(text=text, model=self.model, mock=False)


class MockLLMClient:
    """Deterministic, rule-based stand-in for the Anthropic API.

    NOT a quality benchmark. Only exists so this repo's plumbing (JSON
    contracts between classifier/escalation/drafter/judge and the harness)
    could be exercised without network access. Uses the same keyword rules
    as weak_label.py, which is intellectually honest about its ceiling:
    it's a slightly-dressed-up regex, not a language model.
    """

    def __init__(self, seed: int = 42):
        self.model = "mock-rule-based-stand-in"
        self._rng = random.Random(seed)

    def complete(self, system: str, user: str) -> LLMResponse:
        # Route based on a marker each prompt-builder embeds in the system
        # prompt so the mock knows which "task" it's pretending to do.
        if "TASK=CLASSIFY" in system:
            text = self._mock_classify(user)
        elif "TASK=ESCALATE" in system:
            text = self._mock_escalate(user)
        elif "TASK=DRAFT_REPLY" in system:
            text = self._mock_draft(user)
        elif "TASK=JUDGE" in system:
            text = self._mock_judge(user)
        else:
            text = json.dumps({"error": "mock client does not recognize this task"})
        return LLMResponse(text=text, model=self.model, mock=True)

    # -- crude keyword routing, deliberately reusing weak_label.py's rules --
    def _mock_classify(self, user: str) -> str:
        m = re.search(r"CUSTOMER MESSAGE:\s*(.+?)\s*(?:\n\n|$)", user, re.S)
        msg = (m.group(1) if m else user).lower()
        rules = [
            ("family_plan_management", r"\bfamily\b|\binvite\b"),
            ("login_account_access", r"\blog\s?in\b|\bpassword\b|\blocked out\b|\bhack"),
            ("billing_payment_subscription", r"\bsubscription\b|\bpremium\b|\brefund\b|\bcharg|\bcancel\b|\bpayment\b|\bbill"),
            ("downloads_offline_issue", r"\bdownload|\boffline\b"),
            ("playback_streaming_issue", r"\bcrash|\bbuffer|\bstuck\b|\bskip|\berror\b|\bfreez|\bnot working\b"),
            ("content_missing_restricted", r"\bremoved\b|\bnot available\b|\bmissing\b|\brestrict|\bregion\b"),
            ("device_platform_support", r"\bxbox|\bandroid auto|\bcarplay|\bapp for\b|\bdevice\b"),
            ("feature_request_feedback", r"\bads?\b|\bplease add\b|\bfeature\b|\bsuggest"),
        ]
        intent = "general_chitchat_ack"
        for name, pat in rules:
            if re.search(pat, msg):
                intent = name
                break
        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.55,
                "reasoning": "[MOCK] keyword-rule stand-in, not a real LLM judgment.",
            }
        )

    def _mock_escalate(self, user: str) -> str:
        needs_account = bool(re.search(r"account|email|refund|payment|hack|stolen|compromise", user, re.I))
        return json.dumps(
            {
                "escalate": needs_account,
                "reason": "[MOCK] heuristic: message mentions account/payment/security terms."
                if needs_account
                else "[MOCK] heuristic: no account-specific or security terms detected.",
            }
        )

    def _mock_draft(self, user: str) -> str:
        return json.dumps(
            {
                "reply": (
                    "[MOCK DRAFT — not a real LLM output, for pipeline "
                    "smoke-testing only] Thanks for reaching out — we hear "
                    "you. Could you share a bit more detail so we can help "
                    "from here?"
                ),
                "used_grounding": True,
            }
        )

    def _mock_judge(self, user: str) -> str:
        return json.dumps(
            {
                "groundedness": 3,
                "correctness": 3,
                "tone": 3,
                "actionability": 3,
                "safety": 3,
                "overall": 3,
                "rationale": "[MOCK] fixed mid-scale scores, not a real judgment — for harness smoke-testing only.",
            }
        )


def get_client(mock: bool = False, provider: Optional[str] = None):
    """Returns the right LLM client.

    provider: "gemini", "anthropic", or None (auto-detect from whichever
    API key env var is set -- checks GOOGLE_API_KEY/GEMINI_API_KEY first,
    then ANTHROPIC_API_KEY, since Gemini's free tier is the more
    accessible default for anyone running this without a paid API key --
    see docs/decision_log.md #16).
    """
    if mock:
        return MockLLMClient()

    if provider == "gemini":
        return GeminiLLMClient()
    if provider == "anthropic":
        return AnthropicLLMClient()

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return GeminiLLMClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLMClient()

    raise RuntimeError(
        "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) for the "
        "free Gemini tier, or ANTHROPIC_API_KEY for the Anthropic API, "
        "before running the agent without --mock."
    )
