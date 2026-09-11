"""
config.py — shared constants. One source of truth for the intent list so
the classifier prompt, the escalation module, and the eval harness can't
drift out of sync with docs/intent_taxonomy.md.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Anthropic model used for every LLM call in this project (classifier,
# drafter, escalation reasoning, and the judge). Pinned to one model so
# results are comparable across the pipeline and the eval harness.
MODEL = "claude-sonnet-4-6"

# Keep this in sync with docs/intent_taxonomy.md — it is the same 9-class
# taxonomy used by the Part 1 baselines and the golden set, on purpose:
# the whole point of Part 2 is a fair, same-classes, same-golden-set
# comparison against the trivial and TF-IDF baselines.
INTENTS = {
    "login_account_access": (
        "Can't log in, password reset, locked out, account recovery, "
        "linked social account broken."
    ),
    "family_plan_management": (
        "Add/remove/invite family plan members, address verification for "
        "family plan, member eligibility."
    ),
    "billing_payment_subscription": (
        "Pricing questions, payment method issues, failed charges, "
        "cancel/downgrade/upgrade, promo eligibility, refunds."
    ),
    "playback_streaming_issue": (
        "App crashes, buffering, 'loading' stuck, song skipping, audio "
        "quality, specific error messages."
    ),
    "downloads_offline_issue": (
        "Downloaded songs disappearing, offline mode not working, storage "
        "issues."
    ),
    "content_missing_restricted": (
        "Song/album/artist removed or unavailable, regional licensing "
        "restriction, search not finding content."
    ),
    "device_platform_support": (
        "'Is there an app for X device/OS', platform availability "
        "requests, device-specific compatibility questions."
    ),
    "feature_request_feedback": (
        "Ad complaints, feature suggestions, general product opinions, "
        "not a broken-thing report."
    ),
    "general_chitchat_ack": (
        "Thanks, sign-offs, non-actionable pleasantries, or messages with "
        "no discernible support need."
    ),
}

INTENT_LIST = list(INTENTS.keys())

# The exact escalation policy from docs/golden_set_methodology.md, restated
# here so the LLM escalation module is judged against the SAME rule the
# human labeler used on the golden set — not a rule I made up separately
# for Part 2. See decision log #11.
ESCALATION_POLICY = """Escalate if resolving the message requires account-specific data \
(an email/username handoff, a payment/refund lookup) OR the issue is \
safety/security-sensitive (compromised account) OR generic troubleshooting \
has already failed / doesn't apply and account-specific investigation is \
the only remaining path. Otherwise auto-handle (informational answers, \
generic self-serve troubleshooting steps, feedback acknowledgement)."""

PAIRS_PATH = BASE / "data" / "processed" / "spotify_pairs.csv"
WEAK_LABELED_PATH = BASE / "data" / "processed" / "spotify_weak_labeled.csv"
GOLDEN_PATH = BASE / "data" / "golden" / "golden_set.csv"
RESULTS_DIR = BASE / "results"
