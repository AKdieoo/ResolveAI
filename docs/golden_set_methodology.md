# Golden Evaluation Set — Methodology

**File:** `data/golden/golden_set.csv` — 200 rows.

## Sampling
Source: `data/processed/spotify_weak_labeled.csv` (43,031 real SpotifyCares
customer→agent tweet pairs, extracted from the raw Twitter Customer Support
dataset, weak-labeled by regex — see `src/weak_label.py`).

Sampling was stratified, not purely random, because a purely random sample
would be ~60% `general_chitchat_ack` (the true traffic distribution) and
leave too few examples of minority intents like `family_plan_management` to
evaluate a classifier's per-class behavior with any statistical stability.
Procedure (`src/build_golden_set.py` input construction, seed=42):
1. Take up to 16 examples per non-chitchat weak-label class (8 classes → up
   to 128 rows).
2. Top up to 200 total with a uniform random sample from the remaining pool
   (this is what brings in the realistic proportion of chitchat/ack messages
   and any long-tail cases the stratified step missed).
3. Shuffle.

This intentionally over-represents minority intents relative to true traffic
share — the golden set is for **measuring per-intent quality**, not for
estimating traffic mix. Anyone using it to say "37% of Spotify support
tickets are about playback issues" would be wrong; see the "what's
misleading about the headline number" section of the report.

## Labeling
Each of the 200 examples was read individually — customer message plus the
*real* historical agent reply shown as context (not as ground truth to copy,
but as a signal for what actually happened, e.g. whether the real agent
asked for a DM) — and hand-assigned two labels:

1. **`true_intent`** — one of the 9 classes in `docs/intent_taxonomy.md`.
   The cheap regex weak-labeler agreed with the hand label on only **63%**
   of the 200 examples. The other 37% is exactly the kind of error you'd
   expect from keyword rules: a downloads-related word appearing in an
   otherwise off-topic sentence, multi-topic tweets, sarcasm, and
   context-dependent fragments ("No difference ☹️") that only make sense
   as a reply to an earlier turn the pair-extraction didn't capture (see
   decision log #4 on the single-turn limitation).

2. **`escalate`** — True/False, using a fixed policy applied consistently
   across all 200 (not vibes-based per example):
   > Escalate if resolving the message requires account-specific data
   > (an email/username handoff, a payment/refund lookup) **or** the issue
   > is safety/security-sensitive (compromised account) **or** the real
   > agent skipped generic troubleshooting and went straight to a DM/ticket
   > on a similar case. Otherwise auto-handle (informational answers,
   > generic self-serve troubleshooting steps, feedback acknowledgement).

   36.5% of the golden set was labeled escalate=True. This threshold is a
   judgment call, not a ground truth extracted from Spotify — see decision
   log #7 for why a stricter or looser policy would move this number a lot.

## Known limitations of this golden set (stated up front, not buried)
- **Single labeler.** All 200 labels were produced by one reviewer (the
  author of this repo). No second annotator computed agreement (kappa) on
  the human labels themselves — only weak-label-vs-hand agreement is
  reported above. This is flagged again in the report's "misleading
  headline number" section.
- **Single-turn context.** Labels are based on one customer message + the
  immediate historical reply, not the full multi-turn thread. A few
  examples (rows with follow-up fragments like "This!" or "Should be the
  US") are genuinely ambiguous without earlier thread context; they were
  labeled `general_chitchat_ack` / `escalate=False` as a conservative
  default rather than guessed at, and are called out in the failure
  analysis in the report.
- **One brand.** All examples are Spotify-specific; the taxonomy and
  escalation policy would need re-deriving for another brand (see decision
  log #2).
