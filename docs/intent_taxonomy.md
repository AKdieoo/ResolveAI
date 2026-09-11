# Intent Taxonomy — Spotify Support Agent

Built by reading ~150 raw customer->agent pairs sampled from the extracted
`spotify_pairs.csv` (43,031 real threads) and clustering the recurring
problem types by hand. This is **not** borrowed from Banking77 — Banking77 is
a banking domain and its 77 labels don't transfer to a music-streaming
product. Banking77 is used only as a *methodological reference* (how a
tight, mutually-exclusive intent set with clear decision boundaries should
look) — see decision log #2.

## The 9 intents

| # | Intent | Description | Typical resolution style |
|---|--------|--------------|---------------------------|
| 1 | `login_account_access` | Can't log in, password reset, locked out, account recovery, linked social account broken | Usually requires DM to verify identity → escalate |
| 2 | `family_plan_management` | Add/remove/invite family plan members, address verification for family plan, member eligibility | Requires account-specific lookup → escalate |
| 3 | `billing_payment_subscription` | Pricing questions, payment method issues, failed charges, cancel/downgrade/upgrade, promo eligibility, refunds | Mixed — general pricing can be answered directly; account-specific billing → escalate |
| 4 | `playback_streaming_issue` | App crashes, buffering, "loading" stuck, song skipping, audio quality, specific error messages | Often self-serve (restart/reinstall/check connection) → can auto-handle first-line troubleshooting |
| 5 | `downloads_offline_issue` | Downloaded songs disappearing, offline mode not working, storage issues | Usually self-serve steps exist → can auto-handle |
| 6 | `content_missing_restricted` | Song/album/artist removed or unavailable, regional licensing restriction, search not finding content | Mostly informational (licensing explanation) → can auto-handle |
| 7 | `device_platform_support` | "Is there an app for X device/OS", platform availability requests, device-specific compatibility questions | Informational → can auto-handle |
| 8 | `feature_request_feedback` | Ad complaints, feature suggestions, general product opinions, not a broken-thing report | Acknowledgement only → can auto-handle |
| 9 | `general_chitchat_ack` | Thanks, sign-offs, non-actionable pleasantries, or messages with no discernible support need | Acknowledgement only → can auto-handle |

## Design notes
- Kept to 9 (not 77-style granularity) because Twitter support messages are
  short and multi-topic; more classes would mean lower inter-label agreement
  for a comparable human-labeling effort in the time available.
- Intents 1–2 are *inherently* privacy-gated (Spotify's own agents route
  these to DM ~90%+ of the time in the raw data) — this directly informs the
  escalation policy in Part 2, not just the classifier.
- "Escalate" is deliberately **not** modeled as its own intent — it's a
  downstream decision (see `docs/decision_log.md` #5) because the same
  intent can be auto-handleable or not depending on whether it needs
  account-specific data.
