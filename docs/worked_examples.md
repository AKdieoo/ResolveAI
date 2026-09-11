# Worked Examples — Full Agent Pipeline

**Read this before quoting anything below as a quality result.**

I do not have internet access in this build environment, so I cannot
execute real Anthropic API calls (see `agent/llm_client.py` for the
technical detail — `MockLLMClient` exists for offline plumbing tests only,
and its output is fixed dummy JSON, not a language model's judgment).

Everything on this page is:
- **Real** input (an actual golden-set row) and **real** retrieval (the
  grounding examples below were produced by actually running
  `agent/retriever.py` against the 43k-pair pool, TF-IDF cosine similarity,
  no API needed — these are genuine outputs, not invented).
- **Hand-authored** classify/escalate/draft/judge outputs, written by me in
  the exact JSON shape the real pipeline produces, to show what a
  reasonable agent run *should* look like and to sanity-check the prompts
  in `agent/prompts.py` actually have enough information to produce a good
  answer. **These are not live model outputs.** Run
  `python -m agent.pipeline --golden-row N` with a real `ANTHROPIC_API_KEY`
  to get the real thing.

---

## Example 1 — escalate=True, security/account-access

**Customer message** (`customer_tweet_id=499273`):
> @SpotifyCares you have locked my friend out of her account and she will
> not stop bothering me at work. please help her change her payment info.
> Her name is Riley Barsella and she needs some tunes!!!!!

**Real retrieved grounding** (top-2, intent-filtered `login_account_access`):
1. "@SpotifyCares I'm locked out of my account 😟" → "We see. Could you DM
   us your account's email address and username? We'll take a look
   backstage"
2. "@SpotifyCares I'm locked out of my Spotify account. Who do I need to
   contact?" → "Hey Lindsay, help's here! Can you DM us your account's
   email address? We'll take a look backstage"

**Classifier (hand-simulated):**
```json
{"intent": "login_account_access", "confidence": 0.86, "reasoning": "Locked out of account and payment info change requested is a login/account-access request, not primarily billing."}
```

**Escalation (hand-simulated):**
```json
{"escalate": true, "reason": "Resolving this requires account-specific data (identity verification + payment info change) — policy clause 1 applies."}
```

**Drafted reply (hand-simulated, grounded in both retrieved examples' DM-handoff pattern):**
> Hey! Sorry to hear that. Payment info changes need to come from the
> account holder directly for security — could you have Riley DM us her
> account's email address? We'll take it from there.

**Judge (hand-simulated):**
```json
{"groundedness": 5, "correctness": 5, "tone": 4, "actionability": 5, "safety": 5, "overall": 5, "rationale": "Correctly refuses to act on a third party's behalf, matches the real DM-handoff pattern, states why."}
```
**True golden label:** `login_account_access`, `escalate=True` — matches.

---

## Example 2 — escalate=False, informational (licensing)

**Customer message** (`customer_tweet_id=2631683`):
> @3493 @115888 @SpotifyCares why isn't Red Velvet's Perfect Velvet album
> available yet, please fix :((( @15672

**Real retrieved grounding** — this one is a near-exact duplicate case,
which is a good sign the retriever works, and a useful reminder that
Twitter support datasets contain a lot of near-duplicate questions:
1. "Why Red Velvet/Perfect Velvet album is not available in my country
   Turkey. Please I need to listen th[is]" → "Hey Ezgi! We'll have it
   available to you as soon as it's available to us!"
2. "@SpotifyCares Why i cant find perfect velvet album😌" → "Hi! We'd love
   to have all of Red Velvet's stuff available, but we have some info
   about content..."

**Classifier (hand-simulated):**
```json
{"intent": "content_missing_restricted", "confidence": 0.91, "reasoning": "Specific album reported unavailable — licensing/availability, not a bug report."}
```

**Escalation (hand-simulated):**
```json
{"escalate": false, "reason": "This is a general licensing/availability question answerable without account-specific data — no policy clause requires escalation."}
```

**Drafted reply (hand-simulated):**
> Hey! We hear you — we'll have it up as soon as it's cleared on our end.
> No ETA we can share yet, but keep an eye out!

**Judge (hand-simulated):**
```json
{"groundedness": 5, "correctness": 4, "tone": 5, "actionability": 3, "safety": 5, "overall": 4, "rationale": "Matches the real pattern well; actionability capped at 3 because it gives no timeline or way to get notified, same limitation the real historical replies have."}
```
**True golden label:** `content_missing_restricted`, `escalate=False` — matches. Note the **TF-IDF baseline got this one wrong** (predicted `general_chitchat_ack` — see `docs/report.md` failure analysis #1), which is a real, useful side-by-side data point even from just one example.

---

## Example 3 — escalate=True, family plan

**Customer message** (`customer_tweet_id=1876241`):
> @SpotifyCares hello i really need your help i cannot seem to see the
> names of all the accounts in my family premium package. Need to manage

**Real retrieved grounding:**
1. "Whatever you are doing to manage your member accounts and family plan,
   it's bullocks @115888" → "Hi there, we're sorry you feel that way.
   Don't worry, we'll take a look backstage! Can you D[M]..."
2. "@SpotifyCares I need help!!! Can [you] please contact me regarding my
   family premium" → "Hey Elisa, help's here! Could you send us a DM with
   some more info on what's happening?"

**Classifier (hand-simulated):**
```json
{"intent": "family_plan_management", "confidence": 0.88, "reasoning": "Explicitly about managing family plan member accounts."}
```

**Escalation (hand-simulated):**
```json
{"escalate": true, "reason": "Viewing/managing family plan member accounts requires account-specific data — policy clause 1 applies."}
```

**Drafted reply (hand-simulated):**
> Hey there! We can help sort that out. Could you DM us your account's
> email address? We'll take a look backstage.

**Judge (hand-simulated):**
```json
{"groundedness": 5, "correctness": 5, "tone": 5, "actionability": 5, "safety": 5, "overall": 5, "rationale": "Textbook match to both grounding examples' resolution pattern."}
```
**True golden label:** `family_plan_management`, `escalate=True` — matches.

---

## Example 4 — a harder, more ambiguous case (illustrating a likely failure mode)

**Customer message** (`customer_tweet_id=574261`):
> @115888 what's a token error

**Real retrieved grounding** — note the top hit is about a *different*
kind of token error (CSRF on a payment form), which is exactly the kind of
retrieval noise the report's failure analysis flags:
1. "@SpotifyCares I'm trying to update my card info but keep getting a
   CSRF token error." → "Hi there! Can you DM us your account's email
   address or username? We'll take a look backstage"
2. "@117168 and another error! Can't get into my music" → "Hey there!
   Help's here. Can you let us know the iOS and app versions you're using?"

**Classifier (hand-simulated) — plausible but under-informed:**
```json
{"intent": "playback_streaming_issue", "confidence": 0.42, "reasoning": "Generic error-message report with no other context; low confidence because 'token error' alone doesn't specify billing vs. playback vs. login."}
```

**Escalation (hand-simulated):**
```json
{"escalate": true, "reason": "Message alone is too ambiguous to resolve without more detail; the real historical reply for the closest grounding example also went straight to a DM handoff — policy clause 3 applies."}
```

**Drafted reply (hand-simulated):**
> Hey! Could you DM us your account's email and a screenshot of the exact
> error? We'll take a closer look.

**Judge (hand-simulated):**
```json
{"groundedness": 4, "correctness": 4, "tone": 4, "actionability": 4, "safety": 5, "overall": 4, "rationale": "Reasonable fallback response for genuinely ambiguous input; doesn't guess at a wrong specific fix."}
```
**True golden label:** `playback_streaming_issue`, `escalate=False`. **This
is a case where the hand-simulated agent's escalate call plausibly
disagrees with the golden label** — the message is genuinely
under-specified, and this is exactly the kind of single-turn-context
limitation flagged in decision log #4 and the golden set methodology
(the real historical reply *did* ask for a DM, but the human labeler's
policy application on this fragment leaned auto-handle). This is called
out, not hidden, in `docs/report.md`'s failure analysis.

---

## Example 5 — escalate=False, feature feedback

**Customer message** (`customer_tweet_id=238936`):
> Suggestion for the @115888 marketing team: keep the "happy death day"
> horror ads off the "peace" playlists plz

**Real retrieved grounding:**
1. "@115888 please get more sponsors I'm tired of hearing the same 'Happy
   Death Day' Ad 40x a day" → "Hey, thanks for your feedback - we'll pass
   it on. For now, check out..."
2. "um @115888 the Happy Death Day ad playing every ten minutes where you
   can hear someone getting stabb[ed]" → "Hi Nicole. Thanks for bringing
   this up. We didn't mean to cause any offense. We'll pass the [feedback
   on]"

**Classifier (hand-simulated):**
```json
{"intent": "feature_request_feedback", "confidence": 0.83, "reasoning": "Ad-placement complaint framed as a suggestion, not a technical fault."}
```

**Escalation (hand-simulated):**
```json
{"escalate": false, "reason": "Feedback acknowledgement requires no account-specific data or security handling — auto-handle."}
```

**Drafted reply (hand-simulated):**
> Hey! We didn't mean to cause any offense — we'll pass this on to the
> right team. Thanks for flagging it!

**Judge (hand-simulated):**
```json
{"groundedness": 5, "correctness": 5, "tone": 5, "actionability": 4, "safety": 5, "overall": 5, "rationale": "Near-verbatim match to the real historical resolution pattern for this exact ad complaint."}
```
**True golden label:** `feature_request_feedback`, `escalate=False` — matches.

---

## What these 5 examples are (and are not) evidence of

They **are** evidence that: the prompts in `agent/prompts.py` contain
enough information to produce a reasonable answer by inspection, the
retriever surfaces genuinely relevant grounding (including a literal
duplicate case in Example 2), and the JSON contract between pipeline
stages is coherent end-to-end.

They are **not** evidence of the agent's real accuracy, hallucination
rate, or judge-quality — that requires `python -m eval.harness` run for
real against `ANTHROPIC_API_KEY`, which produces
`results/llm_agent_results.json` in the exact same format as
`results/tfidf_baseline_results.json` for a fair, automatic comparison.
