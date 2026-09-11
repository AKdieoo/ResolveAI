# Decision Log

Plain list of non-obvious calls made while building this, and why. Numbered
so the report can reference them. Part 2 will append #10 onward.

1. **Brand: Spotify (SpotifyCares), not Amazon/Apple/Uber.** Amazon and
   Apple have far more volume (170k/107k agent tweets) but span too many
   product categories (retail, devices, cloud, media) to have a clean small
   intent taxonomy. Spotify is a single-product support surface with real
   back-and-forth resolution threads (not just "please call us"), and 43k
   pairs is plenty for both weak-label training and a 200-example golden set.

2. **Banking77 was used as a methodology reference, not a label source.**
   Its domain (banking) doesn't transfer to music streaming. What *did*
   transfer: the idea of a tight, mutually-exclusive intent set with
   explicit decision boundaries between similar-sounding classes, built by
   reading real examples rather than guessing from the brand's marketing
   copy.

3. **Weak labeling (regex/keyword rules) for the 43k-row training set,
   hand labeling only for the 200-row golden set.** Hand-labeling 43k
   tweets isn't feasible; hand-labeling 0 tweets means you can't trust any
   metric. The split is deliberate: weak labels are *only* used to train the
   TF-IDF baseline (and, in Part 2, possibly as few-shot examples) — they
   are never used as ground truth for evaluation.

4. **Pair extraction is single-turn, not full-thread.** Each training/golden
   example is (customer tweet, first direct agent reply), not the full
   multi-turn thread. This is a real limitation — a few golden examples are
   only interpretable with earlier thread context the extraction didn't
   capture (e.g. "No difference ☹️"). Chose this over full-thread
   reconstruction because: (a) the agent's job in this project is to handle
   the *first* incoming message, not referee a whole thread, and (b) thread
   reconstruction on this dataset is genuinely messy (branching replies,
   quote-tweets, multiple customers CC'd) and would have eaten the time
   budget for a marginal gain. Documented as a limitation, not hidden.

5. **Escalation is a policy applied to (intent, content), not its own
   intent class.** The same intent (e.g. `billing_payment_subscription`)
   can be auto-handleable ("how do I apply the student discount" —
   informational) or must escalate ("someone's using my grandmother's card"
   — needs account verification). Modeling escalate as an 10th intent would
   have hidden this and produced a taxonomy that doesn't match how the
   decision actually needs to be made.

6. **Golden set is stratified-then-topped-up, not purely random.** A purely
   random 200-sample would have ~120 chitchat examples and single digits for
   several real support intents — not enough to say anything meaningful
   about per-class performance. The tradeoff (documented explicitly) is that
   the golden set's intent distribution is NOT representative of true
   traffic mix, so it must not be used to estimate volume/prevalence.

7. **Escalation policy is a judgment call, encoded as an explicit written
   rule, applied consistently — not "vibes per example."** The 36.5%
   escalate rate is a direct function of where that rule draws the line
   (e.g., treating all payment/account-lookup cases as escalate=True). A
   business could reasonably draw it stricter (escalate more, safer, more
   human load) or looser (escalate less, cheaper, more risk). This number
   is flagged in the report as one of the "misleading headline number" risks.

8. **Baseline escalate labels come from a DM-mention regex on the
   *historical* agent reply, not from the golden hand labels.** This keeps
   the "simple" baseline honestly weak-label-trained end-to-end (same
   standard as the intent classifier) rather than accidentally peeking at
   golden-set-quality signal for one of its two heads.

9. **Leakage check: golden-set tweet IDs are explicitly excluded from
   baseline training data**, not just "sampled before training" (which
   would still risk accidental overlap if scripts are re-run out of order).
   `tfidf_baseline.py` filters by ID set at train time as a hard guarantee.

10. **Text cleaning strips agent sign-off codes (e.g. "/LS") and URLs, but
    keeps @mentions and emoji.** Sign-offs are agent-identifier noise with
    no semantic content — stripping them measurably reduces sparse,
    meaningless TF-IDF features. @mentions and emoji were kept because they
    sometimes carry signal (e.g. which brand is being addressed, sentiment).

## Part 2 additions

11. **Escalation is a single LLM call given the EXACT written policy used
    to hand-label the golden set** (`agent/config.py:ESCALATION_POLICY`),
    not a separately-invented rule or a second classifier. The reason: if
    the escalation module were graded against a policy it never saw, a
    disagreement would be uninformative (is it wrong, or just applying a
    different-but-reasonable policy?). Giving it the same policy the human
    labeler used turns "does it match the golden label" into an actual test
    of instruction-following + judgment, not policy-guessing.

12. **Grounding retrieval evaluation caught and fixed a circular-metric
    bug before shipping.** The first version of
    `agent/retriever.py:evaluate_retrieval_quality` filtered the candidate
    pool to `weak_intent == true_intent` and then checked whether retrieved
    results' `weak_intent == true_intent` — which is true by construction
    of the filter, not evidence of anything (it returned a meaningless
    100% "hit rate"). The fixed version measures retrieval on the FULL,
    unfiltered pool using plain text similarity only, giving a real,
    non-trivial number (52.5% weak-intent-match@3 on the golden set — see
    `results/retrieval_eval_results.json`). Documented here because it's a
    realistic mistake ("evaluate the thing after applying the shortcut you
    want to evaluate") worth naming, not quietly fixing.

13. **Pipeline order is classify → escalate → retrieve → draft, not
    retrieve-first.** Retrieval is intent-filtered using the CLASSIFIED
    intent (the only thing available at inference time — the model never
    sees true_intent), and the drafter is told the escalation decision
    before drafting so an escalate=true case gets a reply whose only job is
    the DM handoff, rather than the drafter independently attempting to
    solve an issue that's already been flagged as needing a human.

14. **The LLM-as-judge uses the SAME model as the agent by default**
    (`agent/config.py:MODEL`, both `claude-sonnet-4-6`). This is a real
    limitation, not an oversight: a same-family judge can share blind
    spots with the model it's grading (e.g. both might rate an
    unsupported-but-plausible-sounding claim as "grounded"). The harness
    (`eval/harness.py`) accepts any judge client, so swapping in a
    different model/provider as the judge is a one-line change — flagged
    as a concrete next step in `docs/report.md`.

15. **Human-agreement checking is shipped as a complete, runnable protocol
    (`eval/human_agreement.py`), not fabricated numbers.** I have no
    internet access in this build environment and no second human rater
    available to produce a real agreement statistic myself, and my own
    ratings wouldn't be an independent check since I wrote the judge
    prompt. Rather than invent a number, the script builds a blinded,
    stratified rating sheet from real harness output and computes exact
    agreement / agreement-within-1 / weighted Cohen's kappa / Pearson r
    once a human fills it in — this is the same "real numbers only, ship
    the tool not a faked result" standard applied throughout Part 1.

16. **`agent/llm_client.py` supports Gemini (free tier) as well as the
    Anthropic API, auto-detected from whichever API key env var is set.**
    Added after the fact, for a practical reason: not everyone running
    this repo has a funded Anthropic API key, and Google AI Studio's
    Flash-class models (non-"Preview" ones) have a genuinely free tier
    with no card required. `get_client()` checks `GOOGLE_API_KEY`/
    `GEMINI_API_KEY` first, then `ANTHROPIC_API_KEY`, so the rest of the
    pipeline (`classifier.py`, `escalation.py`, `reply_drafter.py`,
    `eval/llm_judge.py`) needed zero changes — they only ever call
    `client.complete(system, user)` and parse a JSON string back,
    regardless of which provider produced it. This is a real trade-off,
    not a free win: Gemini Flash and Claude Sonnet are different models
    with different failure modes, so results aren't directly comparable
    across a Gemini run and a Claude run without noting which was used —
    the results JSON and CSV filenames should be annotated with the
    provider if both get run.
