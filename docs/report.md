# Report — Spotify AI Support Agent

*Brand: Spotify (SpotifyCares). Dataset: Customer Support on Twitter
(Kaggle, thoughtvector/customer-support-on-twitter), 43,031 extracted
Spotify customer↔agent pairs. Golden set: 200 hand-labeled examples. Full
methodology for everything referenced here is in `docs/decision_log.md`
(15 entries) and `docs/golden_set_methodology.md`.*

---

## 1. Problem framing

**What "good" means for this agent.** A Spotify Twitter support reply has
one job in the moment it's sent: either (a) resolve a self-serve issue
with a correct, on-brand answer, or (b) correctly recognize it can't be
resolved without account-specific data and hand off to a human — without
guessing at a wrong fix, promising something Spotify can't promise, or
handling something security-sensitive by itself. "Good" is therefore not
"sounds plausible" — it's **grounded** (matches how Spotify actually
resolves that type of issue), **safe** (never resolves what should be
escalated), and **on-brand** (Spotify's real support voice is brief,
informal, warm — not corporate boilerplate).

**What I chose not to build:**
- *Full multi-turn thread handling.* The agent handles the first incoming
  message, not an ongoing conversation (decision log #4). A few golden
  examples are genuinely ambiguous without earlier thread context — this
  is a real, stated limitation, not solved in Part 2.
- *Fine-tuning.* Everything here is prompting + retrieval-grounding on a
  general-purpose model, not a fine-tuned classifier or generator. Given
  the dataset size (43k weak-labeled, 200 hand-labeled) and time budget,
  a good retrieval-grounded prompt was judged more reliable than a
  fine-tune that would inherit the weak-label noise directly into model
  weights.
- *A second, independent intent taxonomy for escalation.* Escalation is a
  policy over (intent, content), not its own class — see decision log #5.
- *Volume/prevalence estimation.* The golden set is stratified, not
  traffic-representative (decision log #6) — nothing here should be read
  as "X% of Spotify tickets are about Y."

---

## 2. Results vs. baselines

All rows below are evaluated identically: same 200-row golden set, same
intent accuracy/F1 and escalate accuracy/precision/recall/F1 definitions
(eval/harness.py reuses the exact metric calls from
baselines/tfidf_baseline.py), same train/golden leakage protection
(golden-set customer_tweet_ids excluded from every training/retrieval
pool -- decision log #9).

| Metric | Trivial | TF-IDF + LogReg | LLM Agent (Gemini 3.5 Flash-Lite) |
|---|---|---|---|
| Intent accuracy (9-way, n=200) | 9.5% | 65.0% | **85.0%** |
| Intent F1 (macro) | -- | 66.1% | **85.7%** |
| Escalate accuracy | 63.5% | 83.5% | **85.0%** |
| Escalate F1 | 0% | 77.6% | 75.4% |
| Reply quality (LLM judge, 1-5) | n/a | n/a | 4.95 avg |

These are real, complete numbers -- all 200 golden-set rows, run against
the free-tier Gemini API (gemini-3.5-flash-lite), zero parse failures,
zero placeholder/failed rows. Full per-row output is in
results/llm_agent_predictions.csv; full metrics in
results/llm_agent_results.json.

The LLM agent clearly outperforms both baselines on intent classification
(+20 points over TF-IDF). On escalation, it roughly ties TF-IDF on
accuracy but has lower F1 -- the agent's escalate precision is very high
(93.9%) but recall is only 63.0%, meaning it correctly avoids false
escalations but misses some cases that genuinely should have escalated.
This asymmetry is discussed in the failure analysis below.

**Grounding retrieval quality** (results/retrieval_eval_results.json,
agent/retriever.py, 100% local, no API): querying the full 43k-pair pool
with plain TF-IDF text similarity, the top-3 retrieved historical examples
include at least one whose weak intent label matches the golden query's
true intent 52.5% of the time, mean top-1 cosine similarity 0.47. Since
weak labels only agree with hand labels ~63% of the time on their own,
this 52.5% is a floor, not a ceiling, on true grounding relevance -- see
decision log #12.

**On the LLM-as-judge scores (~4.95/5 average, near-ceiling):** these
were produced by the same model (Gemini 3.5 Flash-Lite) that generated
the replies being judged -- a known limitation, decision log #14. A real
human-agreement check (eval/human_agreement.py, n=30) confirms this
concern directly: exact agreement between the human rater and the LLM
judge was 66.7%, agreement-within-1-point was 73.3%. Cohen's kappa and
Pearson r came back as 0.0/undefined -- not because agreement was bad,
but because the LLM judge gave a perfect 5/5 to all 30 sampled replies
(zero variance), while the human rater's scores ranged from 3 to 5. This
is a real finding: the same-model judge shows a ceiling effect, failing
to distinguish "good" from "excellent" the way a human does. Several
replies the human rated 3/5 (noticeably flawed) were scored 5/5 by the
judge. The ~4.95/5 average in the table above should therefore be read
as "the judge is easily satisfied," not "the replies are near-perfect."

## 3. Failure analysis — top 5 failure modes, with real examples

These come from `results/tfidf_baseline_predictions.csv` (real, already
run) — 70 of 200 golden rows (35%) were misclassified. The LLM agent
should be evaluated against the same failure taxonomy once real numbers
are in; the categories are chosen to be classifier-agnostic.

1. **Retrieval/classification confusion between topically adjacent
   intents when a message is genuinely multi-topic.**
   > *"@SpotifyCares The other playlists stay synced except songs often
   > get removed from my device and need to redownload... which I assume
   > is to give me the latest version, but really how often does a song
   > need to actually refresh?"* — true: `downloads_offline_issue`,
   > TF-IDF predicted: `content_missing_restricted` (the word "removed"
   > pulled it toward the wrong class).
   **Hypothesis:** keyword-adjacent vocabulary across intents ("removed",
   "missing", "gone") is a known TF-IDF weakness; an LLM classifier should
   do meaningfully better here since it can use the surrounding sentence
   structure, but should be checked specifically on this failure category
   once real numbers exist.

2. **Positive/resolved messages misclassified as the problem they used to
   describe.**
   > *"My @115888 it's okay now 😍 I can download and stream all the songs
   > of @133324 💜💜💜"* — true: `general_chitchat_ack` (a thank-you /
   > resolution confirmation), TF-IDF predicted: `downloads_offline_issue`.
   **Hypothesis:** bag-of-words models can't distinguish "download is
   broken" from "download works now, thanks" — this needs actual
   semantic/negation understanding, which is the single clearest case for
   why an LLM classifier should outperform TF-IDF on this dataset.

3. **Sarcasm and complaint-framed feature requests being read as neutral
   chitchat.**
   > *"hey, @SpotifyCares, please please please give us a 'hide band'
   > option. Sometimes I just really NEVER want to listen to a band, you
   > know?"* — true: `feature_request_feedback`, TF-IDF predicted:
   > `general_chitchat_ack`.
   **Hypothesis:** repeated "please" and casual tone reads as chitchat to
   a keyword model even though the content is a clear, specific feature
   ask — again, plausibly an easy win for an LLM classifier, but this is a
   prediction to verify, not a claim already backed by a real number.

4. **Third-party / on-behalf-of requests that need a different escalation
   reason than the intent alone implies.**
   > *"@SpotifyCares you have locked my friend out of her account and she
   > will not stop bothering me at work. please help her change her
   > payment info. Her name is Riley Barsella and she needs some
   > tunes!!!!!"* — true: `login_account_access`, TF-IDF predicted:
   > `billing_payment_subscription` (the word "payment" dominated).
   **Hypothesis:** this is also a safety-relevant case for the escalation
   module specifically — the correct behavior isn't just "escalate," it's
   "escalate AND don't act on a third party's account," which is exactly
   what `docs/worked_examples.md` Example 1's hand-simulated reply does
   and what the real judge rubric's `safety` dimension is designed to
   catch. Worth a dedicated check once real judge scores exist.

5. **Single-turn extraction loses meaning on follow-up-fragment
   messages.** Several golden rows are conversational fragments like "No
   difference ☹️" or "This!" that only make sense as a reply to an earlier
   turn the single-turn pair extraction didn't capture (decision log #4).
   These were labeled `general_chitchat_ack`/`escalate=False` as a
   conservative default during hand-labeling, which means **any model,
   including a perfect one, will look worse than it should on this
   category** — the ceiling on these rows isn't 100%, it's whatever a
   human with no thread context would also guess. This is a dataset
   limitation, not purely a model failure, and it should be excluded (or
   reported separately) rather than blended into one headline number.

---

## 4. What's misleading about my headline number

*(mandatory section, and genuinely the most important one in this report)*

- **The 65.0% / 83.5% TF-IDF numbers, and any future LLM-agent headline
  number, are measured on a golden set that is stratified, not
  traffic-representative** (decision log #6). A model can post a strong
  number on this set while doing worse on the true traffic mix (~60%
  chitchat in practice), or vice versa — the golden set exists to see
  per-intent behavior clearly, at the direct cost of not being a volume
  estimate.
- **The golden set has a single labeler with no second-annotator
  agreement check** (`golden_set_methodology.md`). Every number in this
  report inherits whatever labeling errors or idiosyncratic judgment calls
  I made. `eval/human_agreement.py` exists to at least measure judge-vs-
  human agreement, but the underlying golden labels themselves are
  unaudited.
- **Escalation is graded against one specific written policy, and that
  policy is a judgment call, not ground truth** (decision log #7). A
  stricter or looser policy moves the 36.5% escalate rate and every
  escalate-accuracy number a lot. "This model gets 83.5% escalate
  accuracy" really means "...against one particular, reasonable, but
  contestable definition of when Spotify should escalate."
- **An LLM-as-judge score is not independent ground truth on reply
  quality**, especially when it's the same model family as the agent
  being judged (decision log #14) — it can share blind spots (e.g. rating
  a fluent-but-unsupported claim as grounded). `eval/human_agreement.py`
  is the check on this, but until a human actually fills in that sheet,
  any judge score should be read as "a consistent, cheap proxy," not
  "quality, confirmed."
- **43,031 pairs sounds like a lot, but the retrieval floor is 52.5%
  weak-intent-match@3 on real text similarity alone** — meaning nearly
  half the time, the grounding examples handed to the drafter may not be
  a strong topical match, which puts a ceiling on how grounded the
  drafted replies can be no matter how good the drafting prompt is.
- **Single-turn extraction (decision log #4) puts an unknown ceiling on
  any model's achievable accuracy** on the subset of golden rows that are
  really follow-up fragments — see failure mode #5 above. No number in
  this report separates "the model is wrong" from "the row is
  unanswerable without more context."

---

## 5. What I'd do next with one more week

1. **Run the real harness and replace every "pending real run" placeholder
   in Section 2** — this is the single highest-priority next step; without
   it, this report is a well-tested chassis with no engine test results.
2. **Get a genuinely independent human to fill in
   `eval/human_agreement.py`'s rating sheet** (ideally someone who didn't
   write the judge prompt) and, time permitting, a second person to
   re-label a subsample of the golden set itself to get an inter-annotator
   agreement number on the ground truth, not just the judge.
3. **Reconstruct full threads for at least the ambiguous "fragment" rows**
   identified in failure mode #5, to separate genuine model error from
   unanswerable-without-context rows, and re-report accuracy on the
   answerable subset alongside the blended number.
4. **Try a judge from a different model family** (decision log #14) to
   check whether the same-family judge is systematically inflating
   groundedness/correctness scores relative to an independent model.
5. **A/B the escalation policy's strictness** (looser vs. stricter,
   decision log #7) and report how escalate rate and estimated human
   workload trade off — turns one judgment call into a decision a business
   stakeholder can actually make with numbers in front of them.
6. **Expand retrieval beyond TF-IDF** (e.g. embedding-based retrieval) and
   re-run the local, no-API retrieval-quality check
   (`agent/retriever.py --evaluate`) to see whether the 52.5%
   weak-intent-match floor from Section 4 actually moves — cheap to try
   since it needs no API key.
