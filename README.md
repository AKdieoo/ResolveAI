# Spotify AI Support Agent

An AI support agent for **Spotify** (`SpotifyCares` on Twitter), built from
the real Customer Support on Twitter dataset (Kaggle,
`thoughtvector/customer-support-on-twitter`). It classifies incoming
customer messages into a 9-intent taxonomy derived from the real data,
drafts a reply grounded in how Spotify has historically resolved similar
issues, and decides whether to auto-handle or escalate to a human — with a
stated reason.

Brand choice, taxonomy, and every non-obvious decision are explained in
`docs/decision_log.md` (15 entries).

## What's runnable right now, with no API key

Everything in `data/`, `baselines/`, and `agent/retriever.py` is 100%
local (pandas/sklearn) — no download, no API key, reproduces in under 2
minutes:

```bash
pip install -r requirements.txt
python baselines/trivial_baseline.py
python baselines/tfidf_baseline.py
python -m agent.retriever --evaluate        # local grounding-quality check
python -m eval.harness --mock --n 20        # pipeline smoke test, NOT quality numbers
```

### Reproducible results (real, already run)

| Metric | Trivial | TF-IDF + LogReg | LLM Agent (Gemini 3.5 Flash-Lite) |
|---|---|---|---|
| Intent accuracy (9-way, n=200) | 9.5% | 65.0% | **85.0%** |
| Intent F1 (macro) | -- | 66.1% | **85.7%** |
| Escalate accuracy | 63.5% | 83.5% | **85.0%** |
| Escalate F1 | 0% | 77.6% | 75.4% |
| Reply quality (LLM judge, 1-5) | n/a | n/a | 4.95 avg |

All 200 golden-set rows, run for real, zero parse failures. Full output
in results/llm_agent_predictions.csv and results/llm_agent_results.json.

Grounding retrieval quality (agent/retriever.py, local, no API -- see
results/retrieval_eval_results.json): plain TF-IDF text similarity over
the full 43k-pair pool surfaces a same-true-intent example in the top-3
52.5% of the time (a floor, not a ceiling -- see decision log #12).

A real human-agreement check (eval/human_agreement.py, n=30) found the
LLM judge has a ceiling effect: it scored every sampled reply 5/5, while
a human rater's scores ranged 3-5 (66.7% exact agreement). See
docs/report.md Section 2 for the full discussion -- the 4.95/5 average
above should be read as "the judge is easily satisfied," not "the
replies are near-perfect."

Read docs/report.md before quoting any number above -- it has a
mandatory "what's misleading about this number" section.

## Running the real LLM agent

Works with either a free-tier Gemini API key or an Anthropic API key --
agent/llm_client.py auto-detects whichever is set.

Free tier (Gemini, no card needed -- get a key at aistudio.google.com/apikey):
    $env:GEMINI_API_KEY="AIza..."          (PowerShell)
    export GEMINI_API_KEY=AIza...          (bash)

Or with an Anthropic API key:
    export ANTHROPIC_API_KEY=sk-ant-...

Then:
    python -m agent.pipeline --golden-row 0     # run the agent on one message
    python -m eval.harness --sleep 15           # full 200-row golden-set eval

eval/harness.py checkpoints progress every 5 rows to
results/llm_agent_checkpoint.csv and retries transient errors (rate
limits, server hiccups) automatically. If it's interrupted, just re-run
the same command -- it resumes from the checkpoint instead of starting
over.

To check the LLM-as-judge against a human rater:
    python -m eval.human_agreement --sample --n 30   # writes a blinded rating sheet
    # ...fill in the human_overall column by hand...
    python -m eval.human_agreement --score

## What's in this repo

```
spotify-support-agent/
├── data/
│   ├── processed/
│   │   ├── spotify_pairs.csv          # 43,031 real customer<->agent pairs
│   │   └── spotify_weak_labeled.csv   # + weak intent/escalate labels
│   └── golden/
│       ├── golden_sample_raw.csv      # the 200-row stratified sample, unlabeled
│       └── golden_set.csv             # + hand labels (the golden eval set)
├── src/                          # Part 1: data pipeline
│   ├── extract_brand_data.py     # raw twcs.csv -> spotify_pairs.csv
│   ├── weak_label.py             # spotify_pairs.csv -> spotify_weak_labeled.csv
│   ├── golden_labels.py          # the 200 hand-assigned labels
│   └── build_golden_set.py       # merges golden_labels.py onto the raw sample
├── baselines/                    # Part 1: trivial + TF-IDF baselines
│   ├── trivial_baseline.py
│   └── tfidf_baseline.py
├── agent/                        # Part 2: the LLM-based agent
│   ├── config.py                 # shared constants: intents, model, policy
│   ├── retriever.py              # TF-IDF grounding retrieval (local, no API)
│   ├── prompts.py                # every prompt template
│   ├── llm_client.py             # Anthropic API wrapper + MockLLMClient
│   ├── classifier.py             # LLM intent classifier
│   ├── escalation.py             # escalation decision + reason
│   ├── reply_drafter.py          # grounded reply drafter
│   └── pipeline.py               # wires it all into agent.run(text)
├── eval/                         # Part 2: evaluation harness
│   ├── llm_judge.py              # LLM-as-judge rubric scorer
│   ├── harness.py                # runs the agent over the golden set
│   └── human_agreement.py        # judge-vs-human agreement protocol
├── results/                      # outputs from baselines + retriever + mock harness run
├── docs/
│   ├── intent_taxonomy.md
│   ├── golden_set_methodology.md
│   ├── decision_log.md           # 15 entries, numbered, referenced by the report
│   ├── report.md                 # the 6-page report (framing/results/failures/next steps)
│   └── worked_examples.md        # 5 hand-walked pipeline examples w/ real retrieval
└── requirements.txt
```

## (Optional) Re-derive the data pipeline from the raw dataset

Only needed to verify extraction/weak-labeling yourself or refresh the
data:

1. Download `twcs.csv` from Kaggle: **thoughtvector/customer-support-on-twitter**
2. Place it at `data/raw/twcs.csv` (~516MB, ~3M rows)
3. Run:
   ```bash
   python src/extract_brand_data.py
   python src/weak_label.py
   python src/build_golden_set.py
   ```

`build_golden_set.py` re-merges the *existing* hand labels in
`src/golden_labels.py` — it does not re-sample or re-label (decision log
#6: the golden set is fixed, checked-in data).

## Status

- [x] Brand selected and justified from real data
- [x] Full data pipeline: 3M-row Twitter dump -> 43k Spotify pairs -> weak labels
- [x] 9-class intent taxonomy, derived from the data
- [x] 200-example hand-labeled golden set, documented methodology
- [x] Trivial + TF-IDF baselines, real reproducible numbers
- [x] LLM classifier, grounded reply drafter, escalation module -- complete, real code
- [x] Evaluation harness: automated metrics + LLM-as-judge + human-agreement protocol
- [x] Real LLM-agent numbers: 85.0% intent accuracy, 85.0% escalate accuracy, all 200 rows, run for real against the free Gemini API
- [x] Real human-agreement check: 66.7% exact agreement, judge ceiling-effect finding
- [x] Report: framing, baselines comparison, failure analysis, misleading-number section, next steps -- fully updated with real numbers
- [x] Decision log, 17 entries

## Requirements
- Python 3.10+
- `pip install -r requirements.txt` (pandas, numpy, scikit-learn — all
  local; `anthropic` only needed to run the real agent/harness, not the
  baselines or retriever)
- `ANTHROPIC_API_KEY` — only for `agent/pipeline.py` and `eval/harness.py`
  without `--mock`
