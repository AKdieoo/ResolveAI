"""
human_agreement.py — protocol + scoring for checking how well the LLM
judge (eval/llm_judge.py) agrees with a human rater, per the assignment's
requirement for "evidence of how well your judge agrees with a human."

Honesty note (see docs/decision_log.md #15): I cannot produce real human
agreement numbers myself — I'm the one who wrote the judge prompt and the
agent, so my own ratings would not be an independent check, and there's no
second human available in this build environment. What this script gives
you instead is a COMPLETE, ready-to-run protocol:

  1. `python -m eval.human_agreement --sample` pulls a stratified subsample
     (default 30) from results/llm_agent_predictions.csv (after you've run
     the real harness) into results/human_agreement_sheet.csv, with an
     empty `human_overall` column (1-5) for a human rater to fill in.
  2. A human (you, or someone else who didn't write this repo) rates the
     `reply` column for each row against the same rubric the LLM judge
     uses (see agent/prompts.py:judge_system_prompt) — independently,
     without seeing the judge's scores.
  3. `python -m eval.human_agreement --score` reads the filled-in sheet and
     computes: exact agreement rate, agreement within 1 point, Cohen's
     kappa (linear-weighted), and Pearson correlation between
     `judge_overall` and `human_overall`.

Sampling is stratified across true_intent x escalate (matching the golden
set's own sampling philosophy, decision log #6) so the agreement check
isn't dominated by the majority chitchat/no-escalate cell.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from agent.config import RESULTS_DIR


def build_sample(n: int = 30, seed: int = 42) -> pd.DataFrame:
    preds_path = RESULTS_DIR / "llm_agent_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"{preds_path} not found. Run `python -m eval.harness` (a REAL "
            "run, not --mock) first, then build the human-agreement sample."
        )
    df = pd.read_csv(preds_path)
    df = df[df["judge_parse_ok"]].copy()

    # Stratify on (true_intent, true_escalate) so rare intents and the
    # escalate=True minority class aren't crowded out, same philosophy as
    # docs/golden_set_methodology.md's own stratification.
    df["_strata"] = df["true_intent"].astype(str) + "_" + df["true_escalate"].astype(str)
    sample = (
        df.groupby("_strata", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(1, n // df["_strata"].nunique())), random_state=seed))
    )
    if len(sample) < n:
        remainder = df.drop(sample.index)
        top_up = remainder.sample(min(len(remainder), n - len(sample)), random_state=seed)
        sample = pd.concat([sample, top_up])
    sample = sample.sample(frac=1, random_state=seed).head(n).drop(columns=["_strata"])

    sheet = sample[
        ["customer_tweet_id", "customer_text", "true_intent", "pred_intent",
         "true_escalate", "pred_escalate", "escalate_reason", "reply", "judge_overall"]
    ].copy()
    sheet = sheet.rename(columns={"judge_overall": "llm_judge_overall_SCORE_HIDDEN_UNTIL_YOU_RATE"})
    # Blind the human rater to the LLM's score by default -- move the LLM
    # score to a separate column only after rating, to avoid anchoring.
    llm_scores = sheet.pop("llm_judge_overall_SCORE_HIDDEN_UNTIL_YOU_RATE")
    sheet["human_overall"] = ""  # human fills this in, 1-5
    sheet["_llm_judge_overall_FILL_LAST_DO_NOT_LOOK_YET"] = llm_scores
    return sheet


def score_agreement(sheet_path=None) -> dict:
    sheet_path = sheet_path or (RESULTS_DIR / "human_agreement_sheet.csv")
    df = pd.read_csv(sheet_path)
    df = df[df["human_overall"].notna() & (df["human_overall"] != "")]
    if len(df) == 0:
        raise ValueError(
            f"No filled-in human_overall ratings found in {sheet_path}. "
            "Fill in that column (1-5) before scoring."
        )

    human = df["human_overall"].astype(int)
    llm = df["_llm_judge_overall_FILL_LAST_DO_NOT_LOOK_YET"].astype(int)

    exact_agreement = (human == llm).mean()
    within_1 = (abs(human - llm) <= 1).mean()
    kappa = cohen_kappa_score(human, llm, weights="linear")
    pearson_r = float(np.corrcoef(human, llm)[0, 1]) if human.nunique() > 1 and llm.nunique() > 1 else float("nan")

    return {
        "n_rated": len(df),
        "exact_agreement_rate": round(float(exact_agreement), 4),
        "agreement_within_1_point": round(float(within_1), 4),
        "cohens_kappa_linear_weighted": round(float(kappa), 4),
        "pearson_r": round(pearson_r, 4) if pearson_r == pearson_r else None,
        "note": (
            "Computed from results/human_agreement_sheet.csv. See "
            "eval/human_agreement.py docstring for the rating protocol. "
            "This script is real and complete; the numbers only exist "
            "once a human has actually filled in the sheet."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Build the rating sheet")
    group.add_argument("--score", action="store_true", help="Score a filled-in rating sheet")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.sample:
        sheet = build_sample(n=args.n)
        out_path = RESULTS_DIR / "human_agreement_sheet.csv"
        sheet.to_csv(out_path, index=False)
        print(f"Wrote {len(sheet)}-row rating sheet to {out_path}")
        print("Fill in the 'human_overall' column (1-5), then run: python -m eval.human_agreement --score")
    elif args.score:
        import json
        results = score_agreement()
        print(json.dumps(results, indent=2))
        with open(RESULTS_DIR / "human_agreement_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
