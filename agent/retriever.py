"""
retriever.py — grounds the reply drafter in real historical resolutions.

This is 100% local (TF-IDF + cosine similarity, sklearn) — no API key
needed, same as the Part 1 baselines. Given an incoming customer message,
it returns the top-k most similar historical (customer_text, agent_text)
pairs from the 43k extracted Spotify threads, so the drafter can ground its
reply in how Spotify actually resolved similar issues rather than
hallucinating a plausible-sounding answer.

Design choices (see docs/decision_log.md #12):
- Retrieval pool excludes every golden-set customer_tweet_id, using the
  same hard ID-filter approach as baselines/tfidf_baseline.py, so grounding
  examples can never leak the exact golden-set answer for a golden query.
- Retrieval is optionally intent-filtered (restrict the pool to rows whose
  weak_intent matches the query's classified intent) with a fallback to the
  unfiltered pool if the filtered pool has too few candidates. This trades
  weak-label noise for topical relevance — see decision log #12 for why
  that trade was made.
- A single TfidfVectorizer is fit once on the full retrieval pool and
  cached (module-level singleton) since re-fitting per call would be slow
  and pointless — the vector space doesn't need to change per query.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.config import PAIRS_PATH, WEAK_LABELED_PATH, GOLDEN_PATH, RESULTS_DIR


@dataclass
class GroundingExample:
    customer_tweet_id: int
    customer_text: str
    agent_text: str
    weak_intent: str
    similarity: float


class Retriever:
    """Loads the retrieval pool once; call .retrieve() per query."""

    def __init__(self, exclude_ids: Optional[set] = None):
        df = pd.read_csv(WEAK_LABELED_PATH)
        if exclude_ids:
            df = df[~df["customer_tweet_id"].isin(exclude_ids)].copy()
        self.pool = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        self.pool_matrix = self.vectorizer.fit_transform(
            self.pool["customer_text"].astype(str)
        )

    def retrieve(
        self,
        query_text: str,
        k: int = 3,
        intent_filter: Optional[str] = None,
        min_filtered_pool: int = 20,
    ) -> List[GroundingExample]:
        """Return the top-k most similar historical pairs.

        If intent_filter is given and the intent-filtered pool has at least
        `min_filtered_pool` candidates, retrieval is restricted to that
        pool (more topically relevant grounding). Otherwise it falls back
        to the full pool (more robust for rare intents / classifier
        misfires) — this fallback is itself logged in the returned
        examples via `used_intent_filter` on the caller side if needed.
        """
        pool = self.pool
        matrix = self.pool_matrix

        used_filtered = False
        if intent_filter is not None:
            mask = (pool["weak_intent"] == intent_filter).values
            if mask.sum() >= min_filtered_pool:
                pool = pool[mask].reset_index(drop=True)
                matrix = self.pool_matrix[mask]
                used_filtered = True

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, matrix)[0]
        top_idx = sims.argsort()[::-1][:k]

        results = []
        for i in top_idx:
            row = pool.iloc[i]
            results.append(
                GroundingExample(
                    customer_tweet_id=int(row["customer_tweet_id"]),
                    customer_text=str(row["customer_text"]),
                    agent_text=str(row["agent_text"]),
                    weak_intent=str(row["weak_intent"]),
                    similarity=round(float(sims[i]), 4),
                )
            )
        self._last_used_filtered = used_filtered
        return results


def _load_golden_ids() -> set:
    golden = pd.read_csv(GOLDEN_PATH)
    return set(golden["customer_tweet_id"])


def evaluate_retrieval_quality(k: int = 3) -> dict:
    """Local, no-API metric for how good the TEXT-SIMILARITY signal alone
    is at surfacing topically-correct grounding examples.

    IMPORTANT — this deliberately does NOT use intent_filter. Filtering the
    candidate pool to weak_intent == true_intent and then checking whether
    the retrieved results' weak_intent == true_intent would be circular
    (it's true by construction of the filter, not evidence of anything).
    An earlier version of this script made exactly that mistake — it's
    logged as a caught error in decision log #12, because it's a realistic
    trap ("evaluate the filtered pipeline") worth naming.

    What this actually measures: querying the FULL, unfiltered 43k-example
    pool with plain TF-IDF cosine similarity, does the top-k include at
    least one example whose weak_intent matches the golden query's
    true_intent? This is the signal available BEFORE any classifier
    decision, i.e. a lower bound on how much the intent-filter step (in
    pipeline.py, gated on the classifier's *predicted* intent) can help.
    """
    golden = pd.read_csv(GOLDEN_PATH)
    golden_ids = set(golden["customer_tweet_id"])
    retriever = Retriever(exclude_ids=golden_ids)

    hits = 0
    top1_sims = []
    rows = []
    for _, row in golden.iterrows():
        # Unfiltered: plain text similarity only, no peeking at true_intent.
        examples = retriever.retrieve(row["customer_text"], k=k, intent_filter=None)
        hit = any(ex.weak_intent == row["true_intent"] for ex in examples)
        hits += int(hit)
        top1 = examples[0].similarity if examples else 0.0
        top1_sims.append(top1)
        rows.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "true_intent": row["true_intent"],
                "top1_similarity": top1,
                "top1_weak_intent": examples[0].weak_intent if examples else None,
                "any_weak_intent_match_in_topk_unfiltered": hit,
            }
        )

    n = len(golden)
    results = {
        "metric": "unfiltered_grounding_weak_intent_match_rate@k",
        "k": k,
        "n_golden": n,
        "hit_rate": round(hits / n, 4),
        "mean_top1_similarity": round(sum(top1_sims) / n, 4),
        "caveat": (
            "Measured on the FULL retrieval pool with plain text "
            "similarity (no intent filtering, no peeking at true_intent) "
            "-- see the docstring above for why the filtered version "
            "would have been circular. It also checks retrieved weak_intent "
            "against true_intent, and weak labels only agree with hand "
            "labels ~63% of the time (golden_set_methodology.md), so this "
            "hit rate is a floor, not a ceiling, on true grounding "
            "relevance -- see decision log #12."
        ),
    }
    return results, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test/evaluate the retriever")
    parser.add_argument("--query", type=str, default=None, help="Ad-hoc query to test")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--evaluate", action="store_true", help="Run the golden-set retrieval quality check")
    args = parser.parse_args()

    if args.evaluate:
        results, rows = evaluate_retrieval_quality(k=args.k)
        print(json.dumps(results, indent=2))
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(RESULTS_DIR / "retrieval_eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "retrieval_eval_rows.csv", index=False)
        print(f"\nWrote results to {RESULTS_DIR}")
    elif args.query:
        golden_ids = _load_golden_ids()
        retriever = Retriever(exclude_ids=golden_ids)
        examples = retriever.retrieve(args.query, k=args.k)
        for ex in examples:
            print(json.dumps(asdict(ex), indent=2))
    else:
        parser.print_help()
