"""
tfidf_baseline.py

The "simple, no-LLM" baseline: TF-IDF features + Logistic Regression.
Trained on the 43k weak-labeled pairs (EXCLUDING every customer_tweet_id
that ended up in the golden set, to avoid train/test leakage), evaluated on
the 200 hand-labeled golden examples.

Two independent models:
  1. intent classifier (9-way) trained against weak_intent labels
  2. escalate classifier (binary) trained against weak_escalate labels

Both training label sources are noisy (see docs/golden_set_methodology.md
and weak_label.py). That noise is expected to cap this baseline's ceiling
well below what a good LLM-based classifier can do -- that gap IS the
point of having this baseline.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

BASE = Path(__file__).resolve().parent.parent


def main():
    weak = pd.read_csv(BASE / "data" / "processed" / "spotify_weak_labeled.csv")
    golden = pd.read_csv(BASE / "data" / "golden" / "golden_set.csv")

    # Remove leakage: any row whose customer_tweet_id is in the golden set
    golden_ids = set(golden["customer_tweet_id"])
    train = weak[~weak["customer_tweet_id"].isin(golden_ids)].copy()
    print(f"Training rows after removing {len(golden_ids)} golden-set ids: {len(train)}")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
    X_train = vectorizer.fit_transform(train["customer_text"].astype(str))
    X_golden = vectorizer.transform(golden["customer_text"].astype(str))

    # ---- Intent classifier ----
    intent_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    intent_clf.fit(X_train, train["weak_intent"])
    intent_pred = intent_clf.predict(X_golden)
    intent_true = golden["true_intent"]

    intent_acc = accuracy_score(intent_true, intent_pred)
    intent_f1_macro = f1_score(intent_true, intent_pred, average="macro", zero_division=0)
    intent_f1_weighted = f1_score(intent_true, intent_pred, average="weighted", zero_division=0)
    intent_report = classification_report(intent_true, intent_pred, zero_division=0)

    # ---- Escalate classifier ----
    esc_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    esc_clf.fit(X_train, train["weak_escalate"])
    esc_pred = esc_clf.predict(X_golden)
    esc_true = golden["escalate"]

    esc_acc = accuracy_score(esc_true, esc_pred)
    esc_precision = precision_score(esc_true, esc_pred, zero_division=0)
    esc_recall = recall_score(esc_true, esc_pred, zero_division=0)
    esc_f1 = f1_score(esc_true, esc_pred, zero_division=0)

    results = {
        "baseline": "tfidf_logreg",
        "n_train": len(train),
        "n_golden": len(golden),
        "intent_accuracy": round(intent_acc, 4),
        "intent_f1_macro": round(intent_f1_macro, 4),
        "intent_f1_weighted": round(intent_f1_weighted, 4),
        "escalate_accuracy": round(esc_acc, 4),
        "escalate_precision": round(esc_precision, 4),
        "escalate_recall": round(esc_recall, 4),
        "escalate_f1": round(esc_f1, 4),
    }

    print(json.dumps(results, indent=2))
    print("\n--- Per-intent classification report ---")
    print(intent_report)

    out_path = BASE / "results" / "tfidf_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # save predictions for failure analysis in the report
    golden_out = golden.copy()
    golden_out["tfidf_pred_intent"] = intent_pred
    golden_out["tfidf_pred_escalate"] = esc_pred
    golden_out.to_csv(BASE / "results" / "tfidf_baseline_predictions.csv", index=False)

    with open(BASE / "results" / "tfidf_baseline_classification_report.txt", "w") as f:
        f.write(intent_report)

    print(f"\nWrote results to {BASE / 'results'}")


if __name__ == "__main__":
    main()
