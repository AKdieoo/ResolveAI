"""
trivial_baseline.py

The "dumbest thing that could plausibly ship": always predict the single
most frequent intent from the training distribution, and never escalate
(auto-handle everything). This is the floor every real system must clear.
It also anchors "what's misleading about my headline number": headline
accuracy numbers only mean something in reference to a stated baseline.
"""
import pandas as pd
from pathlib import Path
from collections import Counter
import json

BASE = Path(__file__).resolve().parent.parent


def main():
    train = pd.read_csv(BASE / "data" / "processed" / "spotify_weak_labeled.csv")
    golden = pd.read_csv(BASE / "data" / "golden" / "golden_set.csv")

    majority_class = Counter(train["weak_intent"]).most_common(1)[0][0]
    print(f"Majority class (from weak-labeled training distribution): {majority_class}")

    y_true_intent = golden["true_intent"].tolist()
    y_pred_intent = [majority_class] * len(golden)
    intent_acc = sum(t == p for t, p in zip(y_true_intent, y_pred_intent)) / len(golden)

    y_true_escalate = golden["escalate"].tolist()
    y_pred_escalate = [False] * len(golden)  # never escalate
    escalate_acc = sum(t == p for t, p in zip(y_true_escalate, y_pred_escalate)) / len(golden)
    # precision/recall for escalate=True is undefined (no positive predictions) -> recall = 0
    escalate_recall = 0.0

    results = {
        "baseline": "trivial",
        "intent_accuracy": round(intent_acc, 4),
        "majority_class": majority_class,
        "escalate_accuracy": round(escalate_acc, 4),
        "escalate_recall_for_true_class": escalate_recall,
        "n_golden": len(golden),
    }
    print(json.dumps(results, indent=2))

    out_path = BASE / "results" / "trivial_baseline_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
