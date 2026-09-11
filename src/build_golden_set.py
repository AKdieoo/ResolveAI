"""
build_golden_set.py
Merges the hand-assigned labels in golden_labels.py onto the raw 200-row
sample to produce the final golden evaluation set.
"""
import pandas as pd
from pathlib import Path
from golden_labels import LABELS

BASE = Path(__file__).resolve().parent.parent
raw = pd.read_csv(BASE / "data" / "golden" / "golden_sample_raw.csv")
assert len(raw) == len(LABELS)

raw["true_intent"] = [t for t, e in LABELS]
raw["escalate"] = [e for t, e in LABELS]
raw["weak_label_correct"] = raw["weak_intent"] == raw["true_intent"]

out_cols = ["customer_tweet_id", "customer_text", "true_intent", "escalate",
            "weak_intent", "weak_label_correct", "agent_text", "agent_tweet_id", "created_at"]
raw = raw.rename(columns={"agent_text": "historical_agent_reply"})
out_cols = ["customer_tweet_id", "customer_text", "true_intent", "escalate",
            "weak_intent", "weak_label_correct", "historical_agent_reply",
            "agent_tweet_id", "created_at"]
raw[out_cols].to_csv(BASE / "data" / "golden" / "golden_set.csv", index=False)

print(f"Wrote golden_set.csv with {len(raw)} rows")
print("\nIntent distribution (true labels):")
print(raw["true_intent"].value_counts())
print(f"\nEscalate rate: {raw['escalate'].mean():.1%}")
print(f"Weak-label agreement with hand labels: {raw['weak_label_correct'].mean():.1%}")
