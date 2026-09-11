"""
extract_brand_data.py

Extracts all SpotifyCares (agent) <-> customer conversation pairs from the raw
'Customer Support on Twitter' dataset (twcs.csv).

Why a two-pass streaming approach instead of pandas.read_csv on the whole file:
the raw file is ~3M rows / ~500MB. We only need the ~43k SpotifyCares rows and
whichever customer rows they reply to, so we stream the file twice with the
csv module to keep memory low and avoid needing the whole file in RAM.

Output: data/processed/spotify_pairs.csv
Columns:
    customer_tweet_id, customer_text, agent_tweet_id, agent_text, created_at

A "pair" is one customer inbound tweet and the FIRST direct agent reply to it
(agent tweet whose in_response_to_tweet_id == customer tweet_id). This is a
simplification: many real threads run 3-6 turns before resolution. We capture
the first substantive reply because (a) it's the turn our agent needs to
generate (the initial triage/response), and (b) full thread reconstruction is
left as a documented limitation (see docs/decision_log.md, decision #4).
"""
import csv
import re
import sys
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "twcs.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spotify_pairs.csv"
BRAND = "SpotifyCares"

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
SIGNOFF_RE = re.compile(r"/[A-Z]{2}$")  # spotify agents sign off e.g. "/LS"


def clean_text(t: str) -> str:
    t = URL_RE.sub("", t)
    t = SIGNOFF_RE.sub("", t)
    t = t.strip()
    return t


def main():
    if not RAW_PATH.exists():
        print(f"ERROR: raw file not found at {RAW_PATH}. Place twcs.csv there.", file=sys.stderr)
        sys.exit(1)

    print("Pass 1/2: scanning for agent tweets + all tweet metadata index...")
    agent_replies = []  # list of dict: tweet_id, in_response_to_tweet_id, text, created_at
    all_tweet_index = {}  # tweet_id -> (author_id, inbound, text, created_at)

    with open(RAW_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_tweet_index[row["tweet_id"]] = row
            if row["author_id"] == BRAND and row["inbound"] == "False":
                agent_replies.append(row)

    print(f"Found {len(agent_replies)} agent tweets from {BRAND}")

    print("Pass 2/2: matching agent replies to the customer tweet they answered...")
    pairs = []
    for arow in agent_replies:
        parent_id = arow["in_response_to_tweet_id"]
        if not parent_id or parent_id not in all_tweet_index:
            continue
        crow = all_tweet_index[parent_id]
        if crow["inbound"] != "True":
            continue  # only keep genuine customer-initiated turns
        cust_text = clean_text(crow["text"])
        agent_text = clean_text(arow["text"])
        if len(cust_text) < 8 or len(agent_text) < 8:
            continue
        pairs.append({
            "customer_tweet_id": crow["tweet_id"],
            "customer_text": cust_text,
            "agent_tweet_id": arow["tweet_id"],
            "agent_text": agent_text,
            "created_at": arow["created_at"],
        })

    print(f"Matched {len(pairs)} clean customer->agent pairs")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["customer_tweet_id", "customer_text",
                                                "agent_tweet_id", "agent_text", "created_at"])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
