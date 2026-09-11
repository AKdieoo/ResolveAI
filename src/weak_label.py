"""
weak_label.py

Applies a keyword/regex rule-set to bulk-label the 43k extracted Spotify
pairs with one of the 9 intents from docs/intent_taxonomy.md. This is a
*weak* labeling function -- noisy but free at scale -- used ONLY to give the
TF-IDF baseline something to train on. It is explicitly NOT the golden
evaluation set (that is hand-labeled separately, see build_golden_set.py and
docs/golden_set_methodology.md).

Rules are ordered; first match wins. Order matters (e.g. "family" checked
before generic "account" so family-plan questions don't get misfiled as
login issues).
"""
import re
import csv
from pathlib import Path

PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spotify_pairs.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "spotify_weak_labeled.csv"

RULES = [
    ("family_plan_management", re.compile(
        r"\bfamily\b|\bsub[- ]?account\b|\binvite\b.*(member|plan)|\bremove\b.*(member|plan)", re.I)),
    ("login_account_access", re.compile(
        r"\blog\s?in\b|\blogin\b|\bpassword\b|\blocked out\b|\bcan'?t access\b|\bforgot my\b|\breset my\b|"
        r"\bfacebook (account|login)\b|\bverify (my )?identity\b|\bhack(ed|er)?\b|\bstolen\b|\bsuspicious login\b|"
        r"\bemail (address )?(is )?wrong\b|\bchange (my )?email\b", re.I)),
    ("billing_payment_subscription", re.compile(
        r"\bsubscription\b|\bpremium\b|\brefund\b|\bcancel\b|\bcharged?\b|\btrial\b|\bstudent (discount|price)\b|"
        r"\bpayment\b|\bcredit card\b|\bdebit card\b|\bbill(ed|ing)?\b|\bupgrade\b|\bdowngrade\b|\bprice\b|"
        r"\bpaypal\b|\bgift card\b|\bdiscount\b|\$\s?\d|£\s?\d|€\s?\d", re.I)),
    ("downloads_offline_issue", re.compile(
        r"\bdownload(s|ed|ing)?\b|\boffline\b|\bstorage\b", re.I)),
    ("playback_streaming_issue", re.compile(
        r"\bcrash(es|ed|ing)?\b|\bbuffer(ing)?\b|\bloading\b|\bwon'?t play\b|\bstuck\b|\bskip(s|ping)?\b|"
        r"\berror\b|\bfreez(es|ing)?\b|\bnot working\b|\bstopped working\b|\bbug\b|\blag(gy|ging)?\b|"
        r"\bcraps? out\b|\bis (down|broken)\b|\bweb ?player\b|\bwon'?t (open|load|start)\b|\bkeeps? (crashing|freezing|stopping)\b|"
        r"\bcan'?t (play|listen|log ?in)\b|\bsound(s)? (weird|bad|off)\b|\bmuted?\b|\bvolume\b|\bshuffle\b.*(broken|wrong|not)", re.I)),
    ("content_missing_restricted", re.compile(
        r"\bremoved\b|\bnot available\b|\bmissing\b|\bcan'?t find\b|\brestrict(ed|ion)?\b|\blicens(e|ing)\b|"
        r"\bregion\b|\bcountry\b.*(available|support)|\btaken? (off|down)\b|\bnot on spotify\b|\bwhere (is|are|'?s)\b.*(album|song|playlist|artist)|"
        r"\bwhy (is|isn'?t|are|aren'?t)\b.*(on|available)", re.I)),
    ("device_platform_support", re.compile(
        r"\b(smart ?tv|xbox|playstation|ps4|ps5|android auto|carplay|alexa|sonos|windows|ios|mac|linux|"
        r"apple watch|smartwatch|browser|chromebook)\b|\bwhich devices?\b|\bis there an? app\b|"
        r"\bsupport (finnish|spanish|french|german|[a-z]+ese|language)\b|\bdo you support\b", re.I)),
    ("feature_request_feedback", re.compile(
        r"\bads?\b|\bshould (add|let|allow)\b|\bwish (you|spotify)\b|\bplease (add|fix|bring|work on|consider)\b|"
        r"\bfeature\b|\bsuggest(ion)?\b|\bwould be (nice|great|cool)\b|\bwhy (can'?t|don'?t) (you|we)\b|"
        r"\bindependent (artist|musician)\b|\bfeatured?\b.*(playlist|gig|concert)", re.I)),
]

FALLBACK = "general_chitchat_ack"

# Weak proxy for "was this actually escalated": did the real agent's reply
# ask for a DM / private info hand-off? This is noisy (agents sometimes DM
# for reasons unrelated to our escalation policy, e.g. sharing a discount
# code) but it's the only free signal available at 43k-row scale, and it's
# used ONLY to train the "simple" baseline's escalate head -- the golden set
# escalate labels are hand-assigned independently (see golden_set_methodology.md).
DM_HANDOFF_RE = re.compile(
    r"\bdm\b|\bsent you (a |)dm\b|\bprivate message\b|\bwe'?ve just sent\b|\baccounts team\b", re.I)


def label(text: str) -> str:
    for intent, pattern in RULES:
        if pattern.search(text):
            return intent
    return FALLBACK


def weak_escalate(agent_reply_text: str) -> bool:
    return bool(DM_HANDOFF_RE.search(agent_reply_text))


def main():
    rows = []
    with open(PAIRS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["weak_intent"] = label(row["customer_text"])
            row["weak_escalate"] = weak_escalate(row["agent_text"])
            rows.append(row)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    dist = Counter(r["weak_intent"] for r in rows)
    print("Weak label distribution:")
    for k, v in dist.most_common():
        print(f"  {k:35s} {v:6d}  ({100*v/len(rows):.1f}%)")
    print(f"\nWrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
