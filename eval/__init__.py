"""
eval/ — the Part 2 evaluation harness.

    llm_judge.py            LLM-as-judge rubric scorer for drafted replies
    harness.py               runs the full agent over the golden set,
                             computes automated metrics (same protocol as
                             the Part 1 baselines) + judge scores
    human_agreement.py       protocol + scaffolding for measuring how well
                             the LLM judge agrees with a human rater
"""
