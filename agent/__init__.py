"""
agent/ — the Part 2 LLM-based Spotify support agent.

Modules:
    config.py         shared constants (intents, paths, model name)
    retriever.py       TF-IDF grounding retriever over the 43k historical pairs
                        (100% local, no API key — same as the Part 1 baselines)
    prompts.py          all prompt templates in one place, versioned
    llm_client.py       thin wrapper around the Anthropic API (+ a MockLLMClient
                        for offline smoke-testing the pipeline without a key)
    classifier.py       LLM intent classifier
    escalation.py       escalation decision module (produces a reason string)
    reply_drafter.py    grounded reply drafter
    pipeline.py         wires the above into one agent.run(customer_text)
"""
