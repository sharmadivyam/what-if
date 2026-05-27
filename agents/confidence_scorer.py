"""Agent 5 — Confidence Scorer.

Final node in the pipeline. Assigns a calibrated confidence score to
each simulated consequence in the causal chain.

Responsibilities:
- Take the causal chain from the Reasoning Agent plus the verified
  facts it was built on.
- Score each simulated step (e.g. 0.0-1.0) based on how strongly it
  is supported by verified facts vs. how speculative it is, and how
  far down the causal chain it sits.
- Keep the VERIFIED/SIMULATED boundary intact (Critical Rule #1):
  facts stay at full confidence, consequences carry their own scores.
- Return a Pydantic model with per-step confidence (Critical Rule #4)
  for the report generator to render.
"""
