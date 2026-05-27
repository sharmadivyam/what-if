"""Agent 4 — Reasoning Agent.

Fourth node in the pipeline. Simulates the consequences of the
counterfactual using multi-step causal reasoning, grounded on the
verified facts from the Grounding Layer.

Responsibilities:
- Take the counterfactual premise (Agent 1) and the verified facts
  (Agent 3) as the only allowed inputs to the LLM (Critical Rule #6).
- Build a causal chain of consequences, where each step references
  the verified facts it depends on.
- Enforce a HARD CAP of 4 causal reasoning steps as a hallucination
  guard (Critical Rule #3).
- Clearly mark every produced consequence as SIMULATED, never as fact
  (Critical Rule #1).
- Return a Pydantic model describing the causal chain (Critical Rule #4).
"""
