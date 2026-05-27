"""Agent 3 — Grounding Layer.

Third node in the pipeline. The gatekeeper that converts retrieved
passages into a clean set of VERIFIED facts before any reasoning.

Responsibilities:
- Review the chunks returned by the Retrieval Engine.
- Distill them into atomic, verifiable facts, each one citing the
  ``chunk_id`` it came from (Critical Rule #2).
- Drop, flag, or down-weight passages that are irrelevant,
  contradictory, or not actually supported by a source.
- Keep verified facts strictly separate from anything simulated
  (Critical Rule #1) — this layer outputs facts only, no speculation.
- Return a Pydantic model of grounded facts (Critical Rule #4).

If no facts can be grounded, it signals that downstream so the
reasoning agent does not hallucinate from nothing (Critical Rule #6).
"""
