"""Report generator — final output formatting.

Renders the pipeline's structured result into a human-readable report
that ENFORCES the verified/simulated separation in the presentation.

Responsibilities:
- Take the final pipeline state (verified facts + causal chain +
  confidence scores).
- Produce a report with two clearly separated sections:
    1. VERIFIED FACTS — each with its citing ``chunk_id`` / source.
    2. SIMULATED CONSEQUENCES — each causal step with its confidence
       score.
- Never blend the two (Critical Rule #1); make the boundary obvious.
- Return both a structured object and a display-ready string
  (e.g. Markdown) for the Streamlit frontend to show.
"""
