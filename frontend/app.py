"""Streamlit frontend — HistoryOS UI.

The user-facing entry point. Lets a user ask a "what if" historical
question and shows the grounded, confidence-scored answer.

Responsibilities:
- Render an input box for the counterfactual question and a submit
  action.
- Call ``pipeline/historios_pipeline.run()`` to execute the full
  5-agent pipeline.
- Display the report from ``report_generator.py`` with VERIFIED facts
  (and their citations) visually separated from SIMULATED consequences
  (and their confidence scores).
- Show graceful messages for empty / ungrounded results rather than
  fabricated answers.

Run with: ``streamlit run frontend/app.py``.
"""
