"""Evaluator — pipeline quality harness.

Runs HistoryOS against the curated cases in ``test_cases.json`` and
scores how well it obeys the project's critical rules.

Responsibilities:
- Load test cases (question + expectations) from ``test_cases.json``.
- Run each question through ``historios_pipeline.run()``.
- Check / score, for each result:
    * every verified fact cites a real ``chunk_id`` (Critical Rule #2),
    * verified and simulated content stay separated (Critical Rule #1),
    * the causal chain never exceeds 4 steps (Critical Rule #3),
    * confidence scores look calibrated.
- Print / save an aggregate report so regressions are visible as the
  build progresses.
"""
