"""One-off debug runner — dumps raw_response for the 4 parse-failed cases.

Reproduces the same A1 -> A2 -> A3 -> A4 chain the evaluator uses, but writes
``CounterfactualReasoning.raw_response`` verbatim (plus parse_warnings) to
``evaluation/raw_debug.txt`` so we can see EXACTLY what the LLM emitted on the
four cases the parser couldn't read (no_british_raj, louis_xvi_survives,
no_genghis, cuban_missile_war).

No agent or parser code is modified — this is read-only inspection.

Run from project root:
    D:\\historyos\\venv\\Scripts\\python.exe -m evaluation.dump_raw
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from agents.grounding_layer import ground_context
from agents.query_understanding import analyze_query
from agents.reasoning_agent import reason_about_counterfactual
from agents.retrieval_engine import retrieve_context

TARGET_IDS = {
    "no_british_raj",
    "louis_xvi_survives",
    "no_genghis",
    "cuban_missile_war",
}

TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
OUT_PATH = Path(__file__).resolve().parent / "raw_debug.txt"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    with TEST_CASES_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    cases = [c for c in payload["cases"] if c["id"] in TARGET_IDS]
    print(f"Running {len(cases)} target case(s): {[c['id'] for c in cases]}")

    sections: list[str] = []
    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] {case['id']} — running pipeline...")
        t0 = time.monotonic()
        try:
            analysis = analyze_query(case["question"])
            context = retrieve_context(analysis)
            grounded = ground_context(context, case["question"])
            reasoning = reason_about_counterfactual(analysis, grounded)
            elapsed = time.monotonic() - t0
            print(f"    done in {elapsed:.1f}s — raw_response = {len(reasoning.raw_response)} chars")
            section = "\n".join([
                "#" * 78,
                f"# CASE: {case['id']}",
                f"# Q:    {case['question']}",
                f"# proposed_change: {reasoning.proposed_change}",
                f"# divergence_point: {reasoning.divergence_point}",
                f"# parse_warnings: {reasoning.parse_warnings}",
                f"# steps parsed: {len(reasoning.steps)}",
                "#" * 78,
                "",
                "----- raw_response BEGIN -----",
                reasoning.raw_response or "(empty)",
                "----- raw_response END -----",
                "",
            ])
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            print(f"    FAILED in {elapsed:.1f}s — {type(exc).__name__}: {exc}")
            section = "\n".join([
                "#" * 78,
                f"# CASE: {case['id']}",
                f"# Q:    {case['question']}",
                f"# ERROR: {type(exc).__name__}: {exc}",
                "#" * 78,
                "",
            ])
        sections.append(section)

    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
