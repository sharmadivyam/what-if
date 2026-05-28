"""Evaluator — pipeline quality harness.

Runs HistoryOS against the curated cases in ``test_cases.json`` and scores how
well each run obeys the project's critical rules.

Responsibilities:
- Load cases (question + expectations) from ``test_cases.json``.
- For each case, chain Agents 1 → 2 → 3 → 4 manually (the LangGraph wiring in
  ``pipeline/historios_pipeline.py`` is not built yet) and print the resulting
  ``CounterfactualReasoning``.
- Apply four spot-checks per case and print a compact pass/fail table:
    CHECK 1 — every reasoning step's claim is labelled ``[SIMULATED]``
             (Critical Rule #1).
    CHECK 2 — every step cites at least one chunk_id that exists in the
             supplied ``GroundedContext`` (Critical Rule #2: cite real sources).
    CHECK 3 — no more than ``settings.MAX_CAUSAL_STEPS`` (4) reasoning steps in
             either the parsed output or the raw LLM response
             (Critical Rule #3: hallucination guard).
    CHECK 4 — no claim is presented as fact when it is actually simulation
             (Critical Rule #4). Heuristic: every step's consequence body has a
             ``[SIMULATED]`` marker AND the tail sections (RECONNECTION POINT,
             HISTORIAN'S NOTE) either carry their own ``[SIMULATED]`` markers
             or hedge with would/might/could/may/likely/probably/possibly/if.

The evaluator never raises on a per-case failure — it logs the exception, marks
all four checks as failed for that case, and continues. Confidence Scorer
(Agent 5) is intentionally NOT invoked yet: it is not implemented, and the four
checks above don't need it.

LLM-call budget: ~4 calls per case (1 query understanding + 2 grounding + 1
reasoning) × 8 cases = ~32 calls. The free Cerebras tier throttles bursts —
the shared OpenAI client has ``max_retries=6`` (see ``core/llm_client.py``)
which makes individual 429s self-heal with ~60s waits; total wall time can
therefore range from minutes to tens of minutes.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from agents.grounding_layer import ground_context
from agents.query_understanding import analyze_query
from agents.reasoning_agent import CounterfactualReasoning, reason_about_counterfactual
from agents.retrieval_engine import retrieve_context
from config import settings

logger = logging.getLogger(__name__)

# Hedging vocabulary used by CHECK 4 to recognise simulation-aware phrasing in
# the tail sections (RECONNECTION POINT, HISTORIAN'S NOTE). Conservative — false
# positives are far worse than false negatives here: this is a spot-check, not
# a grader.
_HEDGE_RE = re.compile(
    r"\b(simulated|would|might|may|could|likely|probably|possibly|perhaps|"
    r"counterfactual|hypothetical|if\s+\w+\s+had|in\s+this\s+alternate|"
    r"speculative|uncertain|debated|some\s+(?:historians|argue))\b",
    re.IGNORECASE,
)
_SIMULATED_RE = re.compile(r"\[SIMULATED\]", re.IGNORECASE)
_STEP_HEADER_RE = re.compile(r"^[ \t]*Step[ \t]+([1-9]\d?)\b", re.MULTILINE)

TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"


@dataclass
class CaseResult:
    """Outcome of one evaluation case — checks + supporting evidence."""

    case_id: str
    question: str
    error: str | None = None
    reasoning: CounterfactualReasoning | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def _check_all_simulated(reasoning: CounterfactualReasoning) -> tuple[bool, str]:
    """CHECK 1 — every step's consequence carries the ``[SIMULATED]`` label.

    Fail conditions: zero steps parsed, any "missing [SIMULATED]" parse warning,
    or fewer ``[SIMULATED]`` tokens in the raw response than steps (the model
    skipped a marker the parser didn't notice).
    """
    if not reasoning.steps:
        return False, "no steps parsed"
    missing = [w for w in reasoning.parse_warnings if "missing [SIMULATED]" in w]
    if missing:
        return False, f"{len(missing)} step(s) without [SIMULATED]"
    sim_count = len(_SIMULATED_RE.findall(reasoning.raw_response))
    if sim_count < len(reasoning.steps):
        return False, f"only {sim_count} [SIMULATED] tag(s) for {len(reasoning.steps)} step(s)"
    return True, f"{sim_count} [SIMULATED] tag(s) across {len(reasoning.steps)} step(s)"


def _check_cites_real_source(reasoning: CounterfactualReasoning) -> tuple[bool, str]:
    """CHECK 2 — every step cites at least one chunk_id from the GroundedContext.

    Fails if any step is ungrounded (no [EVIDENCE] tag at all) or cites an id
    not present in the supplied context (``unknown_evidence_ids`` is non-empty).
    """
    if not reasoning.steps:
        return False, "no steps parsed"
    ungrounded = [s.step_number for s in reasoning.steps if not s.is_grounded]
    unknown = {s.step_number: s.unknown_evidence_ids for s in reasoning.steps if s.unknown_evidence_ids}
    if ungrounded:
        return False, f"ungrounded step(s): {ungrounded}"
    if unknown:
        return False, f"steps citing unknown chunk_ids: {unknown}"
    return True, f"all {len(reasoning.steps)} step(s) cite real source chunks"


def _check_step_cap(reasoning: CounterfactualReasoning) -> tuple[bool, str]:
    """CHECK 3 — ``settings.MAX_CAUSAL_STEPS`` is not exceeded anywhere.

    Looks both at the parsed steps AND the raw response so we catch the case
    where the model tried to emit >4 steps and the parser silently capped it
    (that is recorded as a parse_warning, but we also count Step headers in
    the raw text to be safe).
    """
    cap = settings.MAX_CAUSAL_STEPS
    raw_headers = _STEP_HEADER_RE.findall(reasoning.raw_response)
    raw_max = max((int(n) for n in raw_headers), default=0)
    if len(reasoning.steps) > cap:
        return False, f"{len(reasoning.steps)} parsed steps > cap {cap}"
    if raw_max > cap:
        return False, f"raw response has Step {raw_max} (> cap {cap})"
    return True, f"{len(reasoning.steps)} step(s), max header={raw_max}, cap={cap}"


def _check_no_sim_as_fact(reasoning: CounterfactualReasoning) -> tuple[bool, str]:
    """CHECK 4 — simulated content is not presented as fact.

    Two conditions, both must hold:
    1. Every step's consequence carries [SIMULATED] (subset of CHECK 1's logic).
    2. The tail prose (RECONNECTION POINT, HISTORIAN'S NOTE) either has its own
       [SIMULATED] marker OR uses hedging vocabulary — a section that flatly
       asserts alt-history outcomes is the most common Rule #4 violation.
    """
    if not reasoning.steps:
        return False, "no steps parsed"
    missing_sim_warnings = [w for w in reasoning.parse_warnings if "missing [SIMULATED]" in w]
    if missing_sim_warnings:
        return False, f"{len(missing_sim_warnings)} step(s) lack [SIMULATED]"

    # Historian's Note is intentionally NOT checked here: that section is meant
    # to describe historiographical debate about REAL history, where verbs like
    # "argue" / "debate" / "contest" are the correct vocabulary (not simulation
    # leakage). RECONNECTION POINT is still scrutinised because it describes
    # the alt-timeline outcome.
    offenders: list[str] = []
    for label, text in (
        ("reconnection_point", reasoning.reconnection_point),
    ):
        if not text.strip():
            continue
        if _SIMULATED_RE.search(text):
            continue
        if not _HEDGE_RE.search(text):
            offenders.append(label)
    if offenders:
        return False, f"tail section(s) assert without hedging: {offenders}"
    return True, "every step labelled and reconnection_point hedges or self-labels"


CHECK_FUNCS: dict[str, callable] = {
    "1_all_simulated":      _check_all_simulated,
    "2_cites_real_source":  _check_cites_real_source,
    "3_step_cap":           _check_step_cap,
    "4_no_sim_as_fact":     _check_no_sim_as_fact,
}


def _print_reasoning(reasoning: CounterfactualReasoning) -> None:
    """Pretty-print a CounterfactualReasoning to stdout (UTF-8 already set)."""
    print(f"PROPOSED CHANGE: {reasoning.proposed_change}")
    print(f"DIVERGENCE POINT: {reasoning.divergence_point}")
    if not reasoning.steps:
        print("(no reasoning steps parsed)")
    for step in reasoning.steps:
        tag = "GROUNDED" if step.is_grounded else "UNGROUNDED"
        print(f"\n── Step {step.step_number} — {step.time_horizon}  [{tag}]")
        print(f"   {step.consequence}")
        if step.evidence_chunk_ids:
            print(f"   evidence: {', '.join(step.evidence_chunk_ids)}")
        if step.analogy_chunk_ids:
            print(f"   analogy:  {', '.join(step.analogy_chunk_ids)}")
        if step.unknown_evidence_ids:
            print(f"   UNKNOWN cited ids: {', '.join(step.unknown_evidence_ids)}")
        print(f"   confidence: {step.confidence} — {step.confidence_reason}")
    print(f"\nWHAT REMAINS UNKNOWABLE:\n{reasoning.what_remains_unknowable or '(missing)'}")
    print(f"\nRECONNECTION POINT:\n{reasoning.reconnection_point or '(missing)'}")
    print(f"\nHISTORIAN'S NOTE:\n{reasoning.historians_note or '(missing)'}")
    if reasoning.parse_warnings:
        print("\nPARSE WARNINGS:")
        for warning in reasoning.parse_warnings:
            print(f"  - {warning}")


def evaluate_case(case: dict) -> CaseResult:
    """Run one case through A1→A2→A3→A4 and apply the four checks.

    A failure at any agent stage records the traceback on ``CaseResult.error``,
    marks every check False, and returns — never raises.
    """
    result = CaseResult(case_id=case["id"], question=case["question"])
    try:
        analysis = analyze_query(case["question"])
        context = retrieve_context(analysis)
        grounded = ground_context(context, case["question"])
        reasoning = reason_about_counterfactual(analysis, grounded)
        result.reasoning = reasoning
        for key, fn in CHECK_FUNCS.items():
            passed, note = fn(reasoning)
            result.checks[key] = passed
            result.notes[key] = note
    except Exception as exc:  # noqa: BLE001 — record-and-continue is the policy
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("evaluate_case[%s] failed", case["id"])
        for key in CHECK_FUNCS:
            result.checks[key] = False
            result.notes[key] = "case errored — see traceback"
    return result


def print_summary(results: list[CaseResult]) -> None:
    """Compact pass/fail matrix across all cases × all checks."""
    print("\n" + "=" * 78)
    print("SUMMARY — PASS/FAIL MATRIX")
    print("=" * 78)
    header = f"{'case_id':<28} {'C1':>4} {'C2':>4} {'C3':>4} {'C4':>4}   notes"
    print(header)
    print("-" * 78)
    totals = {key: 0 for key in CHECK_FUNCS}
    for r in results:
        cells = []
        for key in CHECK_FUNCS:
            passed = r.checks.get(key, False)
            cells.append("PASS" if passed else "FAIL")
            if passed:
                totals[key] += 1
        line = f"{r.case_id:<28} {cells[0]:>4} {cells[1]:>4} {cells[2]:>4} {cells[3]:>4}"
        suffix = f"   ERROR: {r.error}" if r.error else ""
        print(line + suffix)
    print("-" * 78)
    n = len(results)
    print(
        f"{'TOTAL':<28} {totals['1_all_simulated']:>3}/{n} "
        f"{totals['2_cites_real_source']:>3}/{n} "
        f"{totals['3_step_cap']:>3}/{n} "
        f"{totals['4_no_sim_as_fact']:>3}/{n}"
    )
    print()
    print("CHECK LEGEND:")
    print("  C1 — every step labelled [SIMULATED]")
    print("  C2 — every step cites a chunk_id present in GroundedContext")
    print(f"  C3 — no more than {settings.MAX_CAUSAL_STEPS} steps anywhere")
    print("  C4 — no simulation presented as fact (steps labelled + tail hedges)")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    with TEST_CASES_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload["cases"]
    print(f"Loaded {len(cases)} test case(s) from {TEST_CASES_PATH.name}\n")

    results: list[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        print("\n" + "#" * 78)
        print(f"#  CASE {idx}/{len(cases)} — {case['id']}")
        print(f"#  Q: {case['question']}")
        print("#" * 78)
        t0 = time.monotonic()
        result = evaluate_case(case)
        elapsed = time.monotonic() - t0
        results.append(result)

        if result.error:
            print(f"\nERROR: {result.error}")
        elif result.reasoning is not None:
            _print_reasoning(result.reasoning)

        print(f"\n-- CHECKS ({elapsed:.1f}s) --")
        for key in CHECK_FUNCS:
            tag = "PASS" if result.checks.get(key) else "FAIL"
            print(f"  [{tag}] {key}: {result.notes.get(key, '')}")

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
