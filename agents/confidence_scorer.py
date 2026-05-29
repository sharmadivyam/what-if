"""Agent 5 — Confidence Scorer.

Final node in the pipeline. Assigns a calibrated confidence LEVEL to each
simulated consequence in the causal chain produced by Agent 4, plus a detailed,
human-readable explanation of *why* — so the report generator can show users how
well-supported each SIMULATED step actually is.

Responsibilities:
- Take the causal chain from the Reasoning Agent (``CounterfactualReasoning``)
  plus the grounded facts it was built on (``GroundedContext``).
- Score each simulated step into one of four tiers based purely on how its cited
  ``chunk_id``s map onto the grounded buckets (Critical Rule #1: VERIFIED facts
  stay at full confidence; only the SIMULATED consequences get a score).
- Return a Pydantic ``ScoredReasoning`` (Critical Rule #4), never raw strings.

Scoring rubric (evidence-count based — see ``_score_step``):
- HIGH        — backed by 2+ verified facts (claims from ``verified_facts`` whose
                chunk the step cites).
- MEDIUM      — backed by exactly 1 verified fact, OR by >=1 analogy when no
                verified fact is cited (reasoning by comparison).
- LOW         — no verified fact and no analogy, but cites >=1 debated/background
                fact: plausible inference, thinly grounded.
- SPECULATIVE — no real evidence: ungrounded (no [EVIDENCE] at all) or every
                cited id is unknown/fabricated. Clearly flagged.
Decision precedence is top-to-bottom; the first matching tier wins.

Implementation notes:
- NO LLM CALL. The score is pure logic over the evidence counts already present
  in ``GroundedContext`` and on each ``ReasoningStep`` (its ``evidence_chunk_ids``
  / ``unknown_evidence_ids`` / ``is_grounded``). This keeps Agent 5 deterministic,
  free, and outside the Cerebras free-tier rate-limit budget (see CLAUDE.md). It
  also trivially satisfies Critical Rule #7 — no provider client is instantiated.
- "2+ verified facts" counts FACTS (claims), not distinct chunks: one chunk can
  yield several verified claims in ``GroundedContext``, and the support genuinely
  comes from each of them. A step citing one rich chunk that backs >=2 verified
  claims therefore reaches HIGH.
- The model's OWN self-reported ``confidence`` (High/Medium/Low/Unknown, parsed by
  Agent 4 from the LLM's "Confidence:" line) is preserved on each step and echoed
  in the explanation for transparency — but it does NOT drive the calibrated level.
  This layer recomputes confidence from evidence, independent of what the model
  claimed about itself.
- EMPTY STATE (Critical Rule #5): an empty causal chain (e.g. Agent 4 parsed zero
  steps) yields a valid ``ScoredReasoning`` with no steps, an empty distribution,
  and ``overall_confidence=None`` — never raises.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field

from agents.grounding_layer import GroundedContext
from agents.reasoning_agent import CounterfactualReasoning, ReasoningStep

logger = logging.getLogger(__name__)

# The four calibrated confidence tiers. Defined once so the Literal type, the
# scoring logic, and the aggregation ordering all stay in sync.
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW", "SPECULATIVE"]

# Weakest -> strongest. Used for the chain-level ``overall_confidence`` (a causal
# chain is only as reliable as its least-supported link) and for the distribution
# print order.
_LEVEL_ORDER: tuple[ConfidenceLevel, ...] = ("SPECULATIVE", "LOW", "MEDIUM", "HIGH")


class ScoredStep(ReasoningStep):
    """A reasoning step plus its calibrated confidence.

    Subclasses Agent 4's ``ReasoningStep`` so every original field (consequence,
    ``evidence_chunk_ids``, ``unknown_evidence_ids``, the model's self-reported
    ``confidence`` / ``confidence_reason``, ``is_grounded`` …) is carried through
    unchanged. Adds the scorer's verdict:

    - ``confidence_level``       — the calibrated HIGH/MEDIUM/LOW/SPECULATIVE tier.
    - ``confidence_explanation`` — why, naming the specific chunk_ids and counts.
    """

    confidence_level: ConfidenceLevel
    confidence_explanation: str


class ScoredReasoning(CounterfactualReasoning):
    """Agent 4's output with every step scored, plus chain-level aggregates.

    Subclasses ``CounterfactualReasoning`` and overrides ``steps`` to hold
    ``ScoredStep``s; all other fields (``proposed_change``, ``divergence_point``,
    the three tail sections, ``raw_response``, ``parse_warnings``) pass straight
    through. The two aggregate fields summarise the chain for the report generator.
    """

    steps: list[ScoredStep] = Field(default_factory=list)
    # Count of steps at each level, e.g. {"HIGH": 1, "MEDIUM": 2, "LOW": 0,
    # "SPECULATIVE": 1}. Keys are always all four levels (zero-filled).
    confidence_distribution: dict[str, int] = Field(default_factory=dict)
    # The WEAKEST level present across steps — a causal chain is only as reliable
    # as its least-supported link. ``None`` when there are no steps.
    overall_confidence: ConfidenceLevel | None = None


class _BucketIndex:
    """Per-call lookup of which chunk_ids back which kind of grounded fact.

    Built once from a ``GroundedContext``. Counts FACTS (claims), so a chunk that
    yields multiple verified claims contributes that many to ``verified_backers``
    for the step that cites it — matching the "2+ verified facts" rubric.
    """

    def __init__(self, grounded: GroundedContext) -> None:
        self.verified_ids = {f.source_chunk_id for f in grounded.verified_facts}
        self.analogy_ids = {f.source_chunk_id for f in grounded.analogies}
        self.debated_ids = {f.source_chunk_id for f in grounded.debated_points}
        self.background_ids = {f.source_chunk_id for f in grounded.background_context}
        self.known_ids = (
            self.verified_ids | self.analogy_ids | self.debated_ids | self.background_ids
        )

    def verified_backers(self, cited: set[str]) -> list[str]:
        """chunk_ids (cited) that back >=1 verified fact, listed once each."""
        return [cid for cid in cited if cid in self.verified_ids]

    def analogy_backers(self, cited: set[str]) -> list[str]:
        return [cid for cid in cited if cid in self.analogy_ids]

    def contextual_backers(self, cited: set[str]) -> list[str]:
        """Cited ids that are only debated/background (thin-but-real grounding)."""
        return [cid for cid in cited if cid in (self.debated_ids | self.background_ids)]


def _count_verified_facts(cited: set[str], grounded: GroundedContext) -> int:
    """Number of verified FACTS (claims) whose source chunk the step cites.

    This is the count that drives HIGH vs MEDIUM — one chunk can back several
    verified claims, and each is independent support.
    """
    return sum(1 for f in grounded.verified_facts if f.source_chunk_id in cited)


def _count_analogy_facts(cited: set[str], grounded: GroundedContext) -> int:
    return sum(1 for f in grounded.analogies if f.source_chunk_id in cited)


def _fmt_ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "(none)"


def _score_step(
    step: ReasoningStep, index: _BucketIndex, grounded: GroundedContext
) -> tuple[ConfidenceLevel, str]:
    """Apply the rubric to one step. Returns (level, detailed explanation).

    Pure function — no I/O, no LLM. Looks only at the step's cited chunk_ids and
    how they map onto the grounded buckets.
    """
    cited = set(step.evidence_chunk_ids)
    verified_facts = _count_verified_facts(cited, grounded)
    analogy_facts = _count_analogy_facts(cited, grounded)
    verified_chunks = index.verified_backers(cited)
    analogy_chunks = index.analogy_backers(cited)
    contextual_chunks = index.contextual_backers(cited)
    unknown = list(step.unknown_evidence_ids)

    self_report = f" (model self-reported: {step.confidence})"

    # --- HIGH: 2+ verified facts ---------------------------------------------
    if verified_facts >= 2:
        explanation = (
            f"HIGH — directly backed by {verified_facts} verified fact(s) drawn "
            f"from chunk(s) {_fmt_ids(verified_chunks)}; multiple verified sources "
            f"corroborate this consequence."
        )
        if analogy_chunks:
            explanation += f" Reinforced by analogy from chunk(s) {_fmt_ids(analogy_chunks)}."
        if unknown:
            explanation += (
                f" NOTE: also cites unrecognised chunk_id(s) {_fmt_ids(unknown)} "
                f"absent from the grounded context."
            )
        return "HIGH", explanation + self_report

    # --- MEDIUM: exactly 1 verified fact -------------------------------------
    if verified_facts == 1:
        explanation = (
            f"MEDIUM — supported by a single verified fact from chunk "
            f"{_fmt_ids(verified_chunks)}; plausible and grounded, but resting on "
            f"one piece of verified evidence."
        )
        if analogy_chunks:
            explanation += f" Aided by analogy from chunk(s) {_fmt_ids(analogy_chunks)}."
        if unknown:
            explanation += (
                f" NOTE: also cites unrecognised chunk_id(s) {_fmt_ids(unknown)}."
            )
        return "MEDIUM", explanation + self_report

    # --- MEDIUM: analogy-only (no verified fact) -----------------------------
    if analogy_facts >= 1:
        explanation = (
            f"MEDIUM — no verified fact is cited, but the step is grounded in "
            f"{analogy_facts} historical analog(y/ies) from chunk(s) "
            f"{_fmt_ids(analogy_chunks)}; reasoning by comparison rather than "
            f"direct evidence."
        )
        if unknown:
            explanation += (
                f" NOTE: also cites unrecognised chunk_id(s) {_fmt_ids(unknown)}."
            )
        return "MEDIUM", explanation + self_report

    # --- LOW: only debated/background grounding ------------------------------
    if contextual_chunks:
        explanation = (
            f"LOW — thin evidence: cites only contextual/contested source(s) "
            f"{_fmt_ids(contextual_chunks)} (background or debated), with no "
            f"verified fact or analogy. A plausible inference, weakly grounded."
        )
        if unknown:
            explanation += (
                f" NOTE: also cites unrecognised chunk_id(s) {_fmt_ids(unknown)}."
            )
        return "LOW", explanation + self_report

    # --- SPECULATIVE: no real evidence ---------------------------------------
    if not step.is_grounded and not cited:
        explanation = (
            "SPECULATIVE — no [EVIDENCE] citation at all; this consequence is an "
            "unsupported inference and is flagged as speculative (Rule #1 boundary "
            "preserved)."
        )
    else:
        # Cited something, but nothing maps to a real grounded fact — i.e. every
        # cited id is unknown/fabricated.
        explanation = (
            f"SPECULATIVE — the only cited id(s) {_fmt_ids(unknown or list(cited))} "
            f"are not present in the grounded context (fabricated or unrecognised); "
            f"no verifiable evidence backs this step. Flagged as speculative."
        )
    return "SPECULATIVE", explanation + self_report


def score_reasoning(
    reasoning: CounterfactualReasoning, grounded: GroundedContext
) -> ScoredReasoning:
    """Score every step of a causal chain. Pure logic — makes no LLM call.

    Each ``ReasoningStep`` is classified into HIGH / MEDIUM / LOW / SPECULATIVE
    based on how its cited chunk_ids map onto ``grounded``'s buckets (see the
    module docstring for the rubric), and turned into a ``ScoredStep`` carrying a
    detailed ``confidence_explanation``. All other ``CounterfactualReasoning``
    fields pass through unchanged. Chain-level aggregates (``confidence_distribution``
    and ``overall_confidence`` — the weakest level present) are computed last.

    On an empty chain the returned ``ScoredReasoning`` has no steps, a zero-filled
    distribution, and ``overall_confidence=None`` (Critical Rule #5) — never raises.

    Args:
        reasoning: Agent 4's ``CounterfactualReasoning`` (the causal chain + tails).
        grounded: Agent 3's ``GroundedContext`` (the evidence the chain cites).

    Returns:
        A validated ``ScoredReasoning``.
    """
    index = _BucketIndex(grounded)

    scored_steps: list[ScoredStep] = []
    distribution: dict[str, int] = {level: 0 for level in _LEVEL_ORDER}
    for step in reasoning.steps:
        level, explanation = _score_step(step, index, grounded)
        distribution[level] += 1
        scored_steps.append(
            ScoredStep(
                **step.model_dump(),
                confidence_level=level,
                confidence_explanation=explanation,
            )
        )

    # Weakest link sets the chain's overall confidence. None when no steps.
    overall: ConfidenceLevel | None = None
    if scored_steps:
        overall = next(
            level for level in _LEVEL_ORDER if distribution[level] > 0
        )

    # Carry every other CounterfactualReasoning field through unchanged; ``steps``
    # is excluded from the dump because we replace it with the scored versions.
    scored = ScoredReasoning(
        **reasoning.model_dump(exclude={"steps"}),
        steps=scored_steps,
        confidence_distribution=distribution,
        overall_confidence=overall,
    )

    logger.info(
        "score_reasoning: %d step(s) scored — HIGH=%d MEDIUM=%d LOW=%d SPECULATIVE=%d "
        "| overall=%s",
        len(scored_steps),
        distribution["HIGH"],
        distribution["MEDIUM"],
        distribution["LOW"],
        distribution["SPECULATIVE"],
        overall or "(none)",
    )
    return scored


if __name__ == "__main__":
    # Fully OFFLINE smoke test — no API key, no network, no ChromaDB. Builds a
    # hand-made chain + grounded context that exercises all four tiers, scores it,
    # and prints the result. Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m agents.confidence_scorer
    import sys

    # Explanations carry em-dashes; Windows console defaults to cp1252 (Known Issue
    # "WINDOWS CONSOLE ENCODING" in CLAUDE.md) — reconfigure before printing.
    sys.stdout.reconfigure(encoding="utf-8")

    from agents.grounding_layer import GroundedFact

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    def _vf(cid: str, claim: str) -> GroundedFact:
        return GroundedFact(
            claim=claim, source_chunk_id=cid, source_title="Test Source", confidence_basis="test"
        )

    grounded = GroundedContext(
        verified_facts=[
            _vf("mughal_empire_22", "India produced 24.5% of world manufacturing output until 1750."),
            _vf("mughal_empire_22", "The Mughal state ran a sophisticated revenue administration."),
            _vf("mughal_empire_15", "Mughal administrative reach extended across the subcontinent."),
        ],
        debated_points=[
            _vf("mughal_empire_30", "Historians debate the primary cause of Mughal decline."),
        ],
        background_context=[
            _vf("mughal_empire_31", "The Mughal court patronised the arts."),
        ],
        analogies=[
            _vf("meiji_restoration_4", "Japan modernised rapidly while preserving sovereignty."),
        ],
        source_map={},
    )

    def _step(n: int, horizon: str, cons: str, ev: list[str], unknown: list[str], conf: str) -> ReasoningStep:
        return ReasoningStep(
            step_number=n,
            time_horizon=horizon,
            consequence=cons,
            evidence_chunk_ids=ev,
            confidence=conf,  # type: ignore[arg-type]
            confidence_reason="test",
            is_grounded=bool(ev),
            unknown_evidence_ids=unknown,
        )

    reasoning = CounterfactualReasoning(
        proposed_change="The Mughal Empire industrialised before the British arrived.",
        divergence_point="1700s — South Asia",
        steps=[
            # HIGH: cites a chunk backing 2 verified facts + a 1-verified chunk.
            _step(1, "Immediate (0-10y)", "[SIMULATED] Early factory production emerges.",
                  ["mughal_empire_22", "mughal_empire_15"], [], "High"),
            # MEDIUM: analogy only, no verified fact.
            _step(2, "Short-term (10-30y)", "[SIMULATED] Selective modernisation, Meiji-style.",
                  ["meiji_restoration_4"], [], "Medium"),
            # LOW: only background/debated grounding.
            _step(3, "Medium-term (30-100y)", "[SIMULATED] Cultural shifts ripple outward.",
                  ["mughal_empire_31", "mughal_empire_30"], [], "Low"),
            # SPECULATIVE: only a fabricated citation.
            _step(4, "Long-term (100+y)", "[SIMULATED] A wholly different global order.",
                  ["fabricated_chunk_99"], ["fabricated_chunk_99"], "Low"),
        ],
        what_remains_unknowable="Whether industrial momentum could survive succession crises.",
        reconnection_point="The timeline likely diverges permanently from actual history.",
        historians_note="Historians debate the causes of Mughal decline.",
        raw_response="(omitted in offline test)",
    )

    scored = score_reasoning(reasoning, grounded)

    print(f"\nPROPOSED CHANGE: {scored.proposed_change}")
    print(f"DIVERGENCE POINT: {scored.divergence_point}\n")
    for s in scored.steps:
        print(f"── Step {s.step_number} — {s.time_horizon}  [{s.confidence_level}]")
        print(f"   {s.consequence}")
        print(f"   {s.confidence_explanation}\n")
    print(f"DISTRIBUTION: {scored.confidence_distribution}")
    print(f"OVERALL CONFIDENCE (weakest link): {scored.overall_confidence}")
