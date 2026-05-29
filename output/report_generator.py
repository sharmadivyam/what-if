"""Report generator — final output formatting.

Renders the pipeline's structured result into a human-readable report that ENFORCES
the verified/simulated separation in the presentation (Critical Rule #1).

Responsibilities:
- Take the scored reasoning (Agent 5's ``ScoredReasoning``) and the grounded context
  (Agent 3's ``GroundedContext``) — or a whole ``HistoriosState`` via
  ``report_from_state``.
- Produce a report with two CLEARLY SEPARATED sections:
    1. VERIFIED FACTS — each with its citing ``chunk_id``, source title, and URL.
    2. SIMULATED CONSEQUENCES — each causal step with its calibrated confidence
       level and the scorer's explanation.
- Never blend the two; the section split + banner make the boundary obvious.
- Render error / empty-corpus states as an honest notice rather than fabricated
  content (Critical Rules #5/#6).
- Return BOTH a structured object (``HistoriosReport``) and a display-ready Markdown
  string (``HistoriosReport.markdown``) for the Streamlit frontend.

Design notes:
- This module depends ONLY on the agent Pydantic models, never on the pipeline
  module — so there is no import cycle (the pipeline may import this, not vice
  versa). ``report_from_state`` accepts a plain mapping (the ``HistoriosState``
  dict) and unpacks the fields it needs, keeping the coupling structural-only.
- No LLM call and no network — pure formatting (Critical Rule #7 trivially met).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from agents.confidence_scorer import ScoredReasoning, ScoredStep
from agents.grounding_layer import GroundedContext, GroundedFact

logger = logging.getLogger(__name__)

# Emoji badge per calibrated level — purely presentational, keeps the boundary and
# the confidence tier scannable in the rendered Markdown.
_LEVEL_BADGE: dict[str, str] = {
    "HIGH": "🟢 HIGH",
    "MEDIUM": "🟡 MEDIUM",
    "LOW": "🟠 LOW",
    "SPECULATIVE": "🔴 SPECULATIVE",
}

_BANNER = (
    "> ⚠️ **This report separates VERIFIED historical facts from SIMULATED "
    "consequences.** Verified facts are extracted from cited sources; simulated "
    "consequences are model-generated counterfactual reasoning, each scored by how "
    "well the evidence supports it. Do not read the simulated section as historical "
    "fact."
)

_DYNAMIC_NOTE = (
    "> 🌐 *Augmented with a live Wikipedia fetch — the local corpus had no matching "
    "sources for this scenario, so context was retrieved on demand. Cited sources "
    "below are still real Wikipedia articles.*"
)


def _used_dynamic_sources(grounded: GroundedContext | None) -> bool:
    """True if any retrieved chunk was tagged ``source="wikipedia_dynamic"``.

    Looks up the provenance of every chunk in ``grounded.source_map`` (all chunks
    retrieved for this run) against ChromaDB's stored metadata. Defensive: any
    failure (empty store, lookup error) is swallowed and treated as "curated", so
    provenance reporting can never break report generation.
    """
    if grounded is None or not grounded.source_map:
        return False
    try:
        from vectorstore.chroma_client import get_metadata

        meta = get_metadata(list(grounded.source_map.keys()))
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        logger.debug("provenance lookup failed; assuming curated", exc_info=True)
        return False
    return any(m.get("source") == "wikipedia_dynamic" for m in meta.values())


class HistoriosReport(BaseModel):
    """Structured report + display-ready Markdown for one counterfactual run.

    The structured fields let programmatic consumers (tests, the evaluator) inspect
    the report without parsing text; ``markdown`` is the rendered string the
    frontend displays. ``status`` mirrors the pipeline's ("ok" / "no_context" /
    "error"); ``error`` is populated only when the run failed.
    """

    proposed_change: str = ""
    divergence_point: str = ""
    status: str = "ok"
    overall_confidence: str | None = None
    verified_facts: list[GroundedFact] = Field(default_factory=list)
    simulated_steps: list[ScoredStep] = Field(default_factory=list)
    what_remains_unknowable: str = ""
    reconnection_point: str = ""
    historians_note: str = ""
    augmented_with_dynamic: bool = False  # True if any source was live-fetched
    error: str | None = None
    markdown: str = ""


def _render_verified_section(grounded: GroundedContext | None) -> str:
    """Render the VERIFIED FACTS section — each fact cited to its source chunk."""
    facts = grounded.verified_facts if grounded else []
    lines = [f"## ✅ Verified Facts ({len(facts)})", ""]
    if not facts:
        lines.append(
            "_No verified facts were retrieved for this scenario — the corpus did "
            "not contain directly supporting sources. Treat the entire simulated "
            "section below as especially speculative._"
        )
        return "\n".join(lines)

    source_map = grounded.source_map if grounded else {}
    for fact in facts:
        url = source_map.get(fact.source_chunk_id, "")
        cite = f"`[{fact.source_chunk_id}]` — *{fact.source_title}*"
        if url:
            cite += f" ([source]({url}))"
        lines.append(f"- {fact.claim}")
        lines.append(f"  - {cite}")
    return "\n".join(lines)


def _render_simulated_section(steps: list[ScoredStep]) -> str:
    """Render the SIMULATED CONSEQUENCES section — one block per scored step."""
    lines = [f"## 🔮 Simulated Consequences ({len(steps)})", ""]
    if not steps:
        lines.append(
            "_No simulated consequences were produced (no grounded context to "
            "reason from, or the reasoning step returned nothing)._"
        )
        return "\n".join(lines)

    for step in steps:
        badge = _LEVEL_BADGE.get(step.confidence_level, step.confidence_level)
        lines.append(f"### Step {step.step_number} — {step.time_horizon}")
        lines.append(f"**Confidence:** {badge}")
        lines.append("")
        lines.append(f"[SIMULATED] {_strip_simulated(step.consequence)}")
        lines.append("")
        lines.append(f"- *Why this confidence:* {step.confidence_explanation}")
        if step.evidence_chunk_ids:
            ids = ", ".join(f"`{cid}`" for cid in step.evidence_chunk_ids)
            lines.append(f"- *Evidence cited:* {ids}")
        else:
            lines.append("- *Evidence cited:* _none (ungrounded)_")
        if step.unknown_evidence_ids:
            bad = ", ".join(f"`{cid}`" for cid in step.unknown_evidence_ids)
            lines.append(f"- ⚠️ *Unrecognised citation(s):* {bad}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _strip_simulated(text: str) -> str:
    """Drop a leading ``[SIMULATED]`` marker so we don't print it twice."""
    stripped = text.lstrip()
    if stripped[:11].upper() == "[SIMULATED]":
        return stripped[11:].lstrip()
    return text


def _render_tail(report: HistoriosReport) -> str:
    """Render the three closing prose sections, skipping any that are empty."""
    blocks: list[str] = []
    for title, body in (
        ("What Remains Unknowable", report.what_remains_unknowable),
        ("Reconnection Point", report.reconnection_point),
        ("Historian's Note", report.historians_note),
    ):
        if body and body.strip():
            blocks.append(f"## {title}\n\n{body.strip()}")
    return "\n\n".join(blocks)


def generate_report(
    scored: ScoredReasoning | None,
    grounded: GroundedContext | None,
    *,
    error: str | None = None,
    status: str = "ok",
    timings: dict[str, float] | None = None,
) -> HistoriosReport:
    """Build a ``HistoriosReport`` (structured fields + Markdown) from agent output.

    Args:
        scored: Agent 5's ``ScoredReasoning`` (or ``None`` if the run never reached
            scoring — e.g. an error or empty corpus).
        grounded: Agent 3's ``GroundedContext`` (supplies the VERIFIED section and
            the source URLs); may be ``None``.
        error: An error message if the run failed; renders an error notice.
        status: Pipeline status — "ok" / "no_context" / "error".
        timings: Optional per-node timings; appended as a small footer when present.

    Returns:
        A ``HistoriosReport`` whose ``markdown`` field is ready to display.
    """
    report = HistoriosReport(
        status=status,
        error=error,
        proposed_change=scored.proposed_change if scored else "",
        divergence_point=scored.divergence_point if scored else "",
        overall_confidence=scored.overall_confidence if scored else None,
        verified_facts=grounded.verified_facts if grounded else [],
        simulated_steps=scored.steps if scored else [],
        what_remains_unknowable=scored.what_remains_unknowable if scored else "",
        reconnection_point=scored.reconnection_point if scored else "",
        historians_note=scored.historians_note if scored else "",
        augmented_with_dynamic=_used_dynamic_sources(grounded),
    )

    # --- Error state: honest notice, no fabricated content (Rules #5/#6) ---------
    if status == "error" or error:
        title = report.proposed_change or "this counterfactual"
        report.markdown = (
            f"# Counterfactual Report\n\n"
            f"> ❌ **The pipeline could not complete for {title}.**\n\n"
            f"**Error:** `{error or 'unknown error'}`\n\n"
            f"No report was generated. This is a graceful failure — no simulated "
            f"content is shown because the run did not finish."
        )
        return report

    # --- Header ------------------------------------------------------------------
    parts: list[str] = []
    heading = report.proposed_change or "Counterfactual scenario"
    parts.append(f"# Counterfactual: {heading}")
    meta: list[str] = []
    if report.divergence_point:
        meta.append(f"**Divergence point:** {report.divergence_point}")
    if report.overall_confidence:
        badge = _LEVEL_BADGE.get(report.overall_confidence, report.overall_confidence)
        meta.append(f"**Overall confidence (weakest link):** {badge}")
    if meta:
        parts.append("  \n".join(meta))
    parts.append(_BANNER)
    if report.augmented_with_dynamic:
        parts.append(_DYNAMIC_NOTE)

    # --- The two separated sections ---------------------------------------------
    parts.append(_render_verified_section(grounded))
    parts.append(_render_simulated_section(report.simulated_steps))

    # --- Tail prose --------------------------------------------------------------
    tail = _render_tail(report)
    if tail:
        parts.append(tail)

    # --- Optional timings footer -------------------------------------------------
    if timings:
        shown = ", ".join(
            f"{k}={v:.2f}s" for k, v in timings.items()
        )
        parts.append(f"---\n\n*Pipeline timings: {shown}*")

    report.markdown = "\n\n".join(parts)
    logger.info(
        "generate_report: status=%s, %d verified fact(s), %d simulated step(s)",
        status,
        len(report.verified_facts),
        len(report.simulated_steps),
    )
    return report


def report_from_state(state: Mapping[str, Any]) -> HistoriosReport:
    """Build a report directly from a ``HistoriosState`` mapping (pipeline output).

    Convenience for the frontend: unpacks the ``scored`` / ``grounded`` / ``error``
    / ``status`` / ``timings`` keys and delegates to ``generate_report``. Tolerates
    a partial state (missing keys default sensibly).
    """
    return generate_report(
        scored=state.get("scored"),
        grounded=state.get("grounded"),
        error=state.get("error"),
        status=state.get("status", "ok"),
        timings=state.get("timings"),
    )


if __name__ == "__main__":
    # Fully OFFLINE smoke test — no API/network. Builds a hand-made GroundedContext +
    # ScoredReasoning fixture (exercising all four confidence tiers) and prints the
    # rendered Markdown. Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m output.report_generator
    import sys

    # Markdown carries em-dashes / emoji; Windows console defaults to cp1252 — see
    # CLAUDE.md "WINDOWS CONSOLE ENCODING".
    sys.stdout.reconfigure(encoding="utf-8")

    from agents.confidence_scorer import score_reasoning
    from agents.reasoning_agent import CounterfactualReasoning, ReasoningStep

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    def _vf(cid: str, claim: str) -> GroundedFact:
        return GroundedFact(
            claim=claim, source_chunk_id=cid, source_title="Mughal Empire", confidence_basis="test"
        )

    grounded = GroundedContext(
        verified_facts=[
            _vf("mughal_empire_22", "India produced 24.5% of world manufacturing output until 1750."),
            _vf("mughal_empire_22", "The Mughal state ran a sophisticated revenue administration."),
            _vf("mughal_empire_15", "Mughal administrative reach extended across the subcontinent."),
        ],
        analogies=[_vf("meiji_restoration_4", "Japan modernised rapidly while preserving sovereignty.")],
        background_context=[_vf("mughal_empire_31", "The Mughal court patronised the arts.")],
        source_map={
            "mughal_empire_22": "https://en.wikipedia.org/wiki/Mughal_Empire",
            "mughal_empire_15": "https://en.wikipedia.org/wiki/Mughal_Empire",
            "meiji_restoration_4": "https://en.wikipedia.org/wiki/Meiji_Restoration",
            "mughal_empire_31": "https://en.wikipedia.org/wiki/Mughal_Empire",
        },
    )

    def _step(n, horizon, cons, ev, unknown, conf):
        return ReasoningStep(
            step_number=n, time_horizon=horizon, consequence=cons, evidence_chunk_ids=ev,
            confidence=conf, confidence_reason="test", is_grounded=bool(ev), unknown_evidence_ids=unknown,
        )

    reasoning = CounterfactualReasoning(
        proposed_change="The Mughal Empire industrialized before the British arrived.",
        divergence_point="1700s — South Asia",
        steps=[
            _step(1, "Immediate (0-10y)", "[SIMULATED] Early factory production emerges.",
                  ["mughal_empire_22", "mughal_empire_15"], [], "High"),
            _step(2, "Short-term (10-30y)", "[SIMULATED] Selective modernisation, Meiji-style.",
                  ["meiji_restoration_4"], [], "Medium"),
            _step(3, "Medium-term (30-100y)", "[SIMULATED] Cultural and economic shifts ripple outward.",
                  ["mughal_empire_31"], [], "Low"),
            _step(4, "Long-term (100+y)", "[SIMULATED] A wholly different global balance of power.",
                  ["fabricated_chunk_99"], ["fabricated_chunk_99"], "Low"),
        ],
        what_remains_unknowable="Whether industrial momentum could survive Mughal succession crises.",
        reconnection_point="The timeline likely diverges permanently from actual history by the 19th century.",
        historians_note="Historians debate the primary causes of Mughal decline.",
        raw_response="(offline test)",
    )

    scored = score_reasoning(reasoning, grounded)
    report = generate_report(scored, grounded, status="ok", timings={"understand_query": 1.2, "reason": 3.4})

    print("\n" + "=" * 78)
    print(report.markdown)
    print("=" * 78)
    print(f"\n[structured] status={report.status} overall={report.overall_confidence} "
          f"verified={len(report.verified_facts)} steps={len(report.simulated_steps)}")
