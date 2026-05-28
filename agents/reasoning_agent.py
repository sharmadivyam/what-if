"""Agent 4 — Reasoning Agent.

Fourth node in the pipeline. Simulates the consequences of the counterfactual
using multi-step causal reasoning, grounded strictly on the verified facts +
analogies produced by Agent 3 (Grounding Layer).

Responsibilities:
- Take the counterfactual premise (Agent 1's ``QueryAnalysis``) and the grounded
  context (Agent 3's ``GroundedContext``) — the only allowed inputs to the LLM
  (Critical Rule #6).
- Build a causal chain of consequences where each step cites the chunk_id of the
  evidence it depends on (Critical Rule #2).
- Enforce a HARD CAP of 4 causal reasoning steps as a hallucination guard
  (Critical Rule #3) — any further ``Step N`` blocks the model emits are
  dropped with a warning.
- Mark every produced consequence as SIMULATED, never as fact (Critical Rule #1).
- Return a ``CounterfactualReasoning`` Pydantic model (Critical Rule #4), never
  a raw string.

Implementation notes:
- LLM access goes through ``core.llm_client.get_llm_client()`` with
  ``model=settings.CEREBRAS_MODEL`` (Critical Rule #7). No provider client is
  instantiated here.
- TEMPERATURE: pinned at ``TEMPERATURE`` (0.3) — slightly creative but grounded.
  Distinct from ``settings.LLM_TEMPERATURE`` (0.0, used by the grounding layer
  where determinism matters) and from Agent 1's 0.1 (pure parsing). 0.0 would
  be too rigid for causal reasoning; >0.5 starts inventing.
- ONE LLM call per run (free-tier rate-limit awareness — see CLAUDE.md
  "CEREBRAS FREE-TIER RATE LIMIT" Known Issue). The prompt asks for ALL four
  steps + tail sections in a single completion; we do NOT do per-step calls and
  we do NOT do a corrective reparse retry (would double the budget for this
  agent and risks pushing a pipeline run past the 429 wall).
- STRUCTURED PROSE, NOT JSON: the prompt template's format is itself the
  alignment mechanism — ``[SIMULATED]`` / ``[EVIDENCE: chunk_id]`` markers,
  numbered steps, named tail sections. Switching to JSON mode would erase the
  template's rule reinforcement. We parse the structured response with regex
  and keep the full ``raw_response`` on the output for the Report Generator.
- EMPTY-CONTEXT SHORT-CIRCUIT (Rule #6): when both ``verified_facts`` and
  ``analogies`` are empty there is nothing to ground reasoning on, so we return
  an empty-but-valid ``CounterfactualReasoning`` with a warning and make NO LLM
  call. ``debated_points`` alone is not enough — those feed the Historian's
  Note, not the causal chain.
- UNGROUNDED-STEP FLAG: a step missing ``[EVIDENCE: ...]`` is KEPT but flagged
  ``is_grounded=False`` (per the user's spec — don't drop it, surface it).
  Any cited chunk_id that was NOT in the supplied grounded context is recorded
  on ``unknown_evidence_ids`` so the Confidence Scorer (Agent 5) can penalise
  fabricated citations without us silently discarding the consequence.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from agents.grounding_layer import GroundedContext, GroundedFact
from agents.query_understanding import QueryAnalysis
from config import settings
from core.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Pinned for the reasoning step. See module docstring for why this differs from
# settings.LLM_TEMPERATURE (0.0) and from Agent 1's TEMPERATURE (0.1).
TEMPERATURE = 0.3

Confidence = Literal["High", "Medium", "Low", "Unknown"]


class ReasoningStep(BaseModel):
    """One link in the causal chain — a single [SIMULATED] consequence.

    The text in ``consequence`` is what the LLM produced AFTER its
    ``[SIMULATED]`` marker. ``evidence_chunk_ids`` are the chunk_ids the model
    cited via ``[EVIDENCE: ...]`` in this step's block; ``analogy_chunk_ids``
    are the subset cited specifically on the "Analogy" line. ``is_grounded`` is
    False iff no ``[EVIDENCE: ...]`` tag was found at all (the user's spec:
    flag ungrounded steps rather than drop them). ``unknown_evidence_ids`` are
    cited ids that do NOT appear in the supplied ``GroundedContext`` — kept so
    Agent 5 can score citation hygiene.
    """

    step_number: int
    time_horizon: str  # header text, e.g. "Immediate Consequences (0-10 years)"
    consequence: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    analogy_chunk_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = "Unknown"
    confidence_reason: str = ""
    is_grounded: bool = False
    unknown_evidence_ids: list[str] = Field(default_factory=list)


class CounterfactualReasoning(BaseModel):
    """The reasoning agent's full output for one counterfactual scenario.

    ``steps`` is capped at ``settings.MAX_CAUSAL_STEPS`` (Critical Rule #3).
    ``raw_response`` keeps the full LLM text so the Report Generator can render
    the human-readable version even if our parser missed an edge case.
    ``parse_warnings`` surfaces parser issues (missing tail sections, extra
    steps dropped, no usable content, empty-context short-circuit) without
    raising — a single LLM call has no retry budget for reparse.
    """

    proposed_change: str
    divergence_point: str  # "{time_period} — {geography}"
    steps: list[ReasoningStep] = Field(default_factory=list)
    what_remains_unknowable: str = ""
    reconnection_point: str = ""
    historians_note: str = ""
    raw_response: str = ""
    parse_warnings: list[str] = Field(default_factory=list)


# --- Prompt template (verbatim from the user-supplied spec) ------------------
# The six substitution markers — {proposed_change}, {time_period}, {geography},
# {verified_facts}, {analogies}, {debated_points} — are injected via
# ``str.replace``. NOT an f-string and NOT ``.format``: the template also
# contains illustrative ``{consequence}`` / ``{chunk_id}`` / ``{High / Medium /
# Low}`` placeholders inside the example output, which MUST remain literal text
# in the prompt the LLM sees (they are part of the format spec). Treating
# braces as Python placeholders would either explode on those or require
# escaping every one of them.

_PROMPT_TEMPLATE = """\
You are a historical counterfactual reasoning engine.
Your job is to simulate what MIGHT have happened if
history had been different — while being completely
honest about what is simulation vs verified fact.

You have been given three inputs:
1. VERIFIED FACTS: real historical information extracted
   from sources, each with a source_chunk_id
2. HISTORICAL ANALOGIES: similar situations from other
   times and places, each with a source_chunk_id
3. THE PROPOSED CHANGE: what the user wants to be
   different in history

════════════════════════════════════════════════════════
STRICT RULES — NEVER VIOLATE THESE
════════════════════════════════════════════════════════

RULE 1 — Every simulated claim must be labeled [SIMULATED]
  Never write a consequence without this label.

RULE 2 — Every reasoning step must cite its evidence
  Use this exact format: [EVIDENCE: chunk_id]
  If you have no evidence for a claim — say so explicitly
  and mark it [LOW CONFIDENCE — no direct evidence].

RULE 3 — Maximum 4 reasoning steps
  Stop at 4. Do not extrapolate beyond step 4.
  Depth over breadth — one strong chain beats
  many weak ones.

RULE 4 — Never present simulation as fact
  Wrong: "The Mughal Empire would have industrialized."
  Right: "[SIMULATED] The Mughal Empire may have pursued
  industrialization, based on [EVIDENCE: mughal_empire_22]
  which shows India produced 24.5% of world manufacturing
  output until 1750, suggesting the economic base existed."

RULE 5 — Ground analogies explicitly
  When using an analogy, name it and explain WHY it applies.
  Wrong: "Similar to Japan, India would have modernized."
  Right: "[SIMULATED] A resilient Mughal state might have
  pursued selective modernization similar to the Meiji
  Restoration [EVIDENCE: meiji_restoration_4], where Japan
  maintained sovereignty by rapidly adopting Western
  technology while preserving political structure. The
  Mughal state had comparable administrative sophistication
  [EVIDENCE: mughal_empire_15]."

RULE 6 — Acknowledge what you cannot know
  At step 4, explicitly state what remains genuinely
  unknowable about this counterfactual and why.

════════════════════════════════════════════════════════
REASONING FORMAT — USE THIS EXACTLY
════════════════════════════════════════════════════════

PROPOSED CHANGE: {proposed_change}

DIVERGENCE POINT: {time_period} — {geography}

─────────────────────────────────────────
Step 1 — Immediate Consequences (0-10 years after change)
─────────────────────────────────────────
[SIMULATED] {consequence}

Evidence basis: [EVIDENCE: {chunk_id}] — {one sentence
explaining what the source actually says and why it
supports this consequence}

Analogy (if applicable): [EVIDENCE: {chunk_id}] —
{name the analogous case and explain the parallel}

Confidence: {High / Medium / Low}
Reason for confidence: {why — what makes this
well-supported or uncertain}

─────────────────────────────────────────
Step 2 — Short-term Ripple (10-30 years)
─────────────────────────────────────────
[SIMULATED] {consequence — must follow causally from Step 1}

Evidence basis: [EVIDENCE: {chunk_id}]
Analogy (if applicable): [EVIDENCE: {chunk_id}]
Confidence: {High / Medium / Low}
Reason for confidence: {why}

─────────────────────────────────────────
Step 3 — Medium-term Transformation (30-100 years)
─────────────────────────────────────────
[SIMULATED] {consequence — must follow causally from Step 2}

Evidence basis: [EVIDENCE: {chunk_id}]
Analogy (if applicable): [EVIDENCE: {chunk_id}]
Confidence: {High / Medium / Low}
Reason for confidence: {why}

─────────────────────────────────────────
Step 4 — Long-term Legacy (100+ years)
─────────────────────────────────────────
[SIMULATED] {consequence — must follow causally from Step 3}

Evidence basis: [EVIDENCE: {chunk_id}]
Confidence: Low — long-term consequences are inherently
speculative, BUT [EVIDENCE: chunk_id] is still required.
Even speculative reasoning must cite its closest source.
Do not omit [EVIDENCE] on Step 4.

WHAT REMAINS UNKNOWABLE:
{explicitly state 2-3 things that cannot be determined
even with this reasoning — be honest about the limits}

─────────────────────────────────────────
RECONNECTION POINT
─────────────────────────────────────────
Where does this alternate timeline likely converge with
or permanently diverge from actual history?
{1-2 sentences — grounded in a verified fact or analogy}

─────────────────────────────────────────
HISTORIAN'S NOTE
─────────────────────────────────────────
What do historians actually debate about the real
version of these events that is relevant to this
counterfactual?
{cite debated_points from grounded context if any exist,
otherwise note the closest relevant historiographical
debate}

════════════════════════════════════════════════════════
VERIFIED FACTS AVAILABLE TO YOU:
{verified_facts}

HISTORICAL ANALOGIES AVAILABLE TO YOU:
{analogies}

DEBATED POINTS (use in Historian's Note):
{debated_points}
════════════════════════════════════════════════════════

Remember: You are a reasoning engine that shows its work.
Every [SIMULATED] claim needs [EVIDENCE].
If the evidence is thin — say so. That honesty IS the
value of this system.
"""


def _format_facts(facts: list[GroundedFact]) -> str:
    """Render a list of grounded facts as a numbered list keyed by chunk_id.

    The ``[chunk_id: X]`` prefix is the exact token the model must echo back
    inside ``[EVIDENCE: X]`` — same contract as the grounding layer's chunk
    labelling. ``source_title`` is included as context so the LLM can name the
    analogy ("the Meiji Restoration", "the Ottoman Empire") in its prose.
    """
    if not facts:
        return "(none)"
    return "\n".join(
        f"{i}. [chunk_id: {fact.source_chunk_id}] ({fact.source_title}) {fact.claim}"
        for i, fact in enumerate(facts, start=1)
    )


def _render_prompt(analysis: QueryAnalysis, grounded: GroundedContext) -> str:
    """Inject the six runtime variables into the template (str.replace, not %).

    See the note above ``_PROMPT_TEMPLATE`` for why this is not ``.format``.
    """
    prompt = _PROMPT_TEMPLATE
    prompt = prompt.replace("{proposed_change}", analysis.proposed_change)
    prompt = prompt.replace("{time_period}", analysis.time_period)
    prompt = prompt.replace("{geography}", analysis.geography)
    prompt = prompt.replace("{verified_facts}", _format_facts(grounded.verified_facts))
    prompt = prompt.replace("{analogies}", _format_facts(grounded.analogies))
    prompt = prompt.replace("{debated_points}", _format_facts(grounded.debated_points))
    return prompt


# --- Parsing -----------------------------------------------------------------
# The model is asked to emit numbered ``Step N`` headers, ``[SIMULATED]`` /
# ``[EVIDENCE: ...]`` markers, ``Confidence:`` / ``Reason for confidence:``
# lines, and three named tail sections. We slice on the structural anchors and
# never raise on a missing section — drift gets logged to ``parse_warnings``.

# Step header on its own line: "Step 1 — Immediate Consequences (0-10 years)".
# Tolerant to any divider style around it (the template uses unicode box
# drawing, but LLMs sometimes render ASCII dashes).
_STEP_HEADER_RE = re.compile(
    r"^[ \t]*(?:#{1,3}\s*|\*{1,2})?Step[ \t]+([1-9]\d?)\b[^\n]*$",
    re.MULTILINE,
)
_EVIDENCE_RE = re.compile(r"\[EVIDENCE:\s*([^\]]+?)\]", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"^\s*Confidence:\s*(High|Medium|Low)\b", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_REASON_RE = re.compile(
    r"^\s*Reason for confidence:\s*(.+?)(?=\n\s*\n|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
# Tail section headers — match the start of the line, case-insensitive,
# tolerant to straight or curly apostrophe in HISTORIAN'S.
_TAIL_HEADERS = {
    "what_remains_unknowable": re.compile(
        r"^[ \t]*(?:#{1,3}\s*|\*{1,2})?WHAT\s+REMAINS\s+UNKNOWABLE\b[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "reconnection_point": re.compile(
        r"^[ \t]*(?:#{1,3}\s*|\*{1,2})?RECONNECTION\s+POINT\b[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "historians_note": re.compile(
        r"^[ \t]*(?:#{1,3}\s*|\*{1,2})?HISTORIAN[’']?S\s+NOTE\b[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _slice_steps(body: str) -> list[tuple[int, str, str]]:
    """Return ``(step_number, header_line, block_text)`` per ``Step N`` block.

    ``block_text`` ends at the next ``Step N`` header or the first tail section
    header (``WHAT REMAINS UNKNOWABLE`` / ``RECONNECTION POINT`` /
    ``HISTORIAN'S NOTE``), whichever comes first.
    """
    headers = list(_STEP_HEADER_RE.finditer(body))
    if not headers:
        return []
    # Earliest tail-section start, if any — clips the final step's block so we
    # don't slurp the unknowable / reconnection / historian's note prose into it.
    tail_starts = [
        m.start() for pat in _TAIL_HEADERS.values() for m in [pat.search(body)] if m
    ]
    body_end = min(tail_starts) if tail_starts else len(body)

    blocks: list[tuple[int, str, str]] = []
    for idx, match in enumerate(headers):
        try:
            step_n = int(match.group(1))
        except ValueError:
            continue
        next_start = headers[idx + 1].start() if idx + 1 < len(headers) else body_end
        # Block text is everything AFTER the header line, up to the next anchor.
        block = body[match.end():next_start].strip()
        blocks.append((step_n, match.group(0).strip(), block))
    return blocks


def _extract_consequence(block: str) -> str:
    """Pull the [SIMULATED] body up to the next field marker (or end of block).

    Falls back to the whole block if [SIMULATED] is missing — the caller logs a
    warning, and ``is_grounded`` still hinges on the presence of [EVIDENCE: ...].
    """
    sim_idx = block.lower().find("[simulated]")
    start = sim_idx + len("[simulated]") if sim_idx != -1 else 0
    tail = block[start:]
    # Stop at the first field marker. Each is matched on its own line so we
    # don't slice in the middle of a sentence that mentions "confidence".
    stop_re = re.compile(
        r"^\s*(Evidence basis|Analogy(?:\s*\(if applicable\))?|Confidence|Reason for confidence)\s*:",
        re.IGNORECASE | re.MULTILINE,
    )
    stop_match = stop_re.search(tail)
    end = stop_match.start() if stop_match else len(tail)
    return tail[:end].strip()


def _extract_analogy_ids(block: str) -> list[str]:
    """Chunk_ids cited specifically on lines whose label is ``Analogy``.

    The "Analogy" label can span more than one line, so we capture from the
    label up to the next field marker (Evidence basis / Confidence / Reason)
    or a blank line, then pull every [EVIDENCE: ...] within that slice.
    """
    label_re = re.compile(
        r"^\s*Analogy(?:\s*\(if applicable\))?\s*:\s*(.*?)(?=^\s*(?:Evidence basis|Confidence|Reason for confidence)\s*:|\n\s*\n|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    ids: list[str] = []
    for section in label_re.findall(block):
        for cid in _EVIDENCE_RE.findall(section):
            cleaned = cid.strip().rstrip("*").strip()
            if cleaned:
                ids.append(cleaned)
    return _dedup_preserve_order(ids)


def _extract_tail(body: str, key: str) -> str:
    """Slice a named tail section's body up to the next tail header or EOF."""
    pat = _TAIL_HEADERS[key]
    match = pat.search(body)
    if not match:
        return ""
    start = match.end()
    # Stop at whichever OTHER tail header appears first after this one.
    candidate_ends = [
        m.start() for k, p in _TAIL_HEADERS.items() if k != key
        for m in [p.search(body, start)] if m
    ]
    end = min(candidate_ends) if candidate_ends else len(body)
    return body[start:end].strip()


def _parse_response(
    text: str, allowed_ids: set[str], warnings: list[str]
) -> dict:
    """Parse the LLM's structured response. Pure function, mutates ``warnings``.

    Returns a dict of fields ready to feed into ``CounterfactualReasoning``
    (minus the contextual ``proposed_change`` / ``divergence_point`` / raw_response
    that the caller fills in). Never raises on shape drift.
    """
    blocks = _slice_steps(text)
    if not blocks:
        warnings.append("no reasoning steps parsed — see raw_response")

    steps: list[ReasoningStep] = []
    seen_numbers: set[int] = set()
    cap = settings.MAX_CAUSAL_STEPS
    dropped_extra = 0

    for step_n, header, block in blocks:
        if step_n > cap:
            dropped_extra += 1
            continue
        if step_n in seen_numbers:
            warnings.append(f"duplicate Step {step_n} block — kept the first")
            continue
        seen_numbers.add(step_n)

        consequence = _extract_consequence(block)
        if "[simulated]" not in block.lower():
            warnings.append(f"Step {step_n}: missing [SIMULATED] marker")

        all_ids = _dedup_preserve_order(
            [
                cid.strip().rstrip("*").strip()
                for cid in _EVIDENCE_RE.findall(block)
                if cid.strip().rstrip("*").strip()
            ]
        )
        analogy_ids = _extract_analogy_ids(block)
        # time_horizon: strip leading "Step N —" / "Step N -" / "Step N" so we
        # keep the descriptive label only.
        horizon = re.sub(
            r"^\s*(?:#{1,3}\s*|\*{1,2})?Step\s+\d+\s*[—\-:]?\s*", "", header
        ).strip().rstrip("*").strip()

        conf_match = _CONFIDENCE_RE.search(block)
        confidence: Confidence = (
            conf_match.group(1).title() if conf_match else "Unknown"  # type: ignore[assignment]
        )
        reason_match = _CONFIDENCE_REASON_RE.search(block)
        confidence_reason = reason_match.group(1).strip() if reason_match else ""

        unknown_ids = [cid for cid in all_ids if cid not in allowed_ids]

        steps.append(
            ReasoningStep(
                step_number=step_n,
                time_horizon=horizon,
                consequence=consequence,
                evidence_chunk_ids=all_ids,
                analogy_chunk_ids=analogy_ids,
                confidence=confidence,
                confidence_reason=confidence_reason,
                is_grounded=bool(all_ids),
                unknown_evidence_ids=unknown_ids,
            )
        )

    if dropped_extra:
        warnings.append(
            f"dropped {dropped_extra} step(s) beyond MAX_CAUSAL_STEPS={cap} (Critical Rule #3)"
        )

    steps.sort(key=lambda s: s.step_number)

    tail = {key: _extract_tail(text, key) for key in _TAIL_HEADERS}
    for key, value in tail.items():
        if not value:
            warnings.append(f"missing tail section: {key}")

    return {
        "steps": steps,
        "what_remains_unknowable": tail["what_remains_unknowable"],
        "reconnection_point": tail["reconnection_point"],
        "historians_note": tail["historians_note"],
    }


# --- Public API --------------------------------------------------------------


def reason_about_counterfactual(
    analysis: QueryAnalysis, grounded: GroundedContext
) -> CounterfactualReasoning:
    """Run one counterfactual through the reasoning chain — exactly one LLM call.

    The prompt template (verbatim from the project spec) is rendered with the
    six runtime variables, then sent as the system message of a single
    completion at ``TEMPERATURE`` (0.3). The response is parsed for the four
    ``Step N`` blocks and the three tail sections; missing or extra sections
    are recorded on ``parse_warnings`` rather than raising.

    Args:
        analysis: Agent 1's ``QueryAnalysis`` — supplies ``proposed_change``,
            ``time_period``, ``geography``.
        grounded: Agent 3's ``GroundedContext`` — supplies ``verified_facts``,
            ``analogies``, ``debated_points``.

    Returns:
        A validated ``CounterfactualReasoning`` with up to
        ``settings.MAX_CAUSAL_STEPS`` reasoning steps, the three tail sections,
        the full ``raw_response``, and any ``parse_warnings``.

    Raises:
        RuntimeError: if ``CEREBRAS_API_KEY`` is not configured (from
            ``get_llm_client``) — only reachable when there is context to
            reason on (the empty-context path makes no LLM call).
    """
    divergence_point = f"{analysis.time_period} — {analysis.geography}"

    # Rule #6: nothing to ground reasoning on ⇒ no LLM call.
    if not grounded.verified_facts and not grounded.analogies:
        logger.warning(
            "reason_about_counterfactual: empty verified_facts AND analogies "
            "for %r — no LLM call (Rule #6); reasoning skipped",
            analysis.proposed_change,
        )
        return CounterfactualReasoning(
            proposed_change=analysis.proposed_change,
            divergence_point=divergence_point,
            parse_warnings=["no grounded context — reasoning skipped (Rule #6)"],
        )

    prompt = _render_prompt(analysis, grounded)
    client = get_llm_client()
    response = client.chat.completions.create(
        model=settings.CEREBRAS_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Reason about the proposed change using the format above, "
                    "exactly. Emit Steps 1 through 4, then WHAT REMAINS UNKNOWABLE, "
                    "RECONNECTION POINT, and HISTORIAN'S NOTE."
                ),
            },
        ],
    )
    raw_response = (response.choices[0].message.content or "").strip()

    # Allowed chunk_ids = everything we put in front of the model. Citing
    # anything outside this set is logged on the step's unknown_evidence_ids.
    allowed_ids: set[str] = set()
    allowed_ids.update(f.source_chunk_id for f in grounded.verified_facts)
    allowed_ids.update(f.source_chunk_id for f in grounded.analogies)
    allowed_ids.update(f.source_chunk_id for f in grounded.debated_points)
    allowed_ids.update(f.source_chunk_id for f in grounded.background_context)

    warnings: list[str] = []
    parsed = _parse_response(raw_response, allowed_ids, warnings)

    reasoning = CounterfactualReasoning(
        proposed_change=analysis.proposed_change,
        divergence_point=divergence_point,
        steps=parsed["steps"],
        what_remains_unknowable=parsed["what_remains_unknowable"],
        reconnection_point=parsed["reconnection_point"],
        historians_note=parsed["historians_note"],
        raw_response=raw_response,
        parse_warnings=warnings,
    )

    ungrounded = sum(1 for s in reasoning.steps if not s.is_grounded)
    unknown_cites = sum(len(s.unknown_evidence_ids) for s in reasoning.steps)
    logger.info(
        "reason_about_counterfactual: %d step(s) parsed (%d ungrounded), "
        "%d unknown citation(s), %d parse warning(s)",
        len(reasoning.steps),
        ungrounded,
        unknown_cites,
        len(warnings),
    )
    return reasoning


if __name__ == "__main__":
    # Live smoke test on the Mughal counterfactual — chains A2 + A3 + A4 to get
    # real grounded context. Total LLM calls: 2 (grounding) + 1 (reasoning) = 3,
    # well within the free-tier budget. Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m agents.reasoning_agent
    import sys

    # Windows console defaults to cp1252; reasoning text can contain non-cp1252
    # chars (em-dashes, macrons) — see CLAUDE.md "WINDOWS CONSOLE ENCODING".
    sys.stdout.reconfigure(encoding="utf-8")

    from agents.grounding_layer import ground_context
    from agents.retrieval_engine import retrieve_context

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    # Built by hand (no Agent 1 LLM call) so this test isolates reasoning. Mirrors
    # the grounding_layer.py smoke test's QueryAnalysis so the corpus matches.
    analysis = QueryAnalysis(
        time_period="1526-1857",
        geography="South Asia",
        key_actors=["Mughal Empire", "British East India Company"],
        counterfactual_type="political",
        proposed_change="The Mughal Empire never declined and was never replaced by British rule",
        search_queries=[
            "Mughal Empire decline and fall causes",
            "British colonization of India East India Company",
            "Mughal Empire administration and economy",
        ],
        analogy_queries=[
            "Ottoman Empire decline and longevity",
            "Meiji Restoration modernization",
        ],
    )
    user_question = "What if the Mughal Empire had never declined and the British never ruled India?"

    print(f"\nQuestion: {user_question}\n")
    context = retrieve_context(analysis)
    print(
        f"Retrieved: primary={len(context.primary_chunks)}  "
        f"analogy={len(context.analogy_chunks)}"
    )

    if not context.primary_chunks and not context.analogy_chunks:
        print("No verified context retrieved — is ChromaDB populated? (run ingestion first)")
        sys.exit(0)

    grounded = ground_context(context, user_question)
    print(
        f"Grounded: verified={len(grounded.verified_facts)}  "
        f"debated={len(grounded.debated_points)}  "
        f"background={len(grounded.background_context)}  "
        f"analogies={len(grounded.analogies)}\n"
    )

    reasoning = reason_about_counterfactual(analysis, grounded)

    print(f"PROPOSED CHANGE: {reasoning.proposed_change}")
    print(f"DIVERGENCE POINT: {reasoning.divergence_point}\n")

    if not reasoning.steps:
        print("(no reasoning steps parsed)")
    for step in reasoning.steps:
        grounded_tag = "GROUNDED" if step.is_grounded else "UNGROUNDED"
        print(f"── Step {step.step_number} — {step.time_horizon}  [{grounded_tag}]")
        print(f"   {step.consequence}")
        if step.evidence_chunk_ids:
            print(f"   evidence: {', '.join(step.evidence_chunk_ids)}")
        if step.analogy_chunk_ids:
            print(f"   analogy:  {', '.join(step.analogy_chunk_ids)}")
        if step.unknown_evidence_ids:
            print(f"   UNKNOWN cited ids: {', '.join(step.unknown_evidence_ids)}")
        print(f"   confidence: {step.confidence} — {step.confidence_reason}")
        print()

    print(f"WHAT REMAINS UNKNOWABLE:\n{reasoning.what_remains_unknowable or '(missing)'}\n")
    print(f"RECONNECTION POINT:\n{reasoning.reconnection_point or '(missing)'}\n")
    print(f"HISTORIAN'S NOTE:\n{reasoning.historians_note or '(missing)'}\n")

    if reasoning.parse_warnings:
        print("PARSE WARNINGS:")
        for warning in reasoning.parse_warnings:
            print(f"  - {warning}")
