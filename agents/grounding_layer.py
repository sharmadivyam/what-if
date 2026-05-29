"""Agent 3 — Grounding Layer.

Third node in the LangGraph pipeline. Sits between Retrieval (Agent 2) and
Reasoning (Agent 4). Turns the raw retrieved chunks into CLASSIFIED, individually
cited grounded facts — the verified/debated/background foundation that the
reasoning agent is allowed to build on.

Responsibilities:
- Take the ``RetrievalContext`` from Agent 2 plus the user's original question.
- Extract the factual claims explicitly PRESENT in the retrieved chunks, classify
  each as VERIFIED / DEBATED / BACKGROUND, and attach its source citation
  (title + chunk_id + url).
- Return a Pydantic ``GroundedContext`` (Critical Rule #4) — never raw strings.
- Produce ZERO simulation/speculation (Critical Rule #1): this layer only grounds
  existing source text; it never narrates what *would* have happened.

Implementation notes:
- LLM access goes through ``core.llm_client.call_with_fallback()`` with
  ``model=settings.CEREBRAS_MODEL`` (Critical Rule #7). No provider client is
  instantiated here; the wrapper handles automatic OpenRouter fallback on
  Cerebras 429/quota errors. Temperature is pinned at ``settings.LLM_TEMPERATURE``
  (0.0) — the deterministic temperature for the grounded/generative agents.
- RULE #5 + #6: on an empty ``RetrievalContext`` (no primary AND no analogy
  chunks) we short-circuit to an empty-but-valid ``GroundedContext`` and make NO
  LLM call — Rule #6 forbids calling the LLM without retrieved context, and there
  is nothing to ground. Likewise an empty pool is skipped (never called with zero
  chunks).
- BATCHED PER POOL: one LLM call grounds ALL primary chunks, a second grounds ALL
  analogy chunks. This replaced an earlier per-chunk design (one call per chunk)
  because the free Cerebras tier hard-throttles bursts of ~11 calls/run with
  persistent 429 "queue_exceeded" and ~60s forced waits — per-chunk made a single
  pipeline run take 10+ minutes and still fail. Two calls/run is viable.
- HOW THE CRITICAL RULE IS ENFORCED ("never label VERIFIED unless it directly
  appears in a retrieved chunk"), under batching:
    * The model tags each claim with the ``source_chunk_id`` it came from, but we
      VALIDATE that id against the actual retrieved set for that pool and DROP any
      claim citing an unknown id (logged). A fact can therefore never cite a chunk
      that was not retrieved (Rule #2).
    * ``source_title`` (and the ``source_map`` URL) are attached BY US from the
      matched ``SearchResult`` — never parsed from the model — so citations cannot
      be fabricated.
    * The prompt requires claims to come ONLY from the provided chunk texts;
      temperature is 0.0; any unrecognised classification label is coerced to
      BACKGROUND (the safe, non-VERIFIED default).
  Residual risk vs the old per-chunk design: INTRA-batch misattribution — the
  model could tag a real claim with a different but still-valid chunk_id. That is
  the accepted trade-off for free-tier viability; only unknown ids are detectable.
- Structured output uses JSON mode (``response_format={"type": "json_object"}``)
  validated with Pydantic, with a single corrective retry that feeds the
  validation error back (same pattern as Agent 1). A pool whose response still
  fails to parse is skipped with a warning rather than sinking the whole run.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, get_args

from pydantic import BaseModel, Field

from agents.retrieval_engine import RetrievalContext
from config import settings
from core.llm_client import call_with_fallback
from vectorstore.chroma_client import SearchResult

logger = logging.getLogger(__name__)

# The three grounding classifications. Defined once so the Literal type, the
# coercion guard, and the prompt text all stay in sync.
Classification = Literal["VERIFIED", "DEBATED", "BACKGROUND"]
_CLASSIFICATIONS: tuple[str, ...] = get_args(Classification)


class GroundedFact(BaseModel):
    """One factual claim extracted from a single retrieved chunk, with citation.

    The claim is faithful to the source chunk text (never invented). Which list of
    ``GroundedContext`` a fact lands in encodes its classification; ``source_chunk_id``
    + ``source_title`` make it independently citable (Critical Rule #2).
    """

    claim: str
    source_chunk_id: str
    source_title: str
    confidence_basis: str  # short reason for the classification, grounded in the chunk


class GroundedContext(BaseModel):
    """The grounded, classified output of Agent 3.

    Every fact is sorted into exactly one bucket. ``verified_facts`` are claims the
    source states as established history; ``debated_points`` are claims the source
    itself frames as contested; ``background_context`` is scene-setting detail;
    ``analogies`` are claims drawn from analogous-situation chunks. All four lists
    may be empty (empty collection / no matches / nothing extractable) — Rule #5.
    ``source_map`` maps every retrieved chunk_id to its source URL for citation.
    """

    verified_facts: list[GroundedFact] = Field(default_factory=list)
    debated_points: list[GroundedFact] = Field(default_factory=list)
    background_context: list[GroundedFact] = Field(default_factory=list)
    analogies: list[GroundedFact] = Field(default_factory=list)
    source_map: dict[str, str] = Field(default_factory=dict)  # chunk_id -> url


# --- Internal LLM-response models (validated, then mapped to GroundedFact) -------
# Kept lenient: every field except ``claim`` defaults to empty so a missing field
# does not nuke an otherwise-usable claim. ``claim`` is required — an item without
# it is useless. ``classification`` is a plain str (not the Literal) so an odd
# label triggers conservative coercion rather than a hard validation fail. A blank
# / unknown ``source_chunk_id`` is caught later by validation against the pool.


class _PrimaryClaim(BaseModel):
    source_chunk_id: str = ""
    claim: str
    classification: str = ""
    confidence_basis: str = ""


class _PrimaryExtraction(BaseModel):
    claims: list[_PrimaryClaim] = Field(default_factory=list)


class _AnalogyClaim(BaseModel):
    source_chunk_id: str = ""
    claim: str
    confidence_basis: str = ""


class _AnalogyExtraction(BaseModel):
    claims: list[_AnalogyClaim] = Field(default_factory=list)


_PRIMARY_SYSTEM_PROMPT = f"""\
You are the Grounding Layer of a counterfactual history engine. You are given \
SEVERAL chunks of verified historical source text (each labelled with a chunk_id) \
and the user's "what if" question (for relevance only). Your job is to EXTRACT the \
factual claims that are explicitly PRESENT IN THESE CHUNKS, CLASSIFY each one, and \
TAG each claim with the chunk_id of the chunk it came from.

Absolute rules:
- Extract claims ONLY from the provided chunk texts. NEVER add facts from your own \
knowledge and NEVER infer beyond what the texts say. If a statement is not \
supported by a chunk, do not include it.
- Each claim's "source_chunk_id" MUST be copied EXACTLY from the chunk the claim \
was taken from. Never invent, merge, or alter a chunk_id.
- Do NOT speculate, simulate, or answer the what-if question. You only ground \
text that is already present.

Classify each claim as EXACTLY one of:
- "VERIFIED": the chunk states it as established historical fact.
- "DEBATED": the chunk itself frames it as contested, disputed, or uncertain \
(e.g. "historians disagree", "some argue", "it is debated", "may have", "possibly").
- "BACKGROUND": contextual or descriptive detail that sets the scene rather than \
asserting a sharp factual or causal claim.

Respond with a single JSON object and nothing else:
{{"claims": [{{"source_chunk_id": "<exact chunk_id>", \
"claim": "<concise statement faithful to that chunk>", \
"classification": "VERIFIED" | "DEBATED" | "BACKGROUND", \
"confidence_basis": "<short reason, grounded in the chunk wording>"}}]}}

- If the chunks have no usable factual claims, return {{"claims": []}}.
- Output valid JSON only — no prose, no markdown, no code fences.
- Prefer a handful of clear, self-contained claims per chunk over many fragments."""


_ANALOGY_SYSTEM_PROMPT = """\
You are the Grounding Layer of a counterfactual history engine. You are given \
SEVERAL chunks of source text describing ANALOGOUS historical situations \
(comparable cases elsewhere or in other eras), each labelled with a chunk_id, plus \
the user's "what if" question (for relevance only). Extract the factual claims \
explicitly PRESENT IN THESE CHUNKS that could serve as useful comparisons, and TAG \
each claim with the chunk_id it came from.

Absolute rules:
- Extract claims ONLY from the provided chunk texts. NEVER add facts from your own \
knowledge. Do NOT speculate, simulate, or answer the what-if question.
- Each claim's "source_chunk_id" MUST be copied EXACTLY from the chunk the claim \
was taken from. Never invent, merge, or alter a chunk_id.

Respond with a single JSON object and nothing else:
{"claims": [{"source_chunk_id": "<exact chunk_id>", \
"claim": "<concise statement faithful to that chunk>", \
"confidence_basis": "<short note on why it is a comparable case, grounded in the chunk>"}]}

- If the chunks have no usable claims, return {"claims": []}.
- Output valid JSON only — no prose, no markdown, no code fences."""


def _format_chunks(chunks: list[SearchResult]) -> str:
    """Render a pool of chunks for the prompt, each labelled with its chunk_id.

    The chunk_id label is what the model must copy back into each claim's
    ``source_chunk_id`` so we can re-attach the (trusted) citation metadata.
    """
    return "\n\n".join(
        f'[chunk_id: {chunk.chunk_id}] (source: "{chunk.source}")\n'
        f'"""\n{chunk.text.strip()}\n"""'
        for chunk in chunks
    )


def _build_messages(
    system_prompt: str,
    chunks: list[SearchResult],
    user_question: str,
    *,
    error_feedback: str | None = None,
) -> list[dict]:
    """Assemble the chat messages for one pool's extraction call.

    The user question is supplied for relevance only; the chunk texts are the sole
    source of claims. When ``error_feedback`` is given (the corrective retry), the
    prior validation error is appended so the model can fix its output.
    """
    user_content = (
        f'User\'s what-if question (for relevance only): "{user_question.strip()}"\n\n'
        f"Source chunks ({len(chunks)}). Extract claims ONLY from these texts and tag "
        f"each claim with the EXACT chunk_id it came from:\n\n"
        f"{_format_chunks(chunks)}"
    )
    if error_feedback:
        user_content += (
            "\n\nYour previous response was rejected with this error:\n"
            f"{error_feedback}\n"
            "Return a corrected JSON object that satisfies all the requirements."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _normalize_classification(value: str) -> str:
    """Return a valid classification label, coercing anything unknown to BACKGROUND.

    This backs the CRITICAL rule: the only way a claim earns the VERIFIED label is
    for the model to explicitly return it; any garbled/unexpected value falls back
    to the safe, non-VERIFIED default rather than silently being trusted.
    """
    label = (value or "").strip().upper()
    if label in _CLASSIFICATIONS:
        return label
    logger.warning("grounding: unknown classification %r coerced to BACKGROUND", value)
    return "BACKGROUND"


def _extract_pool(
    chunks: list[SearchResult], user_question: str, *, is_analogy: bool
) -> list[_PrimaryClaim] | list[_AnalogyClaim]:
    """Run one whole pool of chunks through the LLM in a single call.

    JSON mode at temperature 0.0, with a single corrective retry. Returns ``[]``
    for an empty pool WITHOUT calling the LLM (Rule #6). If the response still
    fails to validate after two attempts, the pool is skipped (returns ``[]``)
    with a warning — a malformed response must not sink the whole run.
    """
    if not chunks:
        return []

    label = "analogy" if is_analogy else "primary"
    system_prompt = _ANALOGY_SYSTEM_PROMPT if is_analogy else _PRIMARY_SYSTEM_PROMPT
    model_cls = _AnalogyExtraction if is_analogy else _PrimaryExtraction

    error_feedback: str | None = None
    last_error: Exception | None = None

    for attempt in range(2):
        response = call_with_fallback(
            messages=_build_messages(
                system_prompt, chunks, user_question, error_feedback=error_feedback
            ),
            model=settings.CEREBRAS_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        try:
            extraction = model_cls.model_validate_json(content)
            return extraction.claims
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            error_feedback = str(exc)
            logger.warning(
                "grounding: invalid %s-pool response on attempt %d/2: %s",
                label,
                attempt + 1,
                exc,
            )

    logger.warning(
        "grounding: skipping %s pool — no valid extraction after 2 attempts: %s",
        label,
        last_error,
    )
    return []


def ground_context(context: RetrievalContext, user_question: str) -> GroundedContext:
    """Ground retrieved chunks into classified, individually cited facts.

    Makes at most two LLM calls — one for the primary pool, one for the analogy
    pool. Primary claims are routed into ``verified_facts`` / ``debated_points`` /
    ``background_context`` by the model's classification (unknown labels coerced to
    BACKGROUND); analogy claims go into ``analogies``. Each claim's
    ``source_chunk_id`` is VALIDATED against the retrieved chunks of its pool and
    dropped if unknown; the citation's ``source_title`` is taken from the matched
    chunk, never from the model, so citations cannot be hallucinated (Rule #2).

    On an empty ``RetrievalContext`` no LLM call is made (Rules #5 + #6): the
    returned context is empty-but-valid and a warning is logged.

    Args:
        context: The ``RetrievalContext`` from Agent 2 (Retrieval Engine).
        user_question: The user's original "what if" question (for relevance only;
            it is never a source of claims).

    Returns:
        A ``GroundedContext`` with the four classified fact buckets and a
        ``source_map`` of every retrieved chunk_id to its source URL.

    Raises:
        ValueError: if ``user_question`` is empty/blank.
        RuntimeError: if ``CEREBRAS_API_KEY`` is not configured (from
            ``call_with_fallback``'s primary path), or if the fallback path is
            reached and ``OPENROUTER_API_KEY`` is also missing — only reachable
            when there is context to ground.
    """
    if not user_question or not user_question.strip():
        raise ValueError("user_question must be a non-empty string")

    all_chunks = [*context.primary_chunks, *context.analogy_chunks]
    # Built locally from the retrieved chunks (both pools) — full provenance of
    # what was available to ground, independent of any LLM output.
    source_map = {chunk.chunk_id: chunk.source_url for chunk in all_chunks}
    grounded = GroundedContext(source_map=source_map)

    if not all_chunks:
        logger.warning(
            "ground_context: empty RetrievalContext for %r — no LLM call (Rules #5/#6); "
            "nothing to ground",
            context.query_used or user_question,
        )
        return grounded

    # chunk_id -> SearchResult, per pool: the source of truth for re-attaching the
    # (trusted) citation and for validating the chunk_ids the model echoes back.
    primary_by_id = {chunk.chunk_id: chunk for chunk in context.primary_chunks}
    analogy_by_id = {chunk.chunk_id: chunk for chunk in context.analogy_chunks}
    dropped = 0  # claims discarded for citing a chunk_id outside their pool

    # --- Primary pool: classify into VERIFIED / DEBATED / BACKGROUND -------------
    for item in _extract_pool(context.primary_chunks, user_question, is_analogy=False):
        claim_text = item.claim.strip()
        if not claim_text:
            continue
        chunk = primary_by_id.get(item.source_chunk_id.strip())
        if chunk is None:
            dropped += 1
            logger.warning(
                "grounding: dropping primary claim citing unknown chunk_id %r: %.60s",
                item.source_chunk_id,
                claim_text,
            )
            continue
        fact = GroundedFact(
            claim=claim_text,
            source_chunk_id=chunk.chunk_id,
            source_title=chunk.source,
            confidence_basis=item.confidence_basis.strip(),
        )
        label = _normalize_classification(item.classification)
        if label == "VERIFIED":
            grounded.verified_facts.append(fact)
        elif label == "DEBATED":
            grounded.debated_points.append(fact)
        else:  # BACKGROUND (including coerced unknowns)
            grounded.background_context.append(fact)

    # --- Analogy pool: comparison claims, no V/D/B label -------------------------
    for item in _extract_pool(context.analogy_chunks, user_question, is_analogy=True):
        claim_text = item.claim.strip()
        if not claim_text:
            continue
        chunk = analogy_by_id.get(item.source_chunk_id.strip())
        if chunk is None:
            dropped += 1
            logger.warning(
                "grounding: dropping analogy claim citing unknown chunk_id %r: %.60s",
                item.source_chunk_id,
                claim_text,
            )
            continue
        grounded.analogies.append(
            GroundedFact(
                claim=claim_text,
                source_chunk_id=chunk.chunk_id,
                source_title=chunk.source,
                confidence_basis=item.confidence_basis.strip(),
            )
        )

    logger.info(
        "ground_context: %d verified, %d debated, %d background, %d analog(y/ies) "
        "from %d primary + %d analogy chunk(s)%s",
        len(grounded.verified_facts),
        len(grounded.debated_points),
        len(grounded.background_context),
        len(grounded.analogies),
        len(context.primary_chunks),
        len(context.analogy_chunks),
        f" ({dropped} claim(s) dropped for unknown chunk_id)" if dropped else "",
    )
    return grounded


if __name__ == "__main__":
    # Live smoke test — calls the real Cerebras API (needs CEREBRAS_API_KEY) and
    # the populated ChromaDB. Makes only two LLM calls (one per pool). Run from the
    # project root:
    #   D:\historyos\venv\Scripts\python.exe -m agents.grounding_layer
    import sys

    # Windows console defaults to cp1252; grounded claims can carry non-cp1252
    # chars (macrons, en-dashes), so reconfigure stdout before printing (Known Issue).
    sys.stdout.reconfigure(encoding="utf-8")

    from agents.query_understanding import QueryAnalysis
    from agents.retrieval_engine import retrieve_context

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    # Built by hand (no Agent 1 LLM call) so this test isolates grounding. Aimed at
    # the ingested corpus (Mughal / British Raj + analogies), mirroring Agent 2's test.
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
        ],
    )
    user_question = "What if the Mughal Empire had never declined and the British never ruled India?"

    print(f"\nQuestion: {user_question}\n")
    context = retrieve_context(analysis)
    print(
        f"Retrieved: primary={len(context.primary_chunks)}  "
        f"analogy={len(context.analogy_chunks)}\n"
    )

    if not context.primary_chunks and not context.analogy_chunks:
        print("No verified context retrieved — is ChromaDB populated? (run ingestion first)")
        sys.exit(0)

    grounded = ground_context(context, user_question)

    def _show(title: str, facts: list[GroundedFact]) -> None:
        print(f"\n{title} ({len(facts)}):")
        if not facts:
            print("  (none)")
        for fact in facts:
            print(
                f"  - {fact.claim}\n"
                f"      cite: {fact.source_chunk_id} ({fact.source_title})\n"
                f"      basis: {fact.confidence_basis}"
            )

    _show("VERIFIED FACTS", grounded.verified_facts)
    _show("DEBATED POINTS", grounded.debated_points)
    _show("BACKGROUND CONTEXT", grounded.background_context)
    _show("ANALOGIES", grounded.analogies)

    print(f"\nSOURCE MAP ({len(grounded.source_map)} chunk(s)):")
    for chunk_id, url in grounded.source_map.items():
        print(f"  {chunk_id} -> {url or '(no url)'}")
