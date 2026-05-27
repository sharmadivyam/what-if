"""Agent 1 — Query Understanding.

First node in the LangGraph pipeline. Turns a free-form user
"what if" question into a structured, validated counterfactual.

Responsibilities:
- Parse the user's natural-language question.
- Extract: the historical fact being altered (the "point of
  divergence"), the entities involved, the time period / location,
  and the counterfactual premise being proposed.
- Generate clean retrieval queries for the next agent to search.
- Return a Pydantic model (Critical Rule #4), never a raw string.

This agent does NOT call the LLM to invent history — it only
structures the user's request so retrieval can ground it.

Implementation notes:
- LLM access goes through ``core.llm_client.get_llm_client()`` with
  ``model=settings.CEREBRAS_MODEL`` (Critical Rule #7). No provider client is
  instantiated here.
- RULE #6 EXEMPTION: Critical Rule #6 ("never call the LLM without retrieved
  context") guards against generating ungrounded *historical facts*. This agent
  necessarily runs BEFORE retrieval, so it has no context yet — and it produces
  no facts: it only decomposes the question into search parameters. The prompt is
  constrained to parse/classify the wording, never to narrate what happened. So
  this LLM call is a legitimate exception, not a violation.
- Structured output uses JSON mode (``response_format={"type": "json_object"}``),
  which is broadly compatible across OpenAI-compatible providers, then validates
  with Pydantic. A single corrective retry re-prompts with the validation error
  before giving up, so a slightly malformed first response self-heals.
- Temperature is pinned at ``TEMPERATURE`` (0.1) for consistent analysis —
  deliberately distinct from ``settings.LLM_TEMPERATURE`` (0.0), which governs
  the grounded/generative agents later in the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

from config import settings
from core.llm_client import get_llm_client

logger = logging.getLogger(__name__)

# Pinned low (but non-zero) for stable, repeatable query analysis. Kept separate
# from settings.LLM_TEMPERATURE on purpose (see module docstring).
TEMPERATURE = 0.1

# The six counterfactual categories. Defined once so the Literal type, the
# Pydantic enforcement, and the prompt text all stay in sync.
CounterfactualType = Literal[
    "political",
    "economic",
    "military",
    "social",
    "cultural",
    "technological",
]
_COUNTERFACTUAL_TYPES: tuple[str, ...] = get_args(CounterfactualType)


class QueryAnalysis(BaseModel):
    """Structured decomposition of a user's "what if" question.

    This is search *instructions* for the downstream pipeline, not historical
    claims. ``search_queries`` / ``analogy_queries`` are fed verbatim to the
    retrieval engine (ChromaDB) — the former to find the directly relevant
    context, the latter to surface analogous situations elsewhere/elsewhen.
    """

    time_period: str  # e.g. "1600s-1800s"
    geography: str  # e.g. "South Asia"
    key_actors: list[str] = Field(..., min_length=1)  # e.g. ["Mughal Empire", "British"]
    counterfactual_type: CounterfactualType
    proposed_change: str  # the hypothetical the user wants to introduce
    # The prompt asks for 3-4 / 2; bounds are lenient so a slightly-off count
    # from the model doesn't crash the pipeline.
    search_queries: list[str] = Field(..., min_length=1, max_length=6)
    analogy_queries: list[str] = Field(..., min_length=1, max_length=4)

    @field_validator("key_actors", "search_queries", "analogy_queries")
    @classmethod
    def _strip_and_drop_empty(cls, value: list[str]) -> list[str]:
        """Trim whitespace and drop blank entries from list fields."""
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty string")
        return cleaned


_SYSTEM_PROMPT = f"""\
You are the Query Understanding agent of a counterfactual history engine. Your \
job is to PARSE and CLASSIFY a user's "what if" history question into structured \
search parameters. You do NOT answer the question, narrate history, or assert any \
historical facts — you only decompose the wording so a later retrieval step can \
find verified context.

Given the user's question, respond with a single JSON object and nothing else, \
with exactly these keys:

- "time_period": string. The historical era implied, as a compact range, e.g. \
"1600s-1800s" or "1939-1945". Infer it from the entities/events mentioned.
- "geography": string. The region/place implied, e.g. "South Asia", "Western \
Europe", "Mediterranean".
- "key_actors": array of strings. The main entities involved (empires, states, \
leaders, institutions), e.g. ["Mughal Empire", "British East India Company"].
- "counterfactual_type": string. EXACTLY one of: {", ".join(_COUNTERFACTUAL_TYPES)}. \
Pick the domain the proposed change primarily belongs to.
- "proposed_change": string. A concise statement of the hypothetical alteration \
the user wants to introduce (the "point of divergence").
- "search_queries": array of 3-4 strings. Concise keyword/topic queries to \
retrieve the directly relevant verified historical context for this scenario.
- "analogy_queries": array of 2 strings. Queries to find ANALOGOUS situations \
elsewhere or in other eras (similar dynamics, not the same event), useful for \
reasoning by comparison.

Rules:
- Output valid JSON only — no prose, no markdown, no code fences.
- Base every field on what the question implies; do not invent specifics that \
aren't reasonably inferable.
- search_queries and analogy_queries should be search strings, not questions."""


def _build_messages(user_question: str, *, error_feedback: str | None = None) -> list[dict]:
    """Assemble the chat messages for one analysis call.

    When ``error_feedback`` is given (the corrective retry path), the prior
    validation error is appended so the model can fix its output.
    """
    user_content = f'User question: "{user_question.strip()}"'
    if error_feedback:
        user_content += (
            "\n\nYour previous response was rejected with this error:\n"
            f"{error_feedback}\n"
            "Return a corrected JSON object that satisfies all the requirements."
        )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def analyze_query(user_question: str) -> QueryAnalysis:
    """Decompose a user's "what if" question into a validated ``QueryAnalysis``.

    Calls the Cerebras LLM (via ``get_llm_client``) in JSON mode at low
    temperature, validates the result with Pydantic, and retries once with the
    validation error fed back if the first response is malformed.

    Args:
        user_question: The raw natural-language counterfactual question.

    Returns:
        A validated ``QueryAnalysis``.

    Raises:
        ValueError: if ``user_question`` is empty/blank, or if the model fails to
            produce a valid analysis after one corrective retry.
        RuntimeError: if ``CEREBRAS_API_KEY`` is not configured (from
            ``get_llm_client``).
    """
    if not user_question or not user_question.strip():
        raise ValueError("user_question must be a non-empty string")

    client = get_llm_client()
    error_feedback: str | None = None
    last_error: Exception | None = None

    # First attempt + one corrective retry.
    for attempt in range(2):
        response = client.chat.completions.create(
            model=settings.CEREBRAS_MODEL,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            messages=_build_messages(user_question, error_feedback=error_feedback),
        )
        content = (response.choices[0].message.content or "").strip()

        try:
            analysis = QueryAnalysis.model_validate_json(content)
            logger.info(
                "analyze_query: classified as %s | %d search + %d analogy queries",
                analysis.counterfactual_type,
                len(analysis.search_queries),
                len(analysis.analogy_queries),
            )
            return analysis
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            error_feedback = str(exc)
            logger.warning(
                "analyze_query: invalid response on attempt %d/2: %s", attempt + 1, exc
            )

    raise ValueError(
        f"Query analysis failed to produce valid output after 2 attempts: {last_error}"
    )


if __name__ == "__main__":
    # Live smoke test — calls the real Cerebras API (needs CEREBRAS_API_KEY).
    # Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m agents.query_understanding
    import sys

    # Windows console defaults to cp1252; analyses can contain non-cp1252 chars
    # (macrons, en-dashes), so reconfigure stdout before printing (Known Issue).
    sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    sample = "What if the Mughal Empire had industrialized before the British arrived?"
    print(f"\nQuestion: {sample}\n")
    result = analyze_query(sample)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
