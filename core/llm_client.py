"""Provider clients — the single source of LLM + embedding access.

HistoryOS uses a zero-cost stack:
- LLM (default): Cerebras through its OpenAI-compatible API, reached with the
  standard ``openai`` client pointed at the Cerebras base URL.
- LLM (fallback): OpenRouter, also OpenAI-compatible — used when Cerebras hits
  its daily token quota (a hard cap that does NOT self-heal via ``max_retries``).
- Embeddings: the local ``sentence-transformers`` model ``all-mpnet-base-v2``,
  wrapped in ChromaDB's ``SentenceTransformerEmbeddingFunction`` (no API key).

IMPORTANT — every module (especially the agents) MUST obtain its clients here via
``get_llm_client()`` / ``get_embedding_function()``. Do NOT instantiate ``OpenAI``
or an embedding function directly in agent / pipeline files: centralising it keeps
the provider swappable and the model/credentials configured in exactly one place
(``config.py``). Clients are cached per-provider; the embedding function is a
module-level singleton because loading it is expensive.

Each provider has its OWN model setting in ``config.py`` — when switching
providers in a caller, also pass the matching model:
    Cerebras   -> ``settings.CEREBRAS_MODEL``
    OpenRouter -> ``settings.OPENROUTER_MODEL``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config import settings

if TYPE_CHECKING:  # avoid importing heavy deps at module import time for typing
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

# Per-provider client cache. Keyed by provider name so a single process can
# hold both clients simultaneously (used when falling back mid-run).
_llm_clients: dict[str, "OpenAI"] = {}
_embedding_function = None  # chromadb SentenceTransformerEmbeddingFunction


def get_llm_client(provider: str = "cerebras") -> "OpenAI":
    """Return the cached OpenAI-compatible client for the chosen provider.

    Args:
        provider: ``"cerebras"`` (default) or ``"openrouter"``. OpenRouter is
            the fallback to use when Cerebras' daily token quota is exhausted —
            it is NOT auto-activated here; the caller decides when to switch.

    Pass the matching model when creating chat completions:
    ``settings.CEREBRAS_MODEL`` for Cerebras, ``settings.OPENROUTER_MODEL`` for
    OpenRouter (they're different — Cerebras and OpenRouter share no model IDs).

    Raises:
        ValueError: if ``provider`` is not ``"cerebras"`` or ``"openrouter"``.
        RuntimeError: if the selected provider's API key is not configured.
    """
    if provider in _llm_clients:
        return _llm_clients[provider]

    if provider == "cerebras":
        api_key = settings.CEREBRAS_API_KEY
        base_url = settings.CEREBRAS_BASE_URL
        env_name = "CEREBRAS_API_KEY"
    elif provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY
        base_url = settings.OPENROUTER_BASE_URL
        env_name = "OPENROUTER_API_KEY"
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. Expected 'cerebras' or 'openrouter'."
        )

    if not api_key:
        raise RuntimeError(
            f"{env_name} is not set. Copy .env.example to .env and add your "
            f"{provider} API key before making LLM calls."
        )

    from openai import OpenAI

    # max_retries above the SDK default (2): the free Cerebras tier throttles
    # bursts with 429 "queue_exceeded", and HistoryOS fires several sequential
    # calls per run (e.g. the grounding layer's per-pool extraction). The SDK
    # honours Retry-After and backs off exponentially, so the extra headroom
    # lets a run ride out transient rate limits instead of crashing mid-pipeline.
    # OpenRouter free tier also throttles, so the same headroom applies there.
    _llm_clients[provider] = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=6,
    )
    return _llm_clients[provider]


def call_with_fallback(messages: list[dict], **kwargs: Any) -> "ChatCompletion":
    """Run a chat-completion on Cerebras; on a 429 / token-quota error, retry
    automatically on OpenRouter.

    Agents call THIS instead of ``get_llm_client().chat.completions.create(...)``
    so failover is centralised — they never see which provider actually served
    the response, only the ``ChatCompletion``.

    ``kwargs`` is forwarded verbatim to ``chat.completions.create`` on the
    primary call, so callers pass ``model=settings.CEREBRAS_MODEL`` plus
    whatever else they need (``temperature``, ``response_format``, ...). On the
    fallback call, ``model`` is overridden to ``settings.OPENROUTER_MODEL``
    (Cerebras and OpenRouter share no model IDs); every OTHER kwarg is passed
    through unchanged.

    Trigger condition: ``openai.RateLimitError`` (HTTP 429). The SDK's
    ``max_retries=6`` already absorbs transient burst-429s with Retry-After
    backoff, so if a ``RateLimitError`` propagates out it means either the rate
    limit is persistent OR the daily token quota is exhausted — either way,
    fallback is the right move. Other openai errors (auth, server, connection)
    bubble up unchanged: a fallback can't fix those.

    Raises:
        openai.OpenAIError: any non-429 openai error from the primary call, OR
            any error from the fallback call (including a 429 on OpenRouter).
        RuntimeError: if ``CEREBRAS_API_KEY`` is missing (primary) or if the
            fallback path is reached and ``OPENROUTER_API_KEY`` is missing.
    """
    from openai import RateLimitError

    try:
        primary = get_llm_client("cerebras")
        return primary.chat.completions.create(messages=messages, **kwargs)
    except RateLimitError as exc:
        logger.warning(
            "Cerebras quota exceeded — falling back to OpenRouter: %s", exc
        )
        fallback_kwargs = {**kwargs, "model": settings.OPENROUTER_MODEL}
        fallback = get_llm_client("openrouter")
        return fallback.chat.completions.create(messages=messages, **fallback_kwargs)


def get_embedding_function():
    """Return the cached ChromaDB-compatible local embedding function.

    Wraps ``sentence-transformers`` model ``settings.EMBEDDING_MODEL``
    (``all-mpnet-base-v2``). The first call downloads the model (~420 MB) once and
    loads it into memory; subsequent calls reuse the cached instance. The returned
    object is callable on a list of texts and is what ChromaDB expects as a
    collection's ``embedding_function``.
    """
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils import embedding_functions

        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
    return _embedding_function
