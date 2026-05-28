"""Provider clients — the single source of LLM + embedding access.

HistoryOS uses a zero-cost stack:
- LLM: Cerebras ``qwen-3-235b-a22b-instruct-2507`` through its OpenAI-compatible
  API, reached with the standard ``openai`` client pointed at the Cerebras base URL.
- Embeddings: the local ``sentence-transformers`` model ``all-mpnet-base-v2``,
  wrapped in ChromaDB's ``SentenceTransformerEmbeddingFunction`` (no API key).

IMPORTANT — every module (especially the agents) MUST obtain its clients here via
``get_llm_client()`` / ``get_embedding_function()``. Do NOT instantiate ``OpenAI``
or an embedding function directly in agent / pipeline files: centralising it keeps
the provider swappable and the model/credentials configured in exactly one place
(``config.py``). Both clients are cached module-level singletons because building
them — especially loading the embedding model — is expensive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:  # avoid importing heavy deps at module import time for typing
    from openai import OpenAI

_llm_client: "OpenAI | None" = None
_embedding_function = None  # chromadb SentenceTransformerEmbeddingFunction


def get_llm_client() -> "OpenAI":
    """Return the cached OpenAI-compatible client pointed at Cerebras.

    All LLM calls in HistoryOS go through this client; pass
    ``model=settings.CEREBRAS_MODEL`` when creating chat completions.

    Raises:
        RuntimeError: if ``CEREBRAS_API_KEY`` is not configured.
    """
    global _llm_client
    if _llm_client is None:
        if not settings.CEREBRAS_API_KEY:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Copy .env.example to .env and add "
                "your (free) Cerebras key before making LLM calls."
            )
        from openai import OpenAI

        # max_retries above the SDK default (2): the free Cerebras tier throttles
        # bursts with 429 "queue_exceeded", and HistoryOS fires several sequential
        # calls per run (e.g. the grounding layer's per-chunk extraction). The SDK
        # honours Retry-After and backs off exponentially, so the extra headroom
        # lets a run ride out transient rate limits instead of crashing mid-pipeline.
        _llm_client = OpenAI(
            api_key=settings.CEREBRAS_API_KEY,
            base_url=settings.CEREBRAS_BASE_URL,
            max_retries=6,
        )
    return _llm_client


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
