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
import time
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

# Fixed inter-retry wait (seconds). By default the SDK honours the server's
# ``Retry-After`` header, which on the free tiers swings unpredictably between
# ~13s and 60s. We pin every retry to a steady interval instead so back-off is
# predictable (applied via a per-instance override in get_llm_client — no SDK
# edit). max_retries still bounds the number of attempts.
_RETRY_INTERVAL_SECONDS = 30.0

# Embedding-model load retries. On a fresh deployment (e.g. Streamlit Cloud) the
# ~420 MB sentence-transformers model is downloaded from Hugging Face on first
# use; that download can fail transiently (throttled shared IPs, timeouts), so
# the load is attempted a few times before giving up with a clear error.
_EMBED_LOAD_ATTEMPTS = 3
_EMBED_RETRY_WAIT_SECONDS = 10.0


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
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=6,
    )

    # Pin the inter-retry wait to a fixed interval. The SDK calls
    # ``self._calculate_retry_timeout(remaining_retries, options, response_headers)``
    # before each retry and sleeps the returned seconds; overriding it on the
    # instance forces a steady wait regardless of the server's Retry-After.
    client._calculate_retry_timeout = (
        lambda remaining_retries, options, response_headers=None: _RETRY_INTERVAL_SECONDS
    )

    _llm_clients[provider] = client
    return client


def _malformed_reason(response: "ChatCompletion") -> str | None:
    """Return why a ChatCompletion is unusable, or ``None`` if it looks valid.

    Providers sometimes answer HTTP 200 with a body that is NOT a completion
    (e.g. an ``{"error": ...}`` payload from OpenRouter's free tier, or Cerebras
    under queue pressure). The OpenAI SDK does not raise on those — it builds a
    ``ChatCompletion`` with ``choices=None``, and the first ``choices[0]``
    downstream explodes with ``TypeError: 'NoneType' object is not
    subscriptable``. Detect that shape here so callers can fall back / fail loud.
    """
    if not getattr(response, "choices", None):
        return "response has no choices"
    if response.choices[0].message is None:
        return "first choice has no message"
    return None


def _response_body(response: "ChatCompletion") -> str:
    """Best-effort short dump of a response body for diagnostics (never raises)."""
    try:
        return response.model_dump_json(exclude_none=True)[:500]
    except Exception:  # noqa: BLE001 — diagnostics must not mask the real error
        return repr(response)[:500]


def call_with_fallback(messages: list[dict], **kwargs: Any) -> "ChatCompletion":
    """Run a chat-completion on Cerebras; on a 429 / token-quota error OR a
    malformed (no-choices) 200 response, retry automatically on OpenRouter.

    Agents call THIS instead of ``get_llm_client().chat.completions.create(...)``
    so failover is centralised — they never see which provider actually served
    the response, only the ``ChatCompletion``.

    ``kwargs`` is forwarded verbatim to ``chat.completions.create`` on the
    primary call, so callers pass ``model=settings.CEREBRAS_MODEL`` plus
    whatever else they need (``temperature``, ``response_format``, ...). On the
    fallback call, ``model`` is overridden to ``settings.OPENROUTER_MODEL``
    (Cerebras and OpenRouter share no model IDs); every OTHER kwarg is passed
    through unchanged.

    Trigger conditions:
    - ``openai.RateLimitError`` (HTTP 429). The SDK's ``max_retries=6`` already
      absorbs transient burst-429s with Retry-After backoff, so if a
      ``RateLimitError`` propagates out it means either the rate limit is
      persistent OR the daily token quota is exhausted — either way, fallback
      is the right move. Other openai errors (auth, server, connection) bubble
      up unchanged: a fallback can't fix those.
    - A malformed 200 response (``choices`` missing/empty — typically an error
      payload in the body; see ``_malformed_reason``). The guarantee to callers
      is that the RETURNED completion always has ``choices[0].message``, so
      agents can subscript it safely.

    Raises:
        openai.OpenAIError: any non-429 openai error from the primary call, OR
            any error from the fallback call (including a 429 on OpenRouter).
        RuntimeError: if ``CEREBRAS_API_KEY`` is missing (primary); if the
            fallback path is reached and ``OPENROUTER_API_KEY`` is missing; or
            if the fallback response is ALSO malformed (the raw body is included
            in the message so the actual provider error is visible).
    """
    from openai import RateLimitError

    try:
        primary = get_llm_client("cerebras")
        response = primary.chat.completions.create(messages=messages, **kwargs)
        reason = _malformed_reason(response)
        if reason is None:
            return response
        logger.warning(
            "Cerebras returned a malformed completion (%s) — falling back to "
            "OpenRouter. Body: %s",
            reason,
            _response_body(response),
        )
    except RateLimitError as exc:
        logger.warning(
            "Cerebras quota exceeded — falling back to OpenRouter: %s", exc
        )

    fallback_kwargs = {**kwargs, "model": settings.OPENROUTER_MODEL}
    fallback = get_llm_client("openrouter")
    response = fallback.chat.completions.create(messages=messages, **fallback_kwargs)
    reason = _malformed_reason(response)
    if reason is not None:
        raise RuntimeError(
            f"OpenRouter (fallback) returned a malformed completion ({reason}). "
            f"Raw body: {_response_body(response)}"
        )
    return response


def get_embedding_function():
    """Return the cached ChromaDB-compatible local embedding function.

    Wraps ``sentence-transformers`` model ``settings.EMBEDDING_MODEL``
    (``all-mpnet-base-v2``). The first call downloads the model (~420 MB) once and
    loads it into memory; subsequent calls reuse the cached instance. The returned
    object is callable on a list of texts and is what ChromaDB expects as a
    collection's ``embedding_function``.

    The download can fail transiently on a fresh deployment (Streamlit Cloud
    shared IPs are throttled by Hugging Face; see the HF hardening in
    ``config.py``), so the load is retried ``_EMBED_LOAD_ATTEMPTS`` times before
    raising a ``RuntimeError`` that says what to fix (add ``HF_TOKEN``) instead
    of a bare connection error.

    Raises:
        RuntimeError: if the model cannot be loaded after all attempts.
    """
    global _embedding_function
    if _embedding_function is None:
        from chromadb.utils import embedding_functions

        last_exc: Exception | None = None
        for attempt in range(1, _EMBED_LOAD_ATTEMPTS + 1):
            try:
                _embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=settings.EMBEDDING_MODEL
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 — network/HF errors vary widely
                last_exc = exc
                logger.warning(
                    "Embedding model %r failed to load (attempt %d/%d): %s",
                    settings.EMBEDDING_MODEL,
                    attempt,
                    _EMBED_LOAD_ATTEMPTS,
                    exc,
                )
                if attempt < _EMBED_LOAD_ATTEMPTS:
                    time.sleep(_EMBED_RETRY_WAIT_SECONDS * attempt)
        if _embedding_function is None:
            raise RuntimeError(
                f"Could not load embedding model {settings.EMBEDDING_MODEL!r} after "
                f"{_EMBED_LOAD_ATTEMPTS} attempts — the Hugging Face download keeps "
                "failing. On a fresh deployment (e.g. Streamlit Cloud) the model "
                "(~420 MB) must be downloaded once, and unauthenticated downloads "
                "from shared IPs are rate-limited. Add a free read token as HF_TOKEN "
                "(https://huggingface.co/settings/tokens) to the app's secrets/"
                "environment and try again."
            ) from last_exc
    return _embedding_function
