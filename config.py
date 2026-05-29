"""Central configuration for HistoryOS.

Loads environment variables from ``.env`` (via python-dotenv) and
exposes them, alongside the project's fixed settings, as importable
constants. Every other module should read configuration from here
rather than touching ``os.environ`` directly.

Usage:
    from core.llm_client import get_llm_client
    client = get_llm_client()  # OpenAI-compatible client pointed at Cerebras

Copy ``.env.example`` to ``.env`` and fill in the real keys before
running the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (no-op if the file is absent).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating empty strings as unset."""
    value = os.getenv(name, default)
    if value is not None and value.strip() == "":
        return default
    return value


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Truthy: 1/true/yes/on (case-insensitive)."""
    raw = _get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """All HistoryOS configuration in one place."""

    # --- Secret API keys (set these in .env) ---------------------------------
    # Cerebras powers all LLM calls via its OpenAI-compatible API (free tier).
    # OpenRouter is the fallback provider when Cerebras hits its daily token
    # quota. Embeddings are local (sentence-transformers), so no OpenAI key is
    # needed.
    CEREBRAS_API_KEY: str | None = _get("CEREBRAS_API_KEY")
    OPENROUTER_API_KEY: str | None = _get("OPENROUTER_API_KEY")
    TAVILY_API_KEY: str | None = _get("TAVILY_API_KEY")

    # --- LLM: Cerebras (OpenAI-compatible endpoint) --------------------------
    CEREBRAS_BASE_URL: str = _get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    CEREBRAS_MODEL: str = _get("CEREBRAS_MODEL", "gpt-oss-120b")
    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.0)

    # --- LLM: OpenRouter fallback (OpenAI-compatible endpoint) ---------------
    OPENROUTER_BASE_URL: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = _get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    # --- Embeddings: local sentence-transformers (no API key, runs on torch) -
    EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "all-mpnet-base-v2")

    # --- Vector store (ChromaDB) ---------------------------------------------
    CHROMA_PERSIST_DIR: str = _get(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma")
    )
    CHROMA_COLLECTION: str = _get("CHROMA_COLLECTION", "historios")

    # --- Data paths ----------------------------------------------------------
    RAW_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # --- Ingestion / chunking ------------------------------------------------
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 300)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 100)

    # --- Retrieval -----------------------------------------------------------
    TOP_K: int = _get_int("TOP_K", 5)

    # --- Dynamic retrieval fallback (live Wikipedia) -------------------------
    # When local Chroma search finds nothing, fetch Wikipedia on demand, embed it
    # into the store, and retry — so out-of-corpus questions still get grounded.
    # Makes ZERO LLM calls (vector search + local embeddings only), so Rule #6
    # stays trivially satisfied. See agents/retrieval_engine._dynamic_retrieve.
    ENABLE_DYNAMIC_RETRIEVAL: bool = _get_bool("ENABLE_DYNAMIC_RETRIEVAL", True)
    DYNAMIC_SEARCH_LIMIT: int = _get_int("DYNAMIC_SEARCH_LIMIT", 5)  # pages/run
    DYNAMIC_CHUNK_CAP: int = _get_int("DYNAMIC_CHUNK_CAP", 200)  # max chunks added/run
    # Cosine-similarity floor (1 - distance) below which the BEST local primary hit
    # is treated as "no real match" — the trigger for the dynamic fallback. A
    # populated Chroma collection ALWAYS returns top-k regardless of relevance, so
    # an empty-pool check alone never fires for out-of-corpus questions. Calibrated
    # from observed scores: in-corpus tops ~0.72-0.84, out-of-corpus ~0.36-0.53.
    DYNAMIC_MIN_SIMILARITY: float = _get_float("DYNAMIC_MIN_SIMILARITY", 0.6)

    # --- Reasoning guardrails ------------------------------------------------
    # Critical Rule #3: max 4 causal reasoning steps (hallucination guard).
    MAX_CAUSAL_STEPS: int = _get_int("MAX_CAUSAL_STEPS", 4)

    def validate(self) -> None:
        """Raise if a required secret is missing.

        Call this at startup (pipeline / frontend) so failures are loud
        and early rather than mid-run.
        """
        missing = [
            name
            for name in ("CEREBRAS_API_KEY",)
            if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


# Singleton imported everywhere: ``from config import settings``.
settings = Settings()
