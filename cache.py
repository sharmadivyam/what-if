"""File-based result cache for WHAT IF? — instant replay of pipeline answers.

A GENERAL cache, not a table of canned answers. The key is the normalized
question; the value is the subset of the pipeline's ``HistoriosState`` that the
frontend's render path actually consumes. The SAME code path serves a cache hit
whether the question was pre-warmed (``scripts/prewarm.py``) or asked by an
earlier visitor — no question string is ever special-cased.

What is stored (the render contract — see ``frontend/app.py``'s
``render_results`` / ``_record_history``):
    question, status, error, failed_node, timings,
    grounded (GroundedContext), scored (ScoredReasoning)
The pipeline's ``analysis`` / ``context`` / ``reasoning`` are intentionally NOT
stored because nothing in the render path reads them. If the render path ever
starts reading another state field, extend ``_serialize_state`` /
``_deserialize_state`` to match (and bump ``_SCHEMA_VERSION``).

Serialization: ``GroundedContext`` and ``ScoredReasoning`` are Pydantic v2
models over plain str/int/bool/list/dict, so ``model_dump()`` → JSON →
``model_validate()`` round-trips losslessly without touching the pipeline's data
structures.

Robustness: ``get`` treats a missing, corrupt, or schema-mismatched file as a
plain miss (returns ``None``, never raises). ``put`` writes atomically and
swallows every error, so a cache write can never break a live run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
from pathlib import Path
from typing import Any

from agents.confidence_scorer import ScoredReasoning
from agents.grounding_layer import GroundedContext

logger = logging.getLogger(__name__)

# Cache lives next to this module (project root) in a committed ``cache/`` dir.
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Bump when the stored shape changes so stale files are treated as misses.
_SCHEMA_VERSION = 1

# Trailing characters trimmed during normalization so "…never fell" and
# "…never fell?" map to the same entry. Internal punctuation is preserved.
_TRAILING = string.punctuation + string.whitespace


# --- Key / normalization -----------------------------------------------------


def _normalize(question: str) -> str:
    """Lowercase, trim, collapse internal whitespace, strip trailing punctuation."""
    collapsed = " ".join((question or "").lower().strip().split())
    return collapsed.rstrip(_TRAILING)


def _key(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()


def _path(question: str) -> Path:
    return CACHE_DIR / f"{_key(question)}.json"


# --- (De)serialization -------------------------------------------------------


def _serialize_state(question: str, state: dict[str, Any]) -> dict[str, Any]:
    """Pick the render-relevant keys from a HistoriosState into a JSON-able dict.

    Only known keys are copied, so transient flags (e.g. ``from_cache``) are
    never persisted. Pydantic models are dumped to plain dicts.
    """
    grounded = state.get("grounded")
    scored = state.get("scored")
    return {
        "schema_version": _SCHEMA_VERSION,
        "normalized_question": _normalize(question),
        "state": {
            "question": state.get("question") or question,
            "status": state.get("status", "ok"),
            "error": state.get("error"),
            "failed_node": state.get("failed_node"),
            "timings": state.get("timings") or {},
            "grounded": grounded.model_dump() if grounded is not None else None,
            "scored": scored.model_dump() if scored is not None else None,
        },
    }


def _deserialize_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a render-ready state dict (Pydantic models reconstructed)."""
    raw = payload["state"]
    grounded_raw = raw.get("grounded")
    scored_raw = raw.get("scored")
    return {
        "question": raw.get("question"),
        "status": raw.get("status", "ok"),
        "error": raw.get("error"),
        "failed_node": raw.get("failed_node"),
        "timings": raw.get("timings") or {},
        "grounded": (
            GroundedContext.model_validate(grounded_raw)
            if grounded_raw is not None
            else None
        ),
        "scored": (
            ScoredReasoning.model_validate(scored_raw) if scored_raw is not None else None
        ),
        "from_cache": True,
    }


# --- Public API --------------------------------------------------------------


def get(question: str) -> dict[str, Any] | None:
    """Return the cached state for ``question``, or ``None`` on any miss.

    A missing file, corrupt JSON, a schema mismatch, or a model that fails to
    validate are ALL treated as a plain miss — this function never raises, so
    the caller can simply fall through to running the live pipeline.
    """
    path = _path(question)
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            logger.info("cache: schema mismatch in %s — treating as miss", path.name)
            return None
        state = _deserialize_state(payload)
        logger.info("cache: HIT for %.60r", question)
        return state
    except Exception:  # noqa: BLE001 — any read failure is just a miss
        logger.warning("cache: failed to read %s — treating as miss", path.name, exc_info=True)
        return None


def put(question: str, state: dict[str, Any]) -> None:
    """Write ``state`` for ``question`` to the cache. Never raises.

    Writes to a temp file then atomically replaces, so a crashed write can't
    leave a half-written entry. Any failure is logged and swallowed — caching
    is best-effort and must never break a live run.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = _serialize_state(question, state)
        path = _path(question)
        tmp = path.parent / (path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        logger.info("cache: wrote %s for %.60r", path.name, question)
    except Exception:  # noqa: BLE001 — a cache write must never sink a live run
        logger.warning("cache: failed to write entry for %.60r — skipping", question, exc_info=True)
