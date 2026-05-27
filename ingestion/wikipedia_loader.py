"""Wikipedia loader — Phase 1 (Data Ingestion).

Downloads raw historical article text from Wikipedia using the
Wikipedia Python API and persists it to ``data/raw/``.

Responsibilities:
- Accept a list of historical topics / page titles to fetch.
- Pull the full article text (and basic metadata: title, page id,
  source URL, fetch date) for each topic.
- Save one raw text file per article into ``data/raw/`` using a
  deterministic filename so downstream stages can find the source.
- Skip / cache articles that have already been downloaded.

This module only *acquires* text. Cleaning, splitting and embedding
happen later in ``chunker.py`` and ``embedder.py``.

Implementation notes:
- Uses the ``wikipedia-api`` package (``import wikipediaapi``), which is
  the data source declared in ``requirements.txt``. Unlike the older
  ``wikipedia`` package it raises no ``DisambiguationError``; a
  disambiguation page is detected by inspecting the page's categories.
- ``wikipedia-api`` (v0.15+) retries transient failures itself (HTTP 429 /
  5xx / timeouts / connection errors) up to ``max_retries`` with
  exponential backoff, then raises a ``WikipediaException``. We configure
  that retry on the client and simply catch the exception to *skip* the
  topic — so one bad topic never aborts a bulk ingestion run.
- ``page()`` does exact-title lookup (following redirects), not search. So
  when an exact title misses, the loader falls back to Wikipedia full-text
  search and adopts the top-ranked result — logged as "Resolved X -> Y via
  search" and flagged with ``resolved_via_search`` in the metadata. Topics
  with no search hits at all are skipped.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import wikipediaapi
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

# --- Tunables (local to the loader; promote to config.Settings if needed) ----
# wikipedia-api requires a descriptive User-Agent (>= 5 chars) identifying the
# client; see https://meta.wikimedia.org/wiki/User-Agent_policy.
USER_AGENT = "HistoryOS/0.1 (educational; contact: sharmadivyam86@gmail.com)"
REQUEST_TIMEOUT = 15.0  # seconds, forwarded to the underlying httpx.Client
MAX_RETRIES = 3         # wikipedia-api retries transient errors this many times
RETRY_BACKOFF = 1.0     # base seconds; library waits retry_wait * 2**attempt

# Filename sanitisation for cross-platform (incl. Windows) safety.
_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|]+')
_WHITESPACE = re.compile(r"\s+")


def _build_client() -> wikipediaapi.Wikipedia:
    """Construct a Wikipedia client with our User-Agent, retry policy and timeout.

    ``max_retries`` / ``retry_wait`` configure wikipedia-api's built-in retry of
    transient errors; ``timeout`` is forwarded to the underlying ``httpx.Client``.
    """
    return wikipediaapi.Wikipedia(
        user_agent=USER_AGENT,
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,  # plain-text extracts
        max_retries=MAX_RETRIES,
        retry_wait=RETRY_BACKOFF,
        timeout=REQUEST_TIMEOUT,
    )


def _safe_filename(name: str) -> str:
    """Turn a topic into a deterministic, filesystem-safe stem."""
    cleaned = _INVALID_FS_CHARS.sub("", name.strip())
    cleaned = _WHITESPACE.sub("_", cleaned)
    cleaned = cleaned.strip("._")
    return (cleaned.lower() or "untitled")[:120]


def _is_disambiguation(page: wikipediaapi.WikipediaPage) -> bool:
    """True if the page is a disambiguation page (detected via its categories).

    May trigger a network call (categories are fetched lazily); any
    ``WikipediaException`` propagates to the caller, which skips the topic.
    """
    return any("disambiguation" in name.lower() for name in page.categories)


def _resolve_via_search(
    wiki: wikipediaapi.Wikipedia, topic: str
) -> wikipediaapi.WikipediaPage | None:
    """Fall back to full-text search and return the top-ranked page.

    Used when an exact-title lookup misses (the requested topic isn't an
    article title). ``SearchResults.pages`` is relevance-ordered, so the first
    value is the best match. Returns None when search yields no hits.
    """
    results = wiki.search(topic, limit=1)
    if not results.pages:
        return None
    return next(iter(results.pages.values()))


def _save_raw(txt_path, json_path, text: str, metadata: dict) -> None:
    """Persist the article text (.txt) and its metadata sidecar (.json)."""
    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fetch_one(
    wiki: wikipediaapi.Wikipedia, topic: str, force_refresh: bool
) -> Document | None:
    """Fetch a single topic into a Document, or return None to skip it."""
    slug = _safe_filename(topic)
    txt_path = settings.RAW_DIR / f"{slug}.txt"
    json_path = settings.RAW_DIR / f"{slug}.json"

    # Cache: reuse a previous download (text + metadata) without hitting the API.
    if not force_refresh and txt_path.exists() and json_path.exists():
        logger.info("Cache hit for %r -> %s", topic, txt_path.name)
        text = txt_path.read_text(encoding="utf-8")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        return Document(page_content=text, metadata=metadata)

    # All network access (existence, categories, text) is wrapped so that any
    # failure — including transient errors that survived wikipedia-api's own
    # retries — results in a logged skip rather than a crashed batch.
    try:
        page = wiki.page(topic)
        via_search = False

        # Exact-title miss -> fall back to full-text search and take the top hit.
        if not page.exists():
            resolved = _resolve_via_search(wiki, topic)
            if resolved is None:
                logger.warning(
                    "Skipping %r: no exact page and no search match", topic
                )
                return None
            logger.info("Resolved %r -> %r via search", topic, resolved.title)
            page = resolved
            via_search = True

        if _is_disambiguation(page):
            logger.warning(
                "Skipping %r: resolves to a disambiguation page (%s)",
                topic, page.fullurl,
            )
            return None

        text = page.text
        metadata = {
            "title": page.title,
            "url": page.fullurl,
            "fetch_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "page_id": page.pageid,
            "requested_topic": topic,
            "resolved_via_search": via_search,
            "source": "wikipedia",
        }
    except wikipediaapi.WikipediaException as exc:
        logger.warning(
            "Skipping %r: Wikipedia fetch failed after retries (%s)", topic, exc
        )
        return None
    except Exception as exc:  # noqa: BLE001 - one bad topic must not kill the batch
        logger.error("Skipping %r: unexpected error (%s)", topic, exc)
        return None

    _save_raw(txt_path, json_path, text, metadata)
    logger.info("Loaded %r -> %s (%d chars)", topic, txt_path.name, len(text))
    return Document(page_content=text, metadata=metadata)


def load_topics(topics: list[str], *, force_refresh: bool = False) -> list[Document]:
    """Fetch Wikipedia articles for ``topics`` and return them as Documents.

    For each topic the full article text is downloaded (or loaded from the
    ``data/raw/`` cache), saved as ``data/raw/<slug>.txt`` with a ``.json``
    metadata sidecar, and wrapped in a LangChain ``Document`` whose metadata
    carries ``title``, ``url`` and ``fetch_date`` (plus ``page_id`` and the
    originally ``requested_topic`` to aid downstream citation).

    Topics that are missing, ambiguous (disambiguation pages), or fail after
    retries are logged and skipped — the returned list contains only the
    articles that loaded successfully.

    Args:
        topics: Wikipedia page titles / search terms to fetch.
        force_refresh: If True, re-download even when a cached copy exists.

    Returns:
        A list of Documents for the topics that were fetched successfully.
    """
    wiki = _build_client()
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for topic in topics:
        doc = _fetch_one(wiki, topic, force_refresh)
        if doc is not None:
            documents.append(doc)

    logger.info(
        "load_topics complete: %d requested, %d loaded, %d skipped",
        len(topics), len(documents), len(topics) - len(documents),
    )
    return documents


if __name__ == "__main__":
    # Manual smoke test: one normal article, one disambiguation page, one
    # nonsense title. Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m ingestion.wikipedia_loader
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    demo_topics = [
        "French Revolution",                      # normal article -> loaded
        "Mercury",                                # disambiguation  -> skipped
        "Asdkjqwoieur Nonexistent Topic 12345",   # not found       -> skipped
    ]
    docs = load_topics(demo_topics)
    print(f"\nReturned {len(docs)} Document(s):")
    for d in docs:
        print(
            f"  - {d.metadata['title']} "
            f"({len(d.page_content)} chars) {d.metadata['url']}"
        )
