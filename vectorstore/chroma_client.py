"""ChromaDB client — local vector storage.

Thin wrapper around ChromaDB that the ingestion and retrieval layers
share. Owns the persistent client and the HistoryOS collection.

Responsibilities:
- Create / load a persistent ChromaDB client at the configured path
  (CHROMA_PERSIST_DIR) and get-or-create the collection.
- Provide ``store`` to upsert text + metadata (embeddings are computed by
  the collection's local embedding function).
- Provide ``search`` / ``search_with_filter`` for similarity search,
  returning ``SearchResult`` objects that carry the citable ``chunk_id``.
- Provide ``get_collection_stats`` for collection summaries.
- Handle the EMPTY-STATE gracefully (Critical Rule #5): if the
  collection has no documents, return an explicit empty result instead
  of raising, so the pipeline can surface "no verified context found".

All ChromaDB access in the project should go through this module.

Implementation notes:
- The collection is created with cosine distance (``hnsw:space: cosine``) and the
  local embedding function from ``core/llm_client.get_embedding_function()`` — so
  callers add/search with raw text and Chroma embeds it with the SAME model on both
  paths (Critical Rule #7). We deliberately do NOT pass precomputed embeddings:
  embeddings are never produced by a paid API (the project is ₹0 / no OpenAI).
- ``SearchResult.similarity_score = 1 - distance`` (cosine, ~0..1, higher = closer).
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from config import settings
from core.llm_client import get_embedding_function

logger = logging.getLogger(__name__)

# Chroma metadata values must be scalar (str/int/float/bool) and non-null; these
# are the Chunk fields we persist for citation + filtering.
_METADATA_FIELDS = ("source_title", "source_url", "token_count", "chunk_index", "page_id")

_client: chromadb.api.ClientAPI | None = None
_collection = None


class SearchResult(BaseModel):
    """One retrieved chunk with its citation metadata and similarity score."""

    chunk_id: str
    text: str
    source: str  # article title (from metadata ``source_title``)
    similarity_score: float  # 1 - cosine_distance, ~0..1 (higher = more similar)
    source_url: str = ""  # kept beyond the base spec — needed to cite (Rule #2)


def get_client() -> chromadb.api.ClientAPI:
    """Return (and cache) the persistent ChromaDB client at the config path."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    """Return (and cache) the HistoryOS collection (created on first use).

    Cosine space + the local embedding function are bound here, so callers never
    embed text themselves.
    """
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _metadata_for(chunk) -> dict:
    """Build a Chroma-safe metadata dict from a Chunk (drops null values)."""
    meta = {field: getattr(chunk, field) for field in _METADATA_FIELDS}
    return {k: v for k, v in meta.items() if v is not None}


def store(chunks, *, batch_size: int = 128) -> None:
    """Upsert chunks (text + metadata) into the collection.

    Embeddings are computed by the collection's local embedding function — callers
    pass chunks only, never vectors. ``upsert`` makes this idempotent: re-storing
    the same ``chunk_id`` overwrites rather than duplicates, so ingestion is safely
    re-runnable.
    """
    if not chunks:
        return

    collection = get_collection()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[_metadata_for(c) for c in batch],
        )


def _to_results(res) -> list[SearchResult]:
    """Flatten a single-query ChromaDB result into ``SearchResult`` objects."""
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    return [
        SearchResult(
            chunk_id=chunk_id,
            text=doc,
            source=meta.get("source_title", ""),
            similarity_score=1.0 - dist,
            source_url=meta.get("source_url", ""),
        )
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists)
    ]


def search(query: str, n_results: int = settings.TOP_K) -> list[SearchResult]:
    """Similarity search; up to ``n_results`` results, most similar first.

    Returns ``[]`` when the collection is empty (Critical Rule #5) so callers can
    surface "no verified context found" instead of crashing.
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("search(%r): collection is empty", query)
        return []
    res = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
    return _to_results(res)


def search_with_filter(
    query: str, filter_dict: dict, n_results: int = settings.TOP_K
) -> list[SearchResult]:
    """Similarity search restricted by a metadata filter (Chroma ``where`` clause).

    ``filter_dict`` is a ChromaDB ``where`` filter over stored metadata, e.g.
    ``{"source_title": "British Raj"}``. Returns ``[]`` on an empty collection.
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("search_with_filter(%r, %r): collection is empty", query, filter_dict)
        return []
    res = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where=filter_dict,
    )
    return _to_results(res)


def get_collection_stats() -> dict:
    """Collection summary: total vectors and per-article (source_title) counts."""
    collection = get_collection()
    total = collection.count()
    per_source: dict[str, int] = {}
    if total:
        metas = collection.get(include=["metadatas"])["metadatas"]
        for meta in metas:
            title = meta.get("source_title", "unknown")
            per_source[title] = per_source.get(title, 0) + 1
    return {"collection": settings.CHROMA_COLLECTION, "total": total, "per_source": per_source}
