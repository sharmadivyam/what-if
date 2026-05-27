"""Embedder — Phase 1 (Data Ingestion).

Embeds the processed chunks from ``data/processed/`` and loads the
resulting vectors into the ChromaDB collection.

Responsibilities:
- Read processed chunks (text + chunk_id + source metadata).
- Generate embeddings via the local embedding model (``all-mpnet-base-v2``),
  obtained through ``get_embedding_function()`` in ``core/llm_client.py`` — never
  instantiate the embedding function directly (Critical Rule #7). Embedding is
  performed by ChromaDB's collection embedding function inside ``chroma_client``;
  this module never calls a paid embedding API.
- Upsert each vector into ChromaDB through ``chroma_client.py``,
  storing the chunk text and metadata so retrieval can return the
  citable ``chunk_id`` for every result.
- Batch in groups of 50 (progress is printed per batch) and skip chunks
  that are already embedded.

This is the final ingestion step; after it runs the vector store is
ready for the retrieval agent.
"""

from __future__ import annotations

import json
import logging

from config import settings
from ingestion.chunker import Chunk
from vectorstore import chroma_client

logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # embed/store in groups of this many; also the progress cadence


def load_processed_chunks() -> list[Chunk]:
    """Read ``data/processed/*.jsonl`` back into ``Chunk`` objects.

    Inverse of ``chunker.save_chunks``; lets the embedder run from the on-disk
    processed files without re-loading or re-chunking.
    """
    chunks: list[Chunk] = []
    if not settings.PROCESSED_DIR.exists():
        return chunks
    for path in sorted(settings.PROCESSED_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def embed_and_store(chunks: list[Chunk], *, skip_existing: bool = True) -> None:
    """Embed + store chunks in ChromaDB, batching in groups of ``BATCH_SIZE``.

    With ``skip_existing`` (default), chunk_ids already present in the collection
    are dropped first, so re-runs only embed new content. Embedding + storage are
    delegated to ``chroma_client.store`` (local embedding function — Rule #7).
    Progress is printed every ``BATCH_SIZE`` chunks.
    """
    if not chunks:
        print("embed_and_store: nothing to embed")
        return

    to_store = chunks
    if skip_existing:
        collection = chroma_client.get_collection()
        existing = set(collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        to_store = [c for c in chunks if c.chunk_id not in existing]
        if len(to_store) != len(chunks):
            print(f"Skipping {len(chunks) - len(to_store)} already-embedded chunk(s)")

    total = len(to_store)
    for start in range(0, total, BATCH_SIZE):
        batch = to_store[start : start + BATCH_SIZE]
        chroma_client.store(batch)
        print(f"  embedded {min(start + BATCH_SIZE, total)}/{total} chunks")

    logger.info("embed_and_store complete: %d stored, %d skipped",
                total, len(chunks) - total)


if __name__ == "__main__":
    # Final ingestion step in isolation: embed whatever is in data/processed/.
    #   D:\historyos\venv\Scripts\python.exe -m ingestion.embedder
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # article titles may exceed cp1252
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    chunks = load_processed_chunks()
    embed_and_store(chunks)
    stats = chroma_client.get_collection_stats()
    print(f"\nCollection {stats['collection']!r}: {stats['total']} vectors")
    for title, n in sorted(stats["per_source"].items()):
        print(f"  - {title}: {n}")
