"""Agent 2 — Retrieval Engine.

Second node in the pipeline. Retrieves verified historical context
for the structured query produced by the Query Understanding agent.

Responsibilities:
- Take the retrieval queries / entities from Agent 1.
- Search the ChromaDB vector store (via ``chroma_client.py``) for the
  most relevant chunks, each carrying its citable ``chunk_id``.
- Optionally fall back to / augment with Tavily web search when the
  local store lacks coverage.
- Handle the empty vector store gracefully (Critical Rule #5).
- Return a Pydantic model of retrieved passages with source metadata
  (Critical Rule #4) — raw, not yet judged as "verified".

Provides the context that every later LLM call must be grounded on
(Critical Rule #6: never call the LLM without retrieved context).

Implementation notes:
- This agent makes NO LLM call — it is pure vector search, so Rule #6 is
  trivially satisfied (nothing is generated here).
- ``SearchResult`` is reused from ``vectorstore.chroma_client`` (not redefined);
  it already carries the ``chunk_id`` every downstream fact must cite (Rule #2),
  and ``chroma_client.search`` returns ``[]`` on an empty collection, so the
  empty-state contract (Rule #5) falls out naturally.
- Two result pools are kept separate: ``primary_chunks`` (the directly relevant
  context, from ``search_queries``) and ``analogy_chunks`` (analogous situations
  elsewhere, from ``analogy_queries``). A chunk selected into ``primary_chunks`` is
  excluded from ``analogy_chunks`` so no ``chunk_id`` appears twice.
- Tavily web-search augmentation is intentionally NOT implemented yet; the seam
  is marked in ``retrieve_context`` (see ``settings.TAVILY_API_KEY``).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agents.query_understanding import QueryAnalysis
from config import settings
from vectorstore.chroma_client import SearchResult, search

logger = logging.getLogger(__name__)

# How many chunks survive into each pool after dedup + ranking. Kept as module
# constants (not config) for now — easy to promote to config.py later.
MAX_PRIMARY_CHUNKS = 8
MAX_ANALOGY_CHUNKS = 3
# Results requested per individual query; the per-query pools are then merged,
# deduped and truncated to the caps above.
PER_QUERY_RESULTS = settings.TOP_K


class RetrievalContext(BaseModel):
    """Verified context retrieved for one counterfactual scenario.

    ``primary_chunks`` is the directly relevant context; ``analogy_chunks`` holds
    analogous situations elsewhere/elsewhen (disjoint from ``primary_chunks``).
    Both may be empty when the collection is empty or nothing matches (Rule #5),
    so callers should surface "no verified context found" rather than assume hits.
    """

    primary_chunks: list[SearchResult] = Field(default_factory=list)  # top MAX_PRIMARY_CHUNKS
    analogy_chunks: list[SearchResult] = Field(default_factory=list)  # top MAX_ANALOGY_CHUNKS
    total_searched: int = 0  # total raw hits across all queries, before dedup
    query_used: str = ""  # the scenario being retrieved for (proposed_change)


def _run_queries(queries: list[str], *, label: str) -> list[SearchResult]:
    """Run each query through ChromaDB, logging its hit count; return raw hits.

    Results are concatenated across queries (no dedup yet). ``label`` tags the log
    lines ("primary" / "analogy") so the run is auditable.
    """
    all_hits: list[SearchResult] = []
    for query in queries:
        hits = search(query, n_results=PER_QUERY_RESULTS)
        logger.info("[%s] query %r -> %d result(s)", label, query, len(hits))
        all_hits.extend(hits)
    return all_hits


def _dedup_and_rank(hits: list[SearchResult]) -> list[SearchResult]:
    """Collapse duplicate ``chunk_id``s (keeping the highest score) and sort desc.

    When a chunk is returned by more than one query, its best (highest) similarity
    score wins, so a multi-query match isn't penalised.
    """
    best: dict[str, SearchResult] = {}
    for hit in hits:
        existing = best.get(hit.chunk_id)
        if existing is None or hit.similarity_score > existing.similarity_score:
            best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda r: r.similarity_score, reverse=True)


def retrieve_context(analysis: QueryAnalysis) -> RetrievalContext:
    """Retrieve verified context from ChromaDB for a structured query.

    Runs every ``search_queries`` entry to build the primary pool and every
    ``analogy_queries`` entry to build the analogy pool, dedups + ranks each by
    similarity, keeps the top ``MAX_PRIMARY_CHUNKS`` / ``MAX_ANALOGY_CHUNKS``, and
    excludes any chunk already chosen for ``primary_chunks`` from ``analogy_chunks``
    so no ``chunk_id`` appears twice.

    No LLM or network call is made. On an empty collection (or no matches) the
    returned context is empty-but-valid (Rule #5), and a warning is logged.

    Args:
        analysis: The structured query from Agent 1 (Query Understanding).

    Returns:
        A ``RetrievalContext`` with the primary and analogy chunk pools.
    """
    primary_hits = _run_queries(analysis.search_queries, label="primary")
    analogy_hits = _run_queries(analysis.analogy_queries, label="analogy")
    total_searched = len(primary_hits) + len(analogy_hits)

    primary_chunks = _dedup_and_rank(primary_hits)[:MAX_PRIMARY_CHUNKS]
    primary_ids = {chunk.chunk_id for chunk in primary_chunks}

    # Analogy pool excludes anything already selected as primary (global uniqueness).
    analogy_ranked = _dedup_and_rank(analogy_hits)
    analogy_chunks = [c for c in analogy_ranked if c.chunk_id not in primary_ids][
        :MAX_ANALOGY_CHUNKS
    ]

    # --- Tavily web-search seam (not implemented) ----------------------------
    # If primary_chunks is empty/sparse and settings.TAVILY_API_KEY is set, a web
    # fallback would augment the pool here before returning. Out of scope for now;
    # see module docstring.

    if not primary_chunks and not analogy_chunks:
        logger.warning(
            "retrieve_context: no verified context found for %r (collection empty "
            "or no matches across %d quer(y/ies))",
            analysis.proposed_change,
            len(analysis.search_queries) + len(analysis.analogy_queries),
        )
    else:
        logger.info(
            "retrieve_context: %d primary + %d analogy chunk(s) from %d raw hit(s)",
            len(primary_chunks),
            len(analogy_chunks),
            total_searched,
        )

    return RetrievalContext(
        primary_chunks=primary_chunks,
        analogy_chunks=analogy_chunks,
        total_searched=total_searched,
        query_used=analysis.proposed_change,
    )


if __name__ == "__main__":
    # Offline-ish smoke test: hits the real ChromaDB (no LLM). Run from project root:
    #   D:\historyos\venv\Scripts\python.exe -m agents.retrieval_engine
    import sys

    # Windows console defaults to cp1252; chunk text can carry non-cp1252 chars
    # (macrons, en-dashes), so reconfigure stdout before printing (Known Issue).
    sys.stdout.reconfigure(encoding="utf-8")

    from vectorstore.chroma_client import get_collection_stats

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    stats = get_collection_stats()
    print(f"\nCollection {stats['collection']!r}: {stats['total']} vector(s)")
    for source, count in sorted(stats["per_source"].items()):
        print(f"  - {source}: {count}")

    # Built by hand (no LLM) so this test isolates retrieval from Agent 1.
    # Queries are aimed at the ingested corpus (Mughal / British Raj + analogies).
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
            "Mughal emperors and territorial expansion",
        ],
        analogy_queries=[
            "Ottoman Empire decline and longevity",
            "fall of the Western Roman Empire",
        ],
    )

    print(f"\nScenario: {analysis.proposed_change}\n")
    context = retrieve_context(analysis)

    print(
        f"\ntotal_searched={context.total_searched}  "
        f"primary={len(context.primary_chunks)}  analogy={len(context.analogy_chunks)}"
    )

    def _show(title: str, chunks: list[SearchResult]) -> None:
        print(f"\n{title} ({len(chunks)}):")
        if not chunks:
            print("  (none — no verified context found)")
        for chunk in chunks:
            print(
                f"  [{chunk.similarity_score:.3f}] {chunk.chunk_id}  ({chunk.source})\n"
                f"      {chunk.text[:120].strip()}..."
            )

    _show("PRIMARY", context.primary_chunks)
    _show("ANALOGY", context.analogy_chunks)
