"""Canonical corpus topic list for HistoryOS ingestion.

Single source of truth for which Wikipedia articles make up the retrieval
corpus. ``wikipedia_loader`` ingests these; to grow the store, add a topic here
and re-run the ingestion pipeline (loader -> chunker -> embedder).

A few entries are descriptive phrases rather than exact page titles (e.g.
"Meiji Restoration Japan", "Ottoman Empire modernization"). The loader does an
exact-title lookup first and, on a miss, falls back to Wikipedia full-text
search, adopting the top-ranked page and flagging it ``resolved_via_search`` —
so check the loader's "Resolved X -> Y via search" log lines to see what each
of those actually mapped to.
"""

from __future__ import annotations

CORPUS_TOPICS: list[str] = [
    # --- Original corpus (Phase 1) -------------------------------------------
    "Ancient Egypt",
    "British Raj",
    "Causes of World War II",
    "Fall of the Western Roman Empire",
    "French Revolution",
    "Mongol Empire",
    "Mughal Empire",
    "Ottoman Empire",
    # --- Expansion 2026-05-28: analogy depth + coverage breadth --------------
    "Byzantine Empire",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Martin Luther King Jr",
    "Mongol invasions of India",
    "Meiji Restoration Japan",
    "Ottoman Empire modernization",
    "Indian independence movement",
    "Cold War",
    "Industrial Revolution",
]
