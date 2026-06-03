"""Canonical example questions for WHAT IF? — the single source of truth.

The Streamlit UI renders these as the landing-page example chips, and the
pre-warm script (``scripts/prewarm.py``) runs the real pipeline over exactly
this list to populate the committed result cache. Keep this as the ONE place
the example set is defined so the UI and the cache can never drift apart.

This module is import-safe: it has no side effects, so the pre-warm script can
``from frontend.examples import EXAMPLES`` without triggering ``app.main()``
(importing ``frontend.app`` would run the whole Streamlit app).

The landing grid lays these out 3 per row, so a multiple of 3 keeps the rows
even (currently 9 → 3 rows of 3).

ORDER MATTERS — this list is rendered top-to-bottom, so it is curated
best-first: the examples whose pre-warmed answers have the fullest reasoning
chains lead, and the ones that currently cache a thin/empty chain sit last (the
bottom row). Re-order if a re-warm (``scripts/rewarm_thin.py``) improves a thin
entry, so the strongest demos always stay on top.
"""

from __future__ import annotations

EXAMPLES: list[str] = [
    # --- Full ~4-step pre-warmed answers (strongest demos) --------------------
    "What if Constantinople had not fallen to the Ottomans in 1453?",
    "What if the Library of Alexandria had never been destroyed?",
    "What if the Cuban Missile Crisis had escalated into nuclear war?",
    "What if the Industrial Revolution had begun in China instead of Britain?",
    "What if Genghis Khan had died in childhood before unifying the Mongols?",
    "What if Britain had never colonized India?",
    "What if the Western Roman Empire had never fallen?",
    "What if the Ottoman Empire had survived past 1922?",
    # --- Thinner pre-warmed answer (1-step) — improvable via rewarm_thin ------
    "What if the Mughal Empire had industrialized before the British arrived?",
]
