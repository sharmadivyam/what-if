"""Re-warm cache entries whose stored answer has a thin reasoning chain.

Targets only cached examples with fewer than ``MIN_STEPS`` simulated steps
(some runs parse a degraded/truncated chain and land 0-1 steps even at
status=ok). Re-runs the pipeline for those and overwrites the cache ONLY if the
new run is strictly better (more steps). A worse, equal, or failed re-run leaves
the existing entry untouched — so this can never regress the cache.

Run from the project root with the venv interpreter:

    D:\\historyos\\venv\\Scripts\\python.exe -m scripts.rewarm_thin
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cache  # noqa: E402
from frontend.examples import EXAMPLES  # noqa: E402
from pipeline.historios_pipeline import run as run_pipeline  # noqa: E402

# Minimum simulated-step count for a cached answer to count as "full" enough.
MIN_STEPS = 2


def _steps(state: dict | None) -> int:
    scored = state.get("scored") if state else None
    return len(scored.steps) if scored is not None else 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    thin = [(q, _steps(s)) for q in EXAMPLES for s in [cache.get(q)]
            if s is not None and _steps(s) < MIN_STEPS]

    print(f"Found {len(thin)} thin entr(y/ies) (< {MIN_STEPS} steps):")
    for q, n in thin:
        print(f"  - [{n} step(s)] {q}")

    improved = kept = failed = 0
    for i, (q, old) in enumerate(thin, start=1):
        print(f"\n[{i}/{len(thin)}] {q}  (cached: {old} step(s))")
        state = run_pipeline(q)
        if state.get("status") != "ok":
            failed += 1
            print(f"  ! run failed (status={state.get('status')}) — keeping existing {old}-step entry")
            continue
        new = _steps(state)
        if new > old:
            cache.put(q, state)
            improved += 1
            print(f"  + improved {old} -> {new} step(s); cache overwritten")
        else:
            kept += 1
            print(f"  = no improvement ({new} <= {old}); keeping existing entry")

    print(f"\nDone: {improved} improved, {kept} kept, {failed} failed (of {len(thin)} thin).")


if __name__ == "__main__":
    main()
