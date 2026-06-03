"""Pre-warm the WHAT IF? result cache over the canonical example questions.

Runs the REAL pipeline over ``frontend.examples.EXAMPLES`` (the single source of
truth the UI also renders) and saves each successful result via ``cache.put``.
The resulting ``cache/*.json`` files are committed so the public demo serves the
example questions instantly.

Only ``status == "ok"`` runs are cached — a rate-limited / errored run is left
uncached so the next ask re-runs it rather than freezing a failure.

Run from the project root with the venv interpreter (see CLAUDE.md ENV GOTCHA):

    D:\\historyos\\venv\\Scripts\\python.exe -m scripts.prewarm
    D:\\historyos\\venv\\Scripts\\python.exe -m scripts.prewarm --force   # rebuild all

Sequential by design — the free Cerebras tier hard-throttles bursts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as either ``-m scripts.prewarm`` or ``scripts/prewarm.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cache  # noqa: E402
from frontend.examples import EXAMPLES  # noqa: E402
from pipeline.historios_pipeline import run as run_pipeline  # noqa: E402


def main() -> None:
    # Reasoning/claim text carries non-cp1252 chars; Windows console defaults to
    # cp1252 (see CLAUDE.md "WINDOWS CONSOLE ENCODING").
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Pre-warm the WHAT IF? result cache.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run and overwrite cache entries even if already present.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    total = len(EXAMPLES)
    warmed = skipped = failed = 0

    for i, question in enumerate(EXAMPLES, start=1):
        print(f"\n[{i}/{total}] {question}")

        if not args.force and cache.get(question) is not None:
            print("  - already cached, skipping (use --force to rebuild)")
            skipped += 1
            continue

        state = run_pipeline(question)
        status = state.get("status")
        if status == "ok":
            cache.put(question, state)
            warmed += 1
            print("  + cached (status=ok)")
        else:
            failed += 1
            print(f"  ! not cached (status={status}: {state.get('error') or 'no error'})")

    print(
        f"\nDone: {warmed} warmed, {skipped} skipped, {failed} failed "
        f"(of {total} example{'s' if total != 1 else ''})."
    )


if __name__ == "__main__":
    main()
