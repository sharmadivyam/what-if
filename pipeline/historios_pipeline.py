"""HistoryOS pipeline — agent orchestration (LangGraph).

Wires the five agents into a single sequential ``StateGraph`` and exposes the
entry point (``run``) that the frontend and evaluator call.

The flow (one node per agent):

    understand_query -> retrieve -> ground -> reason -> score

    question:str
         |
         v
    understand_query   Agent 1  analyze_query()                  -> analysis
         |
         v
    retrieve           Agent 2  retrieve_context()  (no LLM)      -> context
         |  \\
         |   `-- no_context --> END        (empty store, Rules #5/#6)
         v
    ground             Agent 3  ground_context()    (<=2 LLM)     -> grounded
         |
         v
    reason             Agent 4  reason_about_counterfactual()     -> reasoning
         |                       (self short-circuits if no facts, Rule #6)
         v
    score              Agent 5  score_reasoning()   (no LLM)      -> scored
         |
         v
        END
    (any node error -> recorded on state, routed straight to END — never crashes)

Design notes:
- GRACEFUL FAILURE (don't crash): every node runs inside ``_run_node`` which times
  the agent call and catches any ``Exception``, recording it on ``state["error"]``
  / ``state["failed_node"]`` instead of propagating. A shared ``_route`` then sends
  the run to ``END`` as soon as an error is present, so the remaining nodes are
  skipped. ``run`` adds an outer guard so it itself can never raise.
- TIMING: each node records its wall-clock seconds into ``state["timings"][name]``.
  No LangGraph reducer is needed — the graph is strictly sequential, so each node
  reads the accumulated ``timings`` dict and returns it extended by one entry.
- EMPTY STATE (Rules #5/#6): if retrieval finds nothing, ``retrieve`` routes
  ``no_context`` -> END (no point invoking the LLM-bearing nodes). If grounding
  yields chunks but no usable facts, Agent 4 self-short-circuits without an LLM
  call, so that path is left to flow through naturally.
- The agent functions are imported and used UNCHANGED; this module only orchestrates.
  No LLM client is instantiated here (Critical Rule #7 — the agents own that).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from agents.confidence_scorer import ScoredReasoning, score_reasoning
from agents.grounding_layer import GroundedContext, ground_context
from agents.query_understanding import QueryAnalysis, analyze_query
from agents.reasoning_agent import CounterfactualReasoning, reason_about_counterfactual
from agents.retrieval_engine import RetrievalContext, retrieve_context
from config import settings

logger = logging.getLogger(__name__)

# Optional progress sink, set by ``run`` for the duration of one invocation.
# A process-global (rather than a thread-local / ContextVar) so it fires no matter
# which thread LangGraph executes a node on; the single Streamlit worker is the only
# caller that sets it. ``run`` always resets it to None in a ``finally``. Signature:
# ``cb(node_name: str, elapsed_seconds: float, errored: bool) -> None``.
_progress_cb = None


class HistoriosState(TypedDict, total=False):
    """Shared graph state threaded between the five nodes.

    ``total=False`` so each node may write only the keys it owns; absent keys are
    simply not yet populated. The agent outputs (``analysis`` … ``scored``) are the
    Pydantic models from each agent. ``error`` / ``failed_node`` are set by the
    first node that fails; ``status`` is derived in ``run`` after the graph returns.
    """

    question: str
    analysis: QueryAnalysis | None
    context: RetrievalContext | None
    grounded: GroundedContext | None
    reasoning: CounterfactualReasoning | None
    scored: ScoredReasoning | None
    timings: dict[str, float]  # node name -> seconds (accumulated, sequential)
    error: str | None  # "<ExcType>: <msg>" recorded on the first node failure
    failed_node: str | None
    status: str  # "ok" | "no_context" | "error" — set by run()


# --- Node plumbing -----------------------------------------------------------


def _run_node(
    name: str, state: HistoriosState, fn: Callable[[HistoriosState], dict]
) -> dict:
    """Run one node's work with timing + graceful error capture.

    ``fn`` performs the actual agent call and returns the partial state update
    (e.g. ``{"analysis": ...}``). This wrapper times it, accumulates the elapsed
    seconds into ``timings``, and converts any exception into a recorded error so
    the pipeline halts cleanly rather than crashing. Returns the merged update.
    """
    # If an earlier node already failed we shouldn't be here (the router sends
    # errors to END), but guard anyway so a stray edge can't trigger more work.
    if state.get("error"):
        return {}

    timings = dict(state.get("timings", {}))
    t0 = time.monotonic()
    try:
        update = fn(state)
        elapsed = time.monotonic() - t0
        timings[name] = elapsed
        logger.info("node %-16s ok in %6.2fs", name, elapsed)
        _emit_progress(name, elapsed, errored=False)
        return {**update, "timings": timings}
    except Exception as exc:  # noqa: BLE001 — record-and-halt is the policy
        elapsed = time.monotonic() - t0
        timings[name] = elapsed
        logger.exception("node %-16s FAILED after %6.2fs", name, elapsed)
        _emit_progress(name, elapsed, errored=True)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "failed_node": name,
            "timings": timings,
        }


def _emit_progress(name: str, elapsed: float, *, errored: bool) -> None:
    """Notify the optional progress sink that a node finished. Never raises.

    A UI progress callback must not be able to break the pipeline, so any error
    from it is swallowed (and logged at debug level).
    """
    if _progress_cb is None:
        return
    try:
        _progress_cb(name, elapsed, errored)
    except Exception:  # noqa: BLE001 — a misbehaving UI hook must not crash a run
        logger.debug("progress callback raised; ignoring", exc_info=True)


def _understand_query(state: HistoriosState) -> dict:
    return _run_node(
        "understand_query", state, lambda s: {"analysis": analyze_query(s["question"])}
    )


def _retrieve(state: HistoriosState) -> dict:
    return _run_node(
        "retrieve", state, lambda s: {"context": retrieve_context(s["analysis"])}
    )


def _ground(state: HistoriosState) -> dict:
    return _run_node(
        "ground",
        state,
        lambda s: {"grounded": ground_context(s["context"], s["question"])},
    )


def _reason(state: HistoriosState) -> dict:
    return _run_node(
        "reason",
        state,
        lambda s: {
            "reasoning": reason_about_counterfactual(s["analysis"], s["grounded"])
        },
    )


def _score(state: HistoriosState) -> dict:
    return _run_node(
        "score",
        state,
        lambda s: {"scored": score_reasoning(s["reasoning"], s["grounded"])},
    )


def _route(state: HistoriosState) -> str:
    """Conditional-edge router shared by every transition.

    Returns the branch key: ``"error"`` halts the run (any node failed),
    ``"no_context"`` halts after retrieval when nothing was found (Rules #5/#6),
    otherwise ``"continue"`` proceeds to the next node.
    """
    if state.get("error"):
        return "error"
    context = state.get("context")
    if context is not None and not context.primary_chunks and not context.analogy_chunks:
        return "no_context"
    return "continue"


# --- Graph build / run -------------------------------------------------------

_app = None  # compiled graph singleton (compiling once is enough)


def build_graph():
    """Build and compile the LangGraph ``StateGraph``. Cached after first call."""
    global _app
    if _app is not None:
        return _app

    builder = StateGraph(HistoriosState)
    builder.add_node("understand_query", _understand_query)
    builder.add_node("retrieve", _retrieve)
    builder.add_node("ground", _ground)
    builder.add_node("reason", _reason)
    builder.add_node("score", _score)

    builder.set_entry_point("understand_query")

    # Sequential flow, but every transition is conditional so a recorded error (or
    # an empty retrieval) can short-circuit straight to END instead of crashing.
    builder.add_conditional_edges(
        "understand_query", _route, {"continue": "retrieve", "error": END}
    )
    builder.add_conditional_edges(
        "retrieve", _route, {"continue": "ground", "no_context": END, "error": END}
    )
    builder.add_conditional_edges(
        "ground", _route, {"continue": "reason", "error": END}
    )
    builder.add_conditional_edges(
        "reason", _route, {"continue": "score", "error": END}
    )
    builder.add_edge("score", END)

    _app = builder.compile()
    return _app


def _derive_status(state: HistoriosState) -> str:
    """Map the final graph state to a coarse status label for callers/reports."""
    if state.get("error"):
        return "error"
    context = state.get("context")
    if context is not None and not context.primary_chunks and not context.analogy_chunks:
        return "no_context"
    return "ok"


def run(question: str, progress_callback=None) -> HistoriosState:
    """Execute the full pipeline for one counterfactual question.

    Validates configuration, runs the compiled graph, and returns the final
    ``HistoriosState`` (whatever stage it reached). This function NEVER raises:
    configuration problems, node failures, and unexpected graph errors are all
    captured onto ``state["error"]`` with ``status="error"`` so the frontend can
    render a graceful message (Critical Rules #5/#6).

    Args:
        question: The raw natural-language "what if" question.
        progress_callback: Optional ``cb(node_name, elapsed_seconds, errored)``
            invoked as each node finishes — used by the frontend to light up
            pipeline stages live. It is installed only for the duration of this
            call and any exception it raises is swallowed (it can never break the
            run). Default ``None`` ⇒ no behavioural change for other callers.

    Returns:
        The final ``HistoriosState``. On success ``status == "ok"`` and ``scored``
        holds a ``ScoredReasoning``; on an empty corpus ``status == "no_context"``;
        on any failure ``status == "error"`` and ``error`` describes it.
    """
    global _progress_cb

    initial: HistoriosState = {
        "question": (question or "").strip(),
        "timings": {},
        "error": None,
        "failed_node": None,
    }

    if not initial["question"]:
        return {**initial, "status": "error", "error": "ValueError: question is empty"}

    t0 = time.monotonic()
    _progress_cb = progress_callback
    try:
        settings.validate()  # loud-but-caught: missing key -> graceful error state
        app = build_graph()
        final: HistoriosState = app.invoke(initial)
    except Exception as exc:  # noqa: BLE001 — run() must never crash its caller
        logger.exception("pipeline run failed before/around graph invocation")
        elapsed = time.monotonic() - t0
        return {
            **initial,
            "timings": {**initial.get("timings", {}), "_total": elapsed},
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _progress_cb = None

    elapsed = time.monotonic() - t0
    final["timings"] = {**final.get("timings", {}), "_total": elapsed}
    final["status"] = _derive_status(final)

    logger.info(
        "pipeline run: status=%s in %.2fs (nodes: %s)%s",
        final["status"],
        elapsed,
        ", ".join(f"{k}={v:.2f}s" for k, v in final["timings"].items() if k != "_total"),
        f" — FAILED at {final.get('failed_node')}: {final.get('error')}"
        if final["status"] == "error"
        else "",
    )
    return final


if __name__ == "__main__":
    # Live end-to-end smoke test — needs CEREBRAS_API_KEY and a populated ChromaDB.
    # Roughly 3-4 LLM calls (1 query understanding + <=2 grounding + 1 reasoning).
    # Run from the project root:
    #   D:\historyos\venv\Scripts\python.exe -m pipeline.historios_pipeline
    import sys

    # Windows console defaults to cp1252; reasoning/claim text carries non-cp1252
    # chars (em-dashes, macrons) — see CLAUDE.md "WINDOWS CONSOLE ENCODING".
    sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    question = "What if the Mughal Empire had industrialized before the British arrived?"
    print(f"\nQuestion: {question}\n")

    state = run(question)

    print(f"\nSTATUS: {state.get('status')}")
    if state.get("error"):
        print(f"ERROR (at {state.get('failed_node')}): {state['error']}")

    print("\nPER-NODE TIMINGS:")
    for name, seconds in state.get("timings", {}).items():
        print(f"  {name:<16} {seconds:6.2f}s")

    grounded = state.get("grounded")
    scored = state.get("scored")
    if grounded is not None:
        print(
            f"\nGrounded: verified={len(grounded.verified_facts)} "
            f"debated={len(grounded.debated_points)} "
            f"background={len(grounded.background_context)} "
            f"analogies={len(grounded.analogies)}"
        )
    if scored is not None:
        print(
            f"Scored: {len(scored.steps)} step(s) | "
            f"distribution={scored.confidence_distribution} | "
            f"overall={scored.overall_confidence}"
        )
        for s in scored.steps:
            print(f"  Step {s.step_number} [{s.confidence_level}] — {s.consequence[:80]}")
