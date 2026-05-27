"""HistoryOS pipeline — agent orchestration.

Wires the five agents together into a single sequential LangGraph
and exposes the entry point that the frontend and evaluator call.

Responsibilities:
- Define the shared graph state passed between nodes (the user query,
  structured query, retrieved chunks, verified facts, causal chain,
  confidence scores).
- Register the five agents as nodes in order:
    query_understanding -> retrieval_engine -> grounding_layer
    -> reasoning_agent -> confidence_scorer
  and connect them with sequential edges.
- Compile the graph and provide ``run(question) -> result`` that
  executes the full pipeline and returns the final structured state.
- Surface graceful failures (empty vector store, no grounded facts)
  rather than fabricating output (Critical Rules #5, #6).
"""
