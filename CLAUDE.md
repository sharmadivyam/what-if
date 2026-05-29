# WHAT IF? — Claude Code Instructions

> Product name: **WHAT IF?** (the user-facing brand, shown in the Streamlit UI).
> Internal package/folder names stay `historios` / `agents` / `pipeline` / `core` —
> renaming directories would break every import. "HistoryOS" persists only as the
> internal codename below.

## What This Project Is
A counterfactual historical reasoning engine.
Users ask "what if" questions about history.
System retrieves verified historical context via RAG
and simulates consequences using multi-agent reasoning.
Output explicitly separates VERIFIED facts from
SIMULATED consequences with confidence scores.

## Architecture Overview
- 5 agents connected sequentially via LangGraph
- ChromaDB for local vector storage
- Cerebras qwen-3-235b-a22b-instruct-2507 for all LLM calls (free, OpenAI-compatible)
- sentence-transformers all-mpnet-base-v2 for embeddings (local, free, no API key)
- Streamlit for frontend UI
- Dynamic Wikipedia fallback in the retrieval engine: when local Chroma search has
  no genuinely relevant primary context, it fetches Wikipedia live, embeds it, and
  re-searches (no LLM call). Tuned by four settings (defaults in `config.py`, mirror
  in `.env` / `.env.example` per the CONFIG GOTCHA):
  - `ENABLE_DYNAMIC_RETRIEVAL` (bool, default `True`) — master on/off switch.
  - `DYNAMIC_SEARCH_LIMIT` (int, default `5`) — Wikipedia pages fetched per run.
  - `DYNAMIC_CHUNK_CAP` (int, default `200`) — max chunks added per run (bounds latency).
  - `DYNAMIC_MIN_SIMILARITY` (float, default `0.6`) — cosine floor below which the best
    local primary hit is treated as "no real match" (the fallback trigger).
- Total API cost: ₹0

## Critical Rules — Never Violate These
1. Never mix verified facts with simulated content
2. Every fact must cite its source chunk ID
3. Max 4 causal reasoning steps (hallucination guard)
4. All agents must return Pydantic models, not raw strings
5. Always handle ChromaDB empty state gracefully
6. Never call LLM without retrieved context (no raw GPT)
7. All LLM calls use get_llm_client(), all embeddings use
   get_embedding_function() from core/llm_client.py — never instantiate
   provider clients directly in agent files

## Folder Structure
historios/
├── data/
│   ├── raw/              # downloaded wikipedia text
│   └── processed/        # chunked text ready for embedding
├── ingestion/
│   ├── topics.py             # CORPUS_TOPICS — the canonical ingestion list
│   ├── wikipedia_loader.py
│   ├── chunker.py
│   └── embedder.py
├── vectorstore/
│   └── chroma_client.py
├── core/
│   └── llm_client.py     # provider clients: Cerebras LLM + local embeddings
├── agents/
│   ├── query_understanding.py
│   ├── retrieval_engine.py
│   ├── grounding_layer.py
│   ├── reasoning_agent.py
│   └── confidence_scorer.py
├── pipeline/
│   └── historios_pipeline.py
├── output/
│   └── report_generator.py
├── frontend/
│   └── app.py
├── evaluation/
│   ├── test_cases.json
│   └── evaluator.py
├── .claude/
│   └── commands/         # custom slash commands
├── config.py
├── requirements.txt
├── .env
└── CLAUDE.md

## Tech Stack
- Python 3.11
- LangGraph (agent orchestration)
- LangChain (LLM + retrieval utilities)
- ChromaDB (local vector database)
- Cerebras qwen-3-235b-a22b-instruct-2507 (LLM — free, OpenAI-compatible API)
- sentence-transformers all-mpnet-base-v2 (embeddings — local, free, no API key)
- Wikipedia Python API (data source)
- Tavily API (web search)
- Pydantic v2 (structured outputs)
- Streamlit (frontend)

## Current Build Phase
Phase 0 — Setup complete. Starting Phase 1: Data Ingestion.

## Completed Components
[x] Folder structure
[x] requirements.txt
[x] config.py
[x] wikipedia_loader.py
[x] chunker.py
[x] embedder.py
[x] chroma_client.py
[ ] query_understanding.py
[x] retrieval_engine.py  (incl. dynamic Wikipedia fallback)
[x] grounding_layer.py
[x] reasoning_agent.py
[x] confidence_scorer.py
[x] historios_pipeline.py
[x] report_generator.py
[x] app.py
[ ] evaluation/test_cases.json
[ ] evaluation/evaluator.py

## Known Issues Log
- Runtime Python is 3.12.10, not 3.11 as documented above. Non-blocking so far;
  update the Tech Stack note (or pin 3.11) once confirmed.
- `requirements.txt` declares `wikipedia-api` (import `wikipediaapi`), NOT the
  older `wikipedia` package. The two have incompatible APIs and `wikipedia-api`
  raises no `DisambiguationError`. `wikipedia_loader.py` detects disambiguation
  pages by inspecting page categories instead. Keep this in mind for any future
  Wikipedia code.
- Installed `wikipedia-api` is v0.15.0 — a major rewrite that uses `httpx` (not
  `requests`) and has built-in retry (`max_retries` / `retry_wait`, exponential
  backoff) raising a single `wikipediaapi.WikipediaException` base type. The
  loader relies on that retry and catches `WikipediaException` to skip.
- ENV GOTCHA: the Bash/PowerShell tools do NOT inherit the venv activated in an
  interactive shell. Bare `pip` resolves to the global Programs Python and bare
  `python` to the Windows Store Python — two different interpreters, neither the
  venv. Always invoke the venv explicitly:
  `D:\historyos\venv\Scripts\python.exe -m pip ...` and `... -m ingestion.xxx`.
  (Installed in the venv so far: `wikipedia-api`, `langchain-core`, `tiktoken`,
  `openai`, `torch`, `chromadb`, `sentence-transformers`. Remaining
  `requirements.txt` entries still need installing as later phases need them.)
- CONFIG GOTCHA: values in `.env` SHADOW the defaults in `config.py` (config reads
  the environment first, via `_get`). When changing a setting's default in
  `config.py`, also update `.env` AND `.env.example` — otherwise the stale `.env`
  value silently wins at runtime. This bit the chunker twice (`CHUNK_SIZE`, then
  `EMBEDDING_MODEL`).
- CEREBRAS MODELS: the available model list on this account CHANGES over time —
  always confirm with `client.models.list()`. `llama-3.3-70b` was 404, then
  `qwen-3-235b-a22b-instruct-2507` also went 404 (2026-05-27). Currently only
  `gpt-oss-120b` and `zai-glm-4.7` are available; we run **`gpt-oss-120b`** (set in
  `.env`, `.env.example`, and the `config.py` default — keep all three in sync per
  the CONFIG GOTCHA). If you hit a model 404, re-list and update those three.
- CEREBRAS FREE-TIER RATE LIMIT: the free tier hard-throttles bursts with HTTP 429
  `queue_exceeded` ("high traffic") and ~60s `Retry-After` waits — it canNOT sustain
  ~10+ sequential calls per pipeline run. Two mitigations are in place: (1) the
  shared client uses `max_retries=6` (core/llm_client.py) to ride out transient
  429s; (2) the grounding layer BATCHES per pool — one LLM call for all primary
  chunks + one for all analogy chunks (2 calls/run), not one call per chunk. Keep
  per-run LLM call counts low; prefer batching over per-item loops.
- OpenRouter fallback active — triggers on Cerebras 429/quota errors automatically.
  Agents call `core.llm_client.call_with_fallback(...)` (not `get_llm_client()` /
  `chat.completions.create` directly), which catches `openai.RateLimitError` from
  the Cerebras call and retries on OpenRouter with `settings.OPENROUTER_MODEL`. A
  WARNING is logged on fallback ("Cerebras quota exceeded — falling back to
  OpenRouter"). Requires `OPENROUTER_API_KEY` in `.env`.
- WINDOWS CONSOLE ENCODING: Python stdout defaults to cp1252 here, so PRINTING
  article text containing non-cp1252 chars (e.g. `ā` U+0101, en-dash) raises
  `UnicodeEncodeError` and aborts the script — not just garbled display. Any script
  that prints chunk/article text must set `PYTHONIOENCODING=utf-8` or call
  `sys.stdout.reconfigure(encoding="utf-8")`. The stored data itself is valid UTF-8.
- DYNAMIC FALLBACK TRIGGER — "empty pool" is NOT a usable signal. A populated
  ChromaDB collection ALWAYS returns top-k results regardless of relevance, so for
  an out-of-corpus question the primary pool comes back FULL of low-similarity junk,
  never empty. An empty-pool check therefore only ever fires on a totally empty
  collection — useless for the actual goal. The fallback instead triggers on a
  RELEVANCE FLOOR: `DYNAMIC_MIN_SIMILARITY` (default 0.6), the cosine similarity
  (`1 - distance`) below which the best local primary hit is treated as no real
  match (an empty pool trivially clears that bar too). 0.6 was calibrated against
  measured scores on the current 18-topic corpus: in-corpus top hits land ~0.72–0.84,
  out-of-corpus ~0.36–0.53, leaving a clean gap. THIS THRESHOLD IS CORPUS-DEPENDENT —
  revisit it as the corpus grows (more/denser topics can push out-of-corpus scores
  up, or shift the in-corpus floor), e.g. by re-running the in- vs out-of-corpus
  similarity comparison and moving the floor back into the gap.