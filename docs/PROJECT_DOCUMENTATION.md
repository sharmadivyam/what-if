# WHAT IF? — Project Documentation

> Counterfactual history engine that answers "what if…" questions with **grounded,
> cited, confidence-scored** reasoning — keeping VERIFIED facts strictly separate
> from SIMULATED consequences.
>
> **Product name:** WHAT IF?  ·  **Internal codename / package names:** `historios`,
> `agents`, `pipeline`, `core` (folders keep the original name; renaming would break
> imports).

---

## 1. Overview — what problem it solves

Ask a large language model "What if the Mughal Empire had industrialized before the
British arrived?" and it will happily produce fluent prose that blends **real history**,
**plausible inference**, and **confident hallucination** into one indistinguishable
paragraph — with no sources and no signal about which parts to trust.

**WHAT IF?** is built to make that distinction explicit. For any counterfactual it:

1. **Retrieves** verified historical context from a local, cited corpus (RAG).
2. **Grounds** it — extracts factual claims and classifies each as VERIFIED / DEBATED /
   BACKGROUND, every claim tagged with the exact source chunk it came from.
3. **Reasons** a short causal chain of what *might* have happened, where every simulated
   step is labelled `[SIMULATED]` and must cite `[EVIDENCE: chunk_id]`.
4. **Scores** each step's confidence (HIGH / MEDIUM / LOW / SPECULATIVE) purely from how
   much verified evidence backs it.
5. **Renders** a report that keeps "What We Know" (verified, cited) visually separate
   from "What Might Have Happened" (simulated, scored).

The guiding principle: *honesty about uncertainty is the product.* The system would
rather flag a claim as SPECULATIVE than present it as fact.

**Total running cost: ₹0** — free LLM tier, local embeddings, local vector DB, free
data source (see §10).

---

## 2. Architecture

Five specialized agents are wired into a single **LangGraph** `StateGraph` and run
sequentially. Each returns a **Pydantic** model (never a raw string). Two data planes:
an **offline ingestion pipeline** (run once to build the corpus) and an **online query
pipeline** (run per question).

```
                          ┌───────────────────────────────────────────────────────┐
   INGESTION (offline)    │  Wikipedia ──► wikipedia_loader ──► data/raw/*.txt      │
   run once to build      │                       │                                 │
   the vector store       │                       ▼                                 │
                          │                   chunker  ──► data/processed/*.jsonl   │
                          │                       │   (paragraph-aware, ~300 tok)   │
                          │                       ▼                                 │
                          │                   embedder ──► ChromaDB (all-mpnet,     │
                          │                                 cosine, 18 src · 1,466) │
                          └───────────────────────────────────────────────────────┘

   QUERY (online, per question) — pipeline/historios_pipeline.py (LangGraph)

      question:str
           │
           ▼
   ┌──────────────────┐   error ─┐
   │ understand_query │──────────┤   Agent 1  analyze_query()        LLM ×1 (T=0.1, JSON)
   │  → QueryAnalysis │          │
   └────────┬─────────┘          │
            ▼                    │
   ┌──────────────────┐   error ─┤
   │     retrieve     │──────────┤   Agent 2  retrieve_context()     no LLM (vector +
   │ → RetrievalCtx   │          │              (+ dynamic fallback)   local embeddings)
   └────────┬─────────┘          │
      ok│   └ no_context ────────┤   (empty corpus / no hits → END, Rule #5)
        ▼                        │
   ┌──────────────────┐   error ─┤
   │      ground      │──────────┤   Agent 3  ground_context()       LLM ×2 (T=0.0, JSON,
   │ → GroundedCtx    │          │                                    one call per pool)
   └────────┬─────────┘          │
            ▼                    │
   ┌──────────────────┐   error ─┤
   │      reason      │──────────┤   Agent 4  reason_about_…()       LLM ×1 (T=0.3, prose)
   │ → Counterfactual │          │
   └────────┬─────────┘          │
            ▼                    │
   ┌──────────────────┐   error ─┤
   │      score       │──────────┤   Agent 5  score_reasoning()      no LLM (pure logic)
   │ → ScoredReasoning│          │
   └────────┬─────────┘          │
            ▼                    ▼
          (END) ◄───────────── (END)        ──►  output/report_generator ──► frontend/app.py
                                                  (HistoriosReport: markdown + struct)   (Streamlit UI)
```

**Inside the `retrieve` node — dynamic Wikipedia fallback** (graph topology unchanged):

```
   retrieve_context(analysis)
        │
        ▼
   local Chroma search (search + analogy queries)
        │
        ▼
   best primary similarity ≥ DYNAMIC_MIN_SIMILARITY (0.6) ? ──yes──► use local chunks ─► continue
        │ no  (out-of-corpus: store always returns top-k junk, never empty)
        ▼
   _dynamic_retrieve():  Wikipedia search (top DYNAMIC_SEARCH_LIMIT pages)
        ─► fetch (reuses wikipedia_loader: disambiguation-safe, cached, skips failures)
        ─► chunk (chunker)  ─► save raw + processed files
        ─► embed + upsert into ChromaDB (tagged source="wikipedia_dynamic"; local embeddings)
        ─► RE-RUN the same vector search  ─► normal SearchResults ─► continue
        │
        └─ nothing usable found → return [] → empty pools → no_context → END (Rule #5)
```

- **Edges** are conditional via one shared `_route`: a node that fails records the error
  and the run halts gracefully at `END` (never crashes). Empty retrieval short-circuits
  to `END`.
- **Why a similarity floor, not an empty check:** a populated ChromaDB collection always
  returns top-k results regardless of relevance, so an out-of-corpus question yields a
  *full* primary pool of low-similarity junk, never an empty one. The fallback therefore
  triggers when the best primary hit scores below `DYNAMIC_MIN_SIMILARITY` (0.6 —
  calibrated: in-corpus tops ~0.72–0.84, out-of-corpus ~0.36–0.53; revisit as the corpus
  grows). The fallback makes **zero LLM calls** (vector search + local embeddings only),
  so Critical Rule #6 stays trivially satisfied.
- **LLM-call budget per question:** ~4 (1 understand + 2 ground + 1 reason). Agents 2
  and 5 make **zero** LLM calls — including Agent 2's dynamic Wikipedia fallback.

---

## 3. Every file and what it does

### Configuration & shared clients
| File | Responsibility |
|------|----------------|
| `config.py` | `Settings` singleton loaded from `.env` (via `_get`/`_get_int`/`_get_float`; **env shadows defaults**). Holds API keys, model IDs, ChunK/retrieval/guardrail constants, paths. `validate()` fails loudly if `CEREBRAS_API_KEY` is missing. |
| `core/llm_client.py` | The **only** place provider clients are created (Critical Rule #7). `get_llm_client(provider)` returns a cached OpenAI-compatible client (`max_retries=6` to ride out free-tier 429s). `call_with_fallback()` runs on Cerebras and, on a `RateLimitError`, automatically retries on **OpenRouter**. `get_embedding_function()` returns the cached local sentence-transformers embedder (no API key). |

### Ingestion (offline, `ingestion/`)
| File | Responsibility |
|------|----------------|
| `topics.py` | `CORPUS_TOPICS` — the canonical list of 18 Wikipedia articles that make up the corpus. Single source of truth; add a topic and re-run ingestion to grow the store. |
| `wikipedia_loader.py` | Downloads article text via **`wikipedia-api`** into `data/raw/`. Caches downloads, falls back to full-text search on an exact-title miss (`resolved_via_search`), detects disambiguation pages via categories, and **skips** (never crashes) on per-topic failures after the library's own retries. |
| `chunker.py` | Cleans articles (drops "References"/"See also" boilerplate), splits **paragraph-aware** into ≤`CHUNK_SIZE` (300) token chunks with `CHUNK_OVERLAP` (100) of whole-paragraph overlap. Token sizing via `tiktoken` `cl100k_base` (heuristic). Assigns a **stable `chunk_id`** = `<slug>_<index>` (the citation handle, Rule #2). Persists to `data/processed/*.jsonl`. |
| `embedder.py` | Reads processed chunks and upserts them into ChromaDB in batches of 50, skipping already-embedded ids. Embedding is done by Chroma's local embedding function (Rule #7) — never a paid API. |

### Vector store (`vectorstore/`)
| File | Responsibility |
|------|----------------|
| `chroma_client.py` | Owns the persistent ChromaDB client + the `historios` collection (**cosine** distance, local embedding function bound on both write and read paths). `store()` upserts (idempotent) and accepts an optional **`extra_metadata`** dict merged into every chunk's metadata — used by the dynamic fallback to tag fetches with `source="wikipedia_dynamic"` (curated ingestion passes nothing, so its metadata is unchanged). `search()` returns `SearchResult`s (`similarity_score = 1 − cosine_distance`) and **returns `[]` on an empty collection** (Rule #5). **`get_metadata(chunk_ids)`** returns `{chunk_id: metadata}` for provenance lookups without a similarity search. `get_collection_stats()` reports totals + per-source counts. |

### Agents (`agents/`)
| File | Agent | Responsibility |
|------|-------|----------------|
| `query_understanding.py` | **1** | `analyze_query()` → `QueryAnalysis` (time period, geography, actors, type, proposed change, search + analogy queries). JSON mode, T=0.1, one corrective retry. (Runs before retrieval — the only legitimate Rule #6 exemption: it parses the question, asserts no facts.) |
| `retrieval_engine.py` | **2** | `retrieve_context()` → `RetrievalContext`. **No LLM.** Runs each search/analogy query against ChromaDB, dedups by `chunk_id` (keeping best cosine score), keeps top 8 primary + top 3 analogy chunks (disjoint). **Dynamic Wikipedia fallback:** `_local_primary_is_weak()` flags a pool with no hit ≥ `DYNAMIC_MIN_SIMILARITY` (empty or all-junk); when weak, `_dynamic_retrieve()` fires — `_candidate_titles()` gathers up to `DYNAMIC_SEARCH_LIMIT` page titles via `wiki.search` over the search+analogy queries, `wikipedia_loader.load_topics` fetches them (disambiguation-safe, cached), `chunker` chunks them (capped at `DYNAMIC_CHUNK_CAP`, processed files saved), `chroma_client.store(..., extra_metadata={"source": "wikipedia_dynamic"})` embeds + upserts them, then it **re-runs the same `_run_queries` + `_dedup_and_rank`** so results come back as ordinary `SearchResult`s. Zero LLM calls; returns `[]` on no usable result, leaving the `no_context` route to handle it. (Tavily seam retired in favour of this.) |
| `grounding_layer.py` | **3** | `ground_context()` → `GroundedContext`. **Batched: one LLM call per pool** (primary + analogy = 2 calls). Extracts claims ONLY from chunk text, classifies VERIFIED / DEBATED / BACKGROUND, validates each cited `chunk_id` against the retrieved set (drops unknowns), and re-attaches the **trusted** citation (title/url) from the matched `SearchResult` — so citations can't be fabricated. T=0.0, JSON mode + one corrective retry. |
| `reasoning_agent.py` | **4** | `reason_about_counterfactual()` → `CounterfactualReasoning`. **One LLM call**, T=0.3, structured **prose** (not JSON) with `[SIMULATED]` / `[EVIDENCE: id]` markers, ≤4 steps, plus tail sections (Unknowable / Reconnection / Historian's Note). Regex-parsed; ungrounded steps flagged (`is_grounded=False`), cited-but-unknown ids recorded (`unknown_evidence_ids`). Empty-context ⇒ no LLM call (Rule #6). |
| `confidence_scorer.py` | **5** | `score_reasoning()` → `ScoredReasoning`. **No LLM — pure logic.** Scores each step by evidence count: HIGH (≥2 verified facts), MEDIUM (1 verified fact OR an analogy), LOW (only debated/background), SPECULATIVE (ungrounded or only fabricated citations). Adds per-step `confidence_level` + `confidence_explanation`, a `confidence_distribution`, and `overall_confidence` (weakest-link). |

### Orchestration, output, UI, evaluation
| File | Responsibility |
|------|----------------|
| `pipeline/historios_pipeline.py` | Wires Agents 1→5 into a LangGraph `StateGraph` over the `HistoriosState` TypedDict. `_run_node` times each node and **captures any exception** (records `error`/`failed_node`, never re-raises). `run(question, progress_callback=None)` validates config, invokes the graph, derives `status` (`ok`/`no_context`/`error`), and **never raises**. The optional `progress_callback` fires as each node finishes (used by the UI). |
| `output/report_generator.py` | `generate_report(scored, grounded, …)` → `HistoriosReport` (structured fields **+** display-ready markdown), plus `report_from_state(state)`. Enforces the VERIFIED-vs-SIMULATED split in presentation; renders honest notices for error / empty states. **Provenance:** `_used_dynamic_sources(grounded)` looks up the grounded chunk_ids' Chroma metadata (via `chroma_client.get_metadata`); if any is tagged `wikipedia_dynamic`, the report sets the structured field **`augmented_with_dynamic=True`** and renders a "🌐 *Augmented with a live Wikipedia fetch*" note under the banner. The lookup is best-effort (any failure is swallowed → treated as curated), so it never breaks report generation. No LLM, no network beyond the local Chroma read. |
| `frontend/app.py` | The **WHAT IF?** Streamlit UI (museum/editorial aesthetic; dark default + light toggle; battle-painting page background behind a centred "paper" panel). Landing → 5-stage loading → simulation-first results timeline with confidence-coloured cards + collapsed evidence. The pipeline runs in a **session-state background job polled by an `st.fragment`**, so a theme toggle / rerun never discards an in-flight question. |
| `evaluation/evaluator.py` | Runs the full A1→A5 chain over `test_cases.json` and applies four spot-checks (C1–C4, see §7), printing each case + a pass/fail matrix with an OVERALL-confidence column. Records-and-continues on per-case failure; never crashes. |
| `evaluation/test_cases.json` | 8 curated counterfactual questions targeting the ingested corpus. |
| `.streamlit/config.toml` | `[logger] level = "info"` so pipeline logs are visible (noisy transformers/torch warnings are suppressed in `app.py`). |

---

## 4. Data-flow walkthrough (worked example)

**Question:** *"What if the Mughal Empire had industrialized before the British arrived?"*

1. **Agent 1 — analyze_query** → `QueryAnalysis(time_period="1526-1857",
   geography="South Asia", counterfactual_type="economic",
   search_queries=["Mughal Empire economic structure 16th century", …],
   analogy_queries=["Meiji Restoration Japan industrialization", …])`. *(1 LLM call)*
2. **Agent 2 — retrieve_context** runs each query against ChromaDB → **8 primary +
   3 analogy** chunks (deduped by cosine score). *(no LLM)*
3. **Agent 3 — ground_context** sends the two pools to the LLM (2 calls) → e.g.
   **16 verified facts**, 2 debated, **17 analogies**, each tagged with a validated
   `chunk_id` and a trusted citation. *(2 LLM calls)*
4. **Agent 4 — reason_about_counterfactual** produces 4 `[SIMULATED]` steps, each citing
   `[EVIDENCE: mughal_empire_22]`-style ids, plus Unknowable / Reconnection / Historian's
   Note. *(1 LLM call)*
5. **Agent 5 — score_reasoning** scores the steps from the evidence: e.g. **HIGH×2,
   MEDIUM×2**, `overall_confidence = MEDIUM`. *(no LLM)*
6. **report_generator** builds the `HistoriosReport`; **frontend** renders verified facts
   ("What We Know") above the simulated, colour-coded timeline.

End-to-end wall time on the free tier in a real run: **~152 s** (dominated by Cerebras
rate-limit back-off, not compute).

---

## 5. Tech stack & why

| Choice | Why |
|--------|-----|
| **Python 3.12** (docs say 3.11; runtime 3.12.10) | Ecosystem for LangGraph / LangChain / Chroma / sentence-transformers. |
| **LangGraph** (`StateGraph`) | Deterministic, inspectable agent orchestration with a typed shared state and conditional edges — better than ad-hoc function calls for graceful failure + future branching. |
| **ChromaDB** (local, persistent) | Zero-cost, embedded vector DB; no server, no API key; cosine search; trivial idempotent upsert. |
| **Cerebras `gpt-oss-120b`** (OpenAI-compatible) | Free, fast, OpenAI-compatible API → all LLM calls go through the standard `openai` client. (Model list changes; confirm via `client.models.list()`.) |
| **OpenRouter** (fallback) | Free-tier provider used automatically when Cerebras hits its daily quota. |
| **sentence-transformers `all-mpnet-base-v2`** (local) | Strong general-purpose embeddings, **no API key**, runs on `torch` locally → ₹0 embeddings. 384-token window drives the chunk size. |
| **Pydantic v2** | Every agent returns a validated model (Critical Rule #4); JSON-mode outputs are validated + self-heal via one corrective retry. |
| **Streamlit** | Fast, pure-Python UI; custom CSS gives the editorial look without a JS build. |
| **`wikipedia-api`** + **`tiktoken`** | Free, license-clean data source; tiktoken for model-agnostic chunk sizing. |

---

## 6. Key design decisions & rationale

- **Batched grounding (2 calls/run), not per-chunk.** An earlier per-chunk design made
  ~11 LLM calls/run; the free Cerebras tier hard-throttles bursts with HTTP 429
  `queue_exceeded` + ~60 s waits, so a single run took 10+ minutes and still failed.
  Batching one call per pool keeps runs viable. *Trade-off:* intra-batch
  misattribution is possible (a real claim tagged with a different but still-valid
  `chunk_id`); only **unknown** ids are detectable and dropped.
- **Per-agent temperatures (0.1 / 0.0 / 0.3).** Query understanding = 0.1 (stable
  parsing); grounding = 0.0 (deterministic extraction, no invention); reasoning = 0.3
  (slightly creative causal chains — 0.0 is too rigid, >0.5 starts inventing).
- **Chunk size 300 tokens / 100 overlap.** The embedding model's window is 384 tokens;
  `tiktoken cl100k` (used for sizing) counts differently from the model's own tokenizer,
  so 300 leaves a safe buffer under 384. Overlap preserves cross-paragraph context.
- **Grounding layer trusts only validated citations.** The model tags a `source_chunk_id`,
  but the layer (a) validates it against the actual retrieved pool and drops unknowns,
  and (b) attaches the title/URL **from the matched `SearchResult`**, never from the
  model — so citations can't be fabricated (Rule #2).
- **Reasoning is structured *prose*, not JSON.** The `[SIMULATED]` / `[EVIDENCE]` /
  numbered-step format *is* the alignment mechanism; JSON mode would erase that
  reinforcement. The raw response is kept so the report can render even if the regex
  parser misses an edge case.
- **Max 4 causal steps (hallucination guard, Rule #3).** Depth over breadth; extra
  `Step N` blocks are dropped with a warning. The longer the chain, the more speculative.
- **Confidence is computed, not asked.** Agent 5 ignores the model's self-reported
  confidence and recomputes it from evidence counts — and surfaces the model's
  self-report only for contrast.
- **Never call the LLM without context (Rule #6).** Empty retrieval / empty grounding
  short-circuits without an LLM call and returns an honest "no verified context" result.
- **Rerun-safe UI job.** The Streamlit run executes in a worker thread tracked in
  `session_state` and polled by an `st.fragment`, so theme toggles / reruns never discard
  an in-flight ~2-minute question.

---

## 7. Evaluation methodology & results

`evaluation/evaluator.py` runs all 8 cases through the **full A1→A5 chain** (Agent 5
adds no LLM call) and applies four rule-checks per case:

| Check | What it verifies (Critical Rule) |
|-------|----------------------------------|
| **C1** | Every reasoning step is labelled `[SIMULATED]` (Rule #1). |
| **C2** | Every step cites a `chunk_id` that exists in the grounded context (Rule #2). |
| **C3** | No more than `MAX_CAUSAL_STEPS` (4) steps anywhere (Rule #3). |
| **C4** | No simulation presented as fact — steps labelled + tail sections hedge (Rule #4). |

**Latest full-run scores** (`evaluation/eval_run_v4.log`, 8 cases):

| Check | Score |
|-------|-------|
| C1 — all `[SIMULATED]` | **7 / 8** |
| C2 — cites real source | **4 / 8** ← weakest |
| C3 — ≤ 4 steps | **8 / 8** ✅ |
| C4 — no sim-as-fact | **7 / 8** |

Per-case: `mughal_industrialization`, `rome_resists`, `louis_xvi_survives`, `no_genghis`
passed all four; the C2 failures were ungrounded steps (`no_british_raj`,
`cuban_missile_war`), fabricated `chunk_id`s (`ottoman_modernizes`), and a parse miss
that yielded 0 steps (`no_ww2_invasion`, which then fails C1/C2/C4). C3 held everywhere.
The evaluator now also reports each case's Agent-5 `overall_confidence`.

---

## 8. Known limitations

- **Free-tier latency.** Cerebras 429 back-off makes a full run take ~1–3 minutes; the
  evaluator's 8 cases can take tens of minutes.
- **Citation grounding (C2) is the weakest dimension** — the reasoning model sometimes
  emits ungrounded steps or fabricated `chunk_id`s. These are *flagged* (ungrounded /
  `unknown_evidence_ids` → SPECULATIVE), not silently accepted, but not yet prevented.
- **Parser fragility.** The reasoning output is regex-parsed; a badly-formatted response
  can yield 0 steps (seen once), which cascades to C1/C2/C4 failure. There is no
  corrective re-parse retry (kept to 1 LLM call to respect the rate limit).
- **Intra-batch misattribution** in grounding (accepted trade-off for batching).
- **Curated corpus is small** — 18 Wikipedia articles (1,466 chunks). Questions outside
  these topics now trigger the **dynamic Wikipedia fallback** (fetch → chunk → embed →
  re-search) rather than returning "no verified sources"; only a question that even
  Wikipedia search can't serve falls back to the honest no-context result.
- **Dynamic-fallback cost & calibration** — fetching/embedding live pages adds latency
  the *first* time a new topic is asked (subsequent asks hit the now-persisted chunks).
  The trigger is a fixed cosine floor `DYNAMIC_MIN_SIMILARITY=0.6`, calibrated for the
  current corpus (in-corpus ~0.72–0.84 vs out-of-corpus ~0.36–0.53); it is
  corpus-dependent and should be recalibrated as the corpus grows.
- **Web search fallback (Tavily) still unimplemented** — the live fallback uses Wikipedia
  only; the Tavily seam was retired in favour of it.
- **Model availability drifts** — the Cerebras free model list changes; if a model 404s,
  re-list and update `.env` / `.env.example` / `config.py` together.
- **Windows console encoding** — scripts that print article text must use UTF-8.

---

## 9. Future improvements

- Enforce citation grounding harder (constrained decoding, or a reparse/repair retry
  when a step is ungrounded) to lift the C2 score.
- Recalibrate the dynamic-fallback trigger (`DYNAMIC_MIN_SIMILARITY`) as the corpus
  grows — ideally make it adaptive (e.g. relative to the score distribution) rather than
  a fixed 0.6, so it stays in the in- vs out-of-corpus gap automatically.
- Add **Tavily web search as a secondary fallback** behind the Wikipedia fetch, for
  questions Wikipedia search itself can't serve.
- Grow and diversify the corpus; add per-source quality weighting (and weight
  dynamically-fetched `wikipedia_dynamic` chunks distinctly if needed).
- Cache pipeline results by question to make demos instant.
- Stream tokens / per-step rendering instead of staged polling.
- Add response/result caching and a proper test harness around the agents (not just the
  end-to-end evaluator).
- Add a confidence calibration study (do HIGH steps actually hold up?).

---

## 10. Build cost — ₹0

| Component | Cost |
|-----------|------|
| LLM (Cerebras `gpt-oss-120b`) | **Free** tier |
| LLM fallback (OpenRouter) | **Free** tier |
| Embeddings (`all-mpnet-base-v2`, local) | **Free** (runs on local `torch`) |
| Vector DB (ChromaDB, local persistent) | **Free** (embedded) |
| Data source (Wikipedia via `wikipedia-api`) | **Free** |
| UI / orchestration (Streamlit, LangGraph, Pydantic) | **Free / OSS** |
| **Total API spend** | **₹0** |
