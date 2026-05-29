<div align="center">

# 🏛️ WHAT IF?

### A counterfactual history engine — grounded, cited, and confidence-scored.

*Ask any "what if…" question about history. Get an answer that keeps **verified facts**
strictly separate from **simulated consequences** — with a source for every fact and a
confidence score for every claim.*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6B6B)
![Cerebras](https://img.shields.io/badge/LLM-Cerebras%20gpt--oss--120b-F55036)
![Embeddings](https://img.shields.io/badge/Embeddings-all--mpnet--base--v2%20(local)-FFD21E)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Cost](https://img.shields.io/badge/API%20cost-%E2%82%B90-2EA043)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## ✦ Why not just ask ChatGPT?

A raw chatbot answers a counterfactual by blending real history, plausible inference, and
confident hallucination into one unsourced paragraph. **WHAT IF?** is engineered to keep
those apart. Seven concrete differences:

| # | WHAT IF? | A raw chatbot |
|---|----------|----------------|
| 1 | **Grounded in a cited corpus (RAG)** — facts come from retrieved Wikipedia chunks, not parametric memory. | Facts from memory; no provenance. |
| 2 | **VERIFIED ≠ SIMULATED** — verified facts and model speculation live in two visually separate sections; never blended. | Fact and guess in the same sentence. |
| 3 | **Per-claim confidence** — every reasoning step is scored HIGH / MEDIUM / LOW / SPECULATIVE from how much evidence backs it. | One confident tone throughout. |
| 4 | **Hallucination guards** — max 4 causal steps; fabricated or ungrounded citations are detected and flagged. | Unbounded, unverifiable chains. |
| 5 | **Multi-agent pipeline** — 5 specialized agents (understand → retrieve → ground → reason → score), each returning a validated Pydantic model. | One free-form completion. |
| 6 | **Shows its work** — evidence chips per step, plus "What remains unknowable" and a "Historian's note" on the real debate. | Opaque. |
| 7 | **Auditable & evaluated** — 4 automated rule-checks (C1–C4) across a test suite; reproducible. | Not testable. |

…and it runs at **₹0** — free LLM tier, local embeddings, local vector DB.

---

## 🏗️ Architecture

```
   INGESTION (offline, once)
   Wikipedia ─► wikipedia_loader ─► chunker (~300-tok, paragraph-aware)
                                       └─► embedder ─► ChromaDB
                                            (all-mpnet, cosine · 18 sources · 1,466 chunks)

   QUERY (per question) — LangGraph StateGraph, ~4 LLM calls total

     question
        │
        ▼
   ① understand_query  ─►  ② retrieve  ─►  ③ ground  ─►  ④ reason  ─►  ⑤ score
      QueryAnalysis        RetrievalCtx     GroundedCtx   Counterfactual ScoredReasoning
      LLM (T=0.1)          no LLM           LLM ×2 (T=0)  LLM (T=0.3)    no LLM (logic)
        │
        ▼
   report_generator ─► Streamlit UI   (any node error → graceful halt, never crashes)
```

`① understand` parses the question → `② retrieve` pulls 8 primary + 3 analogy chunks →
`③ ground` extracts & classifies cited facts (VERIFIED/DEBATED/BACKGROUND) → `④ reason`
builds a ≤4-step `[SIMULATED]` causal chain citing `[EVIDENCE: chunk_id]` → `⑤ score`
rates each step by evidence. Full technical writeup: **[`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)**.

---

## 🎬 Demo

> _Placeholder — add a screen recording at `docs/demo.gif`._

![WHAT IF? demo](docs/demo.gif)

---

## 🚀 Run it locally

> Windows paths shown; adapt for macOS/Linux. The pipeline needs a free **Cerebras** API
> key. A full answer takes ~1–3 min on the free tier (rate-limit back-off — worth the wait).

```bash
# 1. Clone + create a virtualenv
git clone <your-repo-url> historyos && cd historyos
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env           # macOS/Linux: cp .env.example .env
#   then edit .env and set CEREBRAS_API_KEY=...   (OPENROUTER_API_KEY optional fallback)

# 4. Build the corpus (one-time; downloads + chunks + embeds the 18 topics)
python -m ingestion.wikipedia_loader     # Wikipedia -> data/raw/
python -m ingestion.chunker              # raw -> data/processed/
python -m ingestion.embedder             # processed -> ChromaDB

# 5. Launch the app
python -m streamlit run frontend/app.py
```

Then open the local URL Streamlit prints and ask a question (or click an example).

**Optional — run the evaluation suite** (8 cases, full A1→A5 chain):
```bash
python -m evaluation.evaluator
```

---

## 📊 Evaluation results

Four rule-checks per case, across the 8-case suite (`evaluation/eval_run_v4.log`):

| Check | Verifies | Score |
|-------|----------|:-----:|
| **C1** | Every step labelled `[SIMULATED]` | 7 / 8 |
| **C2** | Every step cites a real source `chunk_id` | 4 / 8 |
| **C3** | No more than 4 reasoning steps | **8 / 8** |
| **C4** | No simulation presented as fact | 7 / 8 |

C3 (the hallucination-guard step cap) holds across every case. C2 is the weakest
dimension — ungrounded/fabricated citations are **flagged** (→ SPECULATIVE), not silently
accepted. See the docs for the per-case breakdown.

---

## 📜 Example output

**Q:** *What if the Mughal Empire had industrialized before the British arrived?*

> **OVERALL · MEDIUM**
>
> **What might have happened** *(simulated)*
> - 🟢 **HIGH · 0–10 years** — A state-run workshop network emerges around Agra.
>   *Based on `mughal_empire_22`, `mughal_empire_15` — multiple verified facts on
>   pre-colonial manufacturing capacity.*
> - 🟡 **MEDIUM · 10–30 years** — Selective, Meiji-style modernization.
>   *Grounded in an analogy (`meiji_restoration_4`) rather than direct evidence.*
>
> **What We Know** *(verified · 16 facts from 3 sources)*
> - *India produced ~24.5% of world manufacturing output until 1750.* `[mughal_empire_22]` · Mughal Empire
>
> _Simulated consequences are AI-generated inferences, not historical fact._

---

## ⚠️ Known limitations

- **Latency:** ~1–3 min per question on the free Cerebras tier (rate-limit back-off).
- **Citation grounding (C2)** is the weakest check — fabricated/ungrounded citations are
  flagged but not yet prevented.
- **Small corpus** — 18 Wikipedia articles; out-of-corpus questions return an honest
  "no verified sources found".
- **Web fallback** (Tavily) is stubbed, not implemented.
- **Model drift** — the Cerebras free model list changes; update `.env`/`config.py` if a
  model 404s.

---

## 📄 License

Released under the **MIT License** — free to use, modify, and distribute with attribution.

---

<div align="center">
<sub>Built as a study in <b>honest AI</b>: separating what we know from what we imagine, and showing the difference. · API cost: ₹0</sub>
</div>
