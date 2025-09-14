Summary of analysis and recommendations (embedding model + advanced pipeline)

Key findings
- No embedding/vector retrieval is present in the repository. All semantic comparisons, novelty grouping, and idea generation are LLM-only.
- utils/llm_utils.py centralizes chat calls (OpenAI/DeepSeek/OpenRouter) with caching and cost tracking but no embeddings.
- agents/summarize.py, agents/novelty.py, agents/planner.py orchestrate JSON-first LLM flows without vector similarity search or reranking.
- Preflight scan detected many fallback patterns (e.g., `except Exception: pass`) across large generated modules; see artifacts/fallbacks_found.md.

Should you use an embedding model?
- Yes — for advanced novelty analysis, clustering, retrieval-augmented discussion, and stable reranking. Benefits:
  - Robust novelty scoring: compute max similarity to prior work (summaries + related_work) to estimate uniqueness before LLM reasoning.
  - Better clustering/themes: embed each paper’s methods/claims; cluster with KMeans/HDBSCAN to form themes prior to prompting.
  - Cheaper+faster prompts: use embeddings to preselect top‑k relevant context, shrinking LLM context and cost.
  - Deterministic reproducibility: cached vectors enable stable results across runs.

Recommended integration (GPU‑first)
- Module: `utils/embeddings.py` with:
  - `ensure_gpu()` (guard GPU unless `ALLOW_CPU_FOR_TESTS=1`).
  - `get_embedder(cfg)` supporting: local (sentence-transformers; e.g., `bge-small-en-v1.5`, `e5-base-v2`) and OpenAI (`text-embedding-3-large`).
  - `embed_texts(list[str]) -> np.ndarray` with on-disk cache: `data/embeddings/{model}/{sha256(text)}.npy`.
- Config (YAML):
  - `embeddings: { enable: true, provider: local|openai, model: bge-small-en-v1.5, dim: 768, cache: true }`
  - `novelty: { retrieval: { top_k: 8, min_sim: 0.30 } }`
- Usage:
  - Summaries: after writing each summary JSON, persist a compact `summary_text` (e.g., title + novelty_claims + methods) and its vector.
  - Novelty: build a small index (FAISS flat IP; if unavailable, exact numpy dot) of all vectors. For each candidate idea or focus area, retrieve top-k neighbors to feed the LLM.
  - Grouping: cluster vectors to produce initial themes; feed cluster summaries to the LLM for polished narratives.

Model choices
- Local/GPU (privacy + speed): `bge-small-en-v1.5` (384d) or `e5-base-v2` (768d). For larger budgets: `bge-large-en-v1.5`.
- Hosted (simple): OpenAI `text-embedding-3-large`.

Strictness and failure policy
- Enforce GPU: raise if CUDA absent unless `ALLOW_CPU_FOR_TESTS=1`.
- Hard-fail on missing packages (no silent downgrades). Document install commands; do not auto-install.

Next steps (non-breaking)
1) Add `utils/embeddings.py` and YAML config keys (no runtime change unless `embeddings.enable=true`).
2) Extend `agents/summarize.py` to write `summary_text` and emit vector when enabled.
3) Extend `agents/novelty.py` to build/load index and perform retrieval before LLM calls.
4) (Optional) Add FAISS dependency (Windows wheel) or start with exact numpy search for ≤10k items.
5) Remove broad `except Exception: pass` in critical paths (iteratively; see fallbacks report).

