# Questions Requiring Your Confirmation

## Embeddings & Retrieval
1) Provider: Do you prefer local GPU embeddings (sentence-transformers) or hosted (OpenAI `text-embedding-3-large`)?
-local, google/embeddinggemma-300m， required api key for hugging face. if choose openai at the start then use the open ai.
2) Model: If local, pick one: `bge-small-en-v1.5` (384d, fast), `e5-base-v2` (768d), or another exact model name. If hosted, confirm OpenAI model.
-local, google/embeddinggemma-300m， required api key for hugging face. if choose openai at the start then use the open ai.
3) Index: OK to start with exact cosine search (NumPy) and optionally add FAISS FlatIP later? Or require FAISS from day one?
-Use the best one, faiss-gpu
4) Retrieval params: Default `top_k=8`, `min_sim=0.30` acceptable? Any target values?
-top_k=8, min_sim=0.30 acceptable
5) Embed fields: For each summary, embed `title + novelty_claims + methods + limitations` as `summary_text`. Is this the right content, or include other fields?
-Try with the orignal paper content first, if not good then try with title + novelty_claims + methods + limitations.
6) Cache policy: Store vectors under `data/embeddings/{model}/` with content-hash sharding. OK? When summary text changes, re-embed (detected by hash) — confirm.
-Ok.

## LLM Provider & Models
7) Default profile: `llm.default = gpt-5-mini` in `config.yaml`. Keep OpenAI as default for all stages, or do you want to use the custom DeepSeek profile for some stages? Please list any per-stage overrides.
-All stage will be using the choose provider at the start, if choose openai then all stage use openai, if choose deepseek then all stage use deepseek.
8) Keys: Can you confirm availability of env vars for chosen providers (e.g., `OPENAI_API_KEY` for OpenAI, `DEEPSEEK_API_KEY` for DeepSeek)? Do not post values; just confirm presence.
-Yes, I have the key for both openai and deepseek. i also have hugging face api key.
9) Streaming: `llm.stream: true` is set. Keep streaming on for all stages?
-Yes, keep streaming on for all stages.

## GPU / Environment
10) GPU details: Which GPU(s) and VRAM are available? (e.g., RTX 3090 24GB) Any multi-GPU setup?
-RTX 3050 4GB, no multi-GPU setup.
11) Torch CUDA: Is your `deep_learning` Conda env running a CUDA-enabled PyTorch? If not sure, I can add a one-liner check to run manually.
-Yes, since this is my pc, if use other then it may not be deep_learning env.
12) CPU for tests: OK to allow CPU only when `ALLOW_CPU_FOR_TESTS=1` for unit tests (e.g., tiny embedding smoke tests)?
-ok
## Pipeline Behavior
13) Stage toggles: Many are `skip: true` (find_papers, summaries, novelty, planner, iterate). Do you want me to enable them for an end-to-end run after integration? Which stages should run by default?
-Yes, enable them for an end-to-end run after integration.
14) Paper drafting: `pipeline.write_paper_llm.enable: true`. Keep LLM drafting enabled?
-Yes, keep LLM drafting enabled.
15) Concurrency: May I parallelize PDF summarization/embedding within safe rate limits? Any preferred max workers?
-Yes, please parallelize within safe limits.

## Privacy & Compliance
16) External calls: If hosted embeddings/LLMs are used, is it acceptable to send paper text/snippets to the provider? If not, we will stay fully local.
-Yes, it is acceptable to send paper text/snippets to the provider.
17) Data retention: Any constraints on storing vectors/summaries on disk under `data/` and `.cache/`?
-No constraints.

## Testing & Quality Gates
18) Unit tests: OK to add tests for the embeddings module and retrieval (synthetic data) to keep coverage 100% lines/branches?
-yes
19) Network in tests: Allow network for hosted embedding tests, or should we skip hosted code paths in tests and test only local embeddings?
-local embedding test only.
20) CPU-only tests: Permit setting `ALLOW_CPU_FOR_TESTS=1` during CI/unit tests to avoid GPU dependency in CI, while keeping runtime GPU-only?
-no cpu

## Fallbacks Refactor
21) Priority areas: The preflight found many `except Exception: pass` in large generated modules. Which areas should I clean first: `utils/llm_utils.py`, `agents/*` core paths, or generated lab files?
-`utils/llm_utils.py` first.
22) Policy: Do you want hard-fail on all LLM/network errors in core stages, or retain narrowly-scoped retries (with explicit exceptions and logging)?
-Hard-fail on all LLM/network errors in core stages.
## Data & Corpus
23) Scope: Roughly how many papers (order of magnitude) do you plan to process (hundreds, thousands, tens of thousands)? This informs index choice.
-40 papers now, may be 100+ in the future.
24) Multilingual: Any non-English papers expected? If yes, we should select a multilingual embedding model (e.g., `bge-m3`).
-no
25) Additional corpora: Do you want to include your own notes, lab reports, or external corpora in retrieval context?
-no, all are only research paper that related to the goal in the config.yaml.

## PDF Extraction
26) Upgrades: Is it OK to add PyMuPDF (`pymupdf`) for better text extraction if needed? Or keep current `utils/pdf_utils.py` only?
-Ok to add PyMuPDF (`pymupdf`) for better text extraction if needed.

## Logging, Cost & Budgets
27) Budgets: Should I enforce a cost/token budget per run (e.g., hard-fail if exceeding a configured USD or token cap)? If yes, provide caps.
-no
28) Telemetry: Is writing per-run cost summaries to `runs/llm_cost.json` sufficient, or do you want per-stage latency histograms and a compact HTML report?
-per-stage latency histograms and a compact HTML report.

## Windows / Tooling
29) Interpreter path: Confirm this exact Python path exists: `C:\\Users\\lewka\\miniconda3\\envs\\deep_learning\\python.exe`.
-Yes, in this pc it exists. If other pc then may not exist.
30) Package install: May I add pinned dependencies to `requirements.txt` (e.g., `sentence-transformers`, `faiss-cpu` or `faiss-gpu`, `scikit-learn`)? I will not auto-install; just document.

-Yes, you may add pinned dependencies to `requirements.txt`.