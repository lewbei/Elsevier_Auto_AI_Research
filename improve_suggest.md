End-to-End LLM Research — Improvement Suggestions

Scope and Principles
- Goal: end-to-end research (summarize → novelty → plan → iterate → report) with programmatic APIs and no CLI/argparse/offline fallbacks.
- Find-papers stage is considered working; focus elsewhere.

Top Priorities
- Replace subprocess calls with direct Python calls (no CLI). See run_pipeline.py and agents/orchestrator.py.
- Remove offline fallbacks (lit_review, tier1) or gate them behind explicit flags defaulting to off.
- Add schema/shape tests for summarize, novelty, planner; keep GPT‑5 temperature omitted.

Key Components
- Orchestration: run_pipeline.py, agents/orchestrator.py
- Summarize: agents/summarize.py
- Novelty: agents/novelty.py, lab/novelty_scoring.py
- Planner: agents/planner.py
- Iterate/Runner: agents/iterate.py, lab/experiment_runner.py
- LLM: utils/llm_utils.py

Recommendations (Highlights)
1) Orchestration
- Unify on programmatic orchestration. Import modules and call functions instead of spawning Python with "-m".
- De-duplicate summarizer chunk-size logic; keep in one place (prefer orchestrator).

2) Summarization
- Add tests for schema shape and deterministic chunking. Enforce training knobs presence as short key:value lines.
- Keep failure behavior simple: skip on LLMError; do not synthesize offline output.

3) Novelty
- Consolidate to group_novelty_and_ideas_v2; deprecate v1 path in code comments.
- Extend novelty scoring with orthogonality penalty to promote diverse candidates.

4) Planner
- Keep multi-agent normalization and strict schema; add plan hash and compact tests for normalization+validation.

5) Iterate/Runner
- Add optional extended metrics (precision/recall/F1) behind YAML; keep defaults minimal.
- Add dataset validators for imagefolder/custom; clearer errors before training loop.
- Set cudnn.deterministic=True when CUDA is present (guarded) for determinism.

6) Codegen
- Add AST-based import whitelist checks in code_edit_loop for extra safety; add micro-tests for generated modules sanity checks.

7) Lit Review (no offline)
- Remove or gate _build_offline_review(); prefer explicit error when LLM fails.

8) Tier‑1 Validator (no offline)
- Remove or gate skinny fallback artifact writes on LLM failure.

9) LLM Utils
- Add request_id hashing in logs; centralize timeout/max_tries presets by stage; add tests for cache hit accounting.

10) Quick Wins
- agents/personas.py: remove duplicate import of chat_text_cached, LLMError.
- agents/novelty.py: remove redundant get aliases; keep only lab.config.get.
- agents/experiment.py: clarify role vs iterate; consider folding or documenting as demo.

Programmatic Usage (No CLI)
- agents.summarize.process_pdfs(pdf_dir, out_dir, ...)
- agents.novelty.main() after summaries exist
- agents.planner.main() to write data/plan.json
- agents.iterate.iterate(novelty_dict, max_iters)

