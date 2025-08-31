Agent Laboratory — Code-Level Analysis and Lift‑Over Plan

Overview
- Purpose: Multi‑agent workflow for literature, planning, experimentation, and writing with configurable prompts and HITL options.
- Core modules: `lab_agents/agents.py`, `ai_lab_repo.py`, `inference.py`, `utils.py`, `tools.py`, `mlesolver.py`.

Architecture Highlights
- Multi‑agent prompts and overrides:
  - `agents.py:1` loads hierarchical prompt overrides from `AGENT_PROMPTS_DIR`, supports six built‑ins (professor, postdoc, PhD, ML/SW engineer, reviewer), and robust JSON extraction.
- Lab workflow orchestration:
  - `ai_lab_repo.py:1` orchestrates phases (lit review → plan → experiments → interpretation/write), tracking timing/steps and statistics per phase.
  - Extensive env‑driven toggles: recent query dedupe, full‑text budget, auto‑add controls, save cadence, HITL flags.
- Tools and solvers:
  - `tools.py` for Arxiv/HF search and code execution; `mlesolver.py` specialized experimentation harness.
- Inference client:
  - `inference.py` LLM wrapper for model/query routing.

Strengths
- Practical operator features: prompt override tree, HITL toggles, literature dedupe/progress tracking, and per‑phase metrics.
- Review/scoring util handles diverse LLM output shapes.
- Tests cover orchestration, prompts, and tools comprehensively (though require extra deps by default).

Limitations/Risks
- Broad dependency surface (Gemini, various APIs); tests import many modules.
- App‑scale complexity and UI coupling not ideal for minimal headless pipelines.

Lift‑Over Opportunities (to our pipeline)
- Prompt override support: Add a small prompt directory and lookup for planner and novelty grouping prompts.
- Phase stats: Extend our `runs/env.json` to include per‑step timing/summary; we already capture env and metrics.
- Literature metrics: Track dedupe counts, summary counts in our novelty step; log to `logs/lit.jsonl`.
- HITL flags: Add optional pauses for plan confirmation and spec approval.

Concrete Action Items
1) Prompt override loader:
   - New: `prompts/` folder with optional `planner/*.md` and `paper/*.md`; extend agents_planner to load overrides.
2) Phase stats and logs:
   - Add small counters in `agents_novelty.py` for papers processed, kept, downloaded; write `data/lit_stats.json`.
3) HITL hooks:
   - Gate `agents_planner` finalization and `agents_iterate` spec approval behind `HITL_CONFIRM` env.

Validation
- Keep non‑interactive by default. Retain fast pytest. Avoid adding heavy external deps.

