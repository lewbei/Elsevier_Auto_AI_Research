# AI‑Scientist v2 — Code-Level Analysis and Lift‑Over Plan

## Overview
- **Purpose:** End‑to‑end autonomous research with progressive agentic tree search and multi‑stage experimentation.
- **Core modules:** `ai_scientist/llm.py`, `ai_scientist/treesearch/*`, `ai_scientist/tools/*`, `perform_*` executors, LaTeX templates.

## Architecture Highlights
- **LLM API shim (OpenAI + model quirks):**
  - `ai_scientist/llm.py:1` handles model idiosyncrasies (max_tokens vs. max_completion_tokens, temperature constraints) with backoff and token usage tracking.
- **Progressive multi‑stage manager:**
  - `treesearch/agent_manager.py:1` defines explicit stages (initial_implementation → baseline_tuning → creative_research → ablation_studies) with per‑stage goals, max iterations, and transition criteria.
  - Creates `ParallelAgent` for each stage and curates prompts with context, stage names, and goals.
  - Saves checkpoints and journals (best node, good nodes, history).
  - Uses VLM feedback for figure analysis and stage completion checks.
- **Tooling:**
  - `tools/semantic_scholar.py:1` with backoff, search, formatting, and optional API key.
  - Write‑up and plotting steps (`perform_*`).

## Strengths
- Strong staged flow with explicit goals and stopping/transition rules.
- Defensive LLM client handling diverse models and token semantics; backoff and token tracking.
- Journaling and checkpointing allow resumption and post‑hoc analysis.
- VLM feedback integration for figure evaluation.

## Limitations/Risks
- Heavy dependencies (CUDA, Linux focus) and long runtime footprint.
- Tree search complexity increases latency and cost.
- Safety concerns executing LLM‑written code; needs strict sandboxing.

## Lift‑Over Opportunities (to our pipeline)
- Stage orchestration: Add a minimal 2–3 stage manager around our iterate loop with clear goals and stop rules.
- Light beam search: At each refine step, propose K small spec mutations and select the best; reuse our `verify_and_fix_spec` to guard.
- Journaling: Persist per‑stage JSONL with best runs and decisions; we already write summary/best_config.
- LLM shim: Adopt limited quirks handling from `llm.py` into `utils/llm_utils.py` (e.g., token params) behind an environment flag.
- VLM slot: Add an optional VLM review hook with a clear on/off flag to keep headless safe.

## Concrete Action Items
1. **Stage wrapper:**
   - New: `agents_stage_manager.py` with stages [setup, refine, ablate]; wraps `iterate()` calls and applies stop rules (success, budget, max iters).
   - Logs to `runs/stage_session.jsonl`.
2. **Beam mutations:**
   - New: `lab/mutations.py` proposes 2–3 adjacent tweaks (lr, steps, input_size, optimizer) with verify; feed into matrix.
3. **LLM client quirks:**
   - Extend `utils.llm_utils.chat_json` to optionally map `max_tokens`/`max_completion_tokens` and ignore unsupported temperature for specified models.
4. **Optional VLM feedback hook:**
   - New: `vlm_utils.py` interface; analysis of saved `runs/accuracy.png` only if `VLM_ENABLED`.

## Validation
- Keep additive and env‑gated; rely on pytest only. No GPU assumptions.

---
# Project Structure

- agents/: Packaged steps (novelty, planner, iterate, write_paper).
- agents/paper_finder.py: Downloader + relevance filter (Elsevier + DeepSeek).
- lab/: Shared library utilities used across agents.
  - config.py: YAML-first configuration loader.
  - experiment_runner.py: Minimal train/val/test runner with PyTorch.
  - codegen_utils.py: Safe, tiny codegen for aug/head modules.
  - logging_utils.py, report_html.py, plot_utils.py, mutations.py, prompt_overrides.py.
- data/: Pipeline artifacts and derived JSON.
- paper/: Draft outputs (Markdown + LaTeX).
- tests/: Unit tests.
- config.yaml: Domain-agnostic settings (goal, dataset, research outline sizes).

Conventions

- Prompts are domain-agnostic; only `project.goal` shapes intent.
- Dataset supports imagefolder, CIFAR10, or custom loaders; train/val/test are explicit.
- Evaluation reports highlight test metrics; selection remains on validation.

