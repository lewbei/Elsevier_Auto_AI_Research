# Lab Module Overview

The `lab/` package hosts experiments, code-generation helpers, and supporting
utilities used by the pipeline agents. This refactor clarifies how to use and
extend these modules.

## Configuration Flow
- `lab.config` is the single entry point for reading project configuration,
  layering `.env` values with `config.yaml`. All modules should use
  `lab.config.get*` helpers rather than reading the filesystem or env directly.
- `.env` is loaded once the first time `get_config()` is called. Subsequent
  calls reuse the cached dict. No other module should parse `.env`.

## Personas and Logging
- Persona prompts are still triggered by agents via `_persona_phase_notes`, but
  persona logic will be centralized gradually. For now, persona helpers should
  live under `agents/persona_support.py`.
- `lab.logging_utils` provides `append_jsonl`, `capture_env`, and `try_mlflow_log`.
  When adding new logging behavior, extend this module rather than introducing
  module-level prints or file writes.
- Use `lab.log_viewer` to inspect persona transcripts (`tail_persona_log`) and
  recent novelty chat rounds (`tail_novelty_session`). The module also exposes
  `summarize_run_costs` to review aggregated LLM spend from `runs/llm_cost_*.json`.

## Experiment Execution
- `lab.experiment_runner` handles preparing datasets, deciding between real and
  stub runs, and logging outcomes. It will be slimmed down to expose smaller
  functions (`prepare_dataset`, `run_spec`) to simplify testing.
- Supporting utilities:
  - `lab.code_edit_loop`, `lab.codegen_utils`, and `lab.codegen_llm` generate
    lightweight training hooks and augmentations.
  - `lab.mutations` proposes simple spec mutations; `lab.domain_templates`
    suggests defaults based on the detected domain.
  - `lab.analysis` summarizes experiment runs deterministically and with optional
    LLM commentary.

## Todos (Follow-up Work)
1. Extract persona scaffolding into a shared helper (`agents/persona_support`).
2. Provide a `lab.plan_store` module to manage plan artifacts instead of manual
   file handling across agents.
3. Refine `lab.experiment_runner` to separate environment checks from execution.
4. Add integration tests under `tests/lab/` to cover runner and codegen helpers.

This README reflects the refactor as of YYYY-MM-DD; update it as new utilities
are added or reorganized.
