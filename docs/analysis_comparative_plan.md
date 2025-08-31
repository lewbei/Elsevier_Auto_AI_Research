# Comparative Analysis and Gap‑Closure Plan

## Scope
- Compare AI‑Scientist v2 and Agent Laboratory to our current pipeline and propose a practical roadmap to surpass them on targeted dimensions while preserving our lean, headless, Windows‑friendly design.

## Comparative Snapshot
- **Planning & Control:**
  - AI‑Scientist: staged tree search (4 stages), journal/checkpoint, VLM‑aided evaluation.
  - Agent Lab: multi‑agent prompts, phase metrics, HITL toggles, prompt overrides.
  - Ours: novelty → iterate with refine, basic planner, best‑run/aggregates, plots.
- **Experiments:**
  - AI‑Scientist: progressive multi‑dataset, debugging depth, job manager.
  - Agent Lab: MLESolver, tool‑based experimentation.
  - Ours: small matrix + repeats, dataset flag, guardrails.
- **Writing & Reporting:**
  - AI‑Scientist: LaTeX templates, plotting, bibliography.
  - Agent Lab: report workflows.
  - Ours: MD/LaTeX, refs.bib from CSV, accuracy figure, best run, mean±std.
- **Ops:**
  - AI‑Scientist: heavy GPU/Linux stack, long runs.
  - Agent Lab: broad integrations, more dependencies.
  - Ours: minimal dependencies, resilient, fast tests.

## Prioritized Roadmap (Additive, Env‑Gated)
1. **Stage Manager (light) and Search**
   - `agents_stage_manager.py`: stages [setup, refine, ablate]; stop rules (success, budget, max iters).
   - `lab/mutations.py`: propose 2–3 safe spec mutations per refine (beam=2) and integrate into matrix.
2. **Prompt Overrides and HITL**
   - `prompts/` for planner and paper sections; optional `HITL_CONFIRM` to require confirmation before finalize/iterate.
3. **Literature Metrics**
   - Track counts in novelty step; write `data/lit_stats.json`; surface in paper.
4. **Optional VLM Hook**
   - `vlm_utils.py` interface; image‑only feedback on `runs/accuracy.png` when `VLM_ENABLED`.
5. **MLflow Default Helper (env‑overridable)**
   - Auto‑enable if installed and unset; keep opt‑out via environment variable.

## Validation Plan
- Extend tests for mutations selection, stage manager transitions, prompt overrides, and helper utilities.
- Keep pytest fast; do not introduce GPU/runtime‑heavy tests.

## Expected Outcome
- Planning depth, search coverage, and reporting move closer to AI‑Scientist/Agent Lab while retaining our strengths: reliability, simplicity, portability, and speed.

