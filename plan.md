Here’s a focused plan and analysis of what your pipeline lacks compared to AI‑Scientist‑v2 and lab_agents, with a minimal roadmap to close the most valuable gaps.

Plan

Inventory features in your repo (current state).
Map against AI‑Scientist and lab_agents capabilities.
Identify missing pieces by category (planning, codegen, experiments, ops, writing).
Propose a lean, staged roadmap to close gaps without overengineering.

Next‑Step Checklist

- Add dataset flag with a small adapter (no heavy deps).
- Add codegen guardrails (compile/import/smoke) for generated aug/head.
- Record best run across the iteration matrix for reporting.
- Update paper draft to include best result details.
- Add 1–2 tiny tests to cover guards and best‑selection.

In‑Progress (Phase 1 + 2)

- Multi‑agent planning session (PI → Engineer → Reviewer) with session log JSONL and offline fallback.
- Paper drafting polish: generate BibTeX from the CSV and cite in LaTeX; embed accuracy figure and mean±std table.

Upcoming (Current Sprint)

- Export best_config.json and aggregates.json in runs/.
- Render mean±std table for baseline/novelty in Markdown and LaTeX, and embed accuracy figure.
- Add tests for table rendering and file exports (logic-level only).
What You Already Have

One‑command pipeline: find → summarize/critic → novelty synthesis → iterative runs.
Novelty‑driven codegen: generated augmentation and classifier head, auto‑wired to training.
Iteration depth: baseline/novelty/ablation × model variant + small lr/input tweaks; refine step.
Ops/repro: seeds, env capture, JSON logs, MD + HTML dashboard; optional MLflow; LLM caching.
Practical robustness: OA gating, dedupe resume, 429 handling, time budget, optional parallel.
What You Likely Lack (vs. AI‑Scientist‑v2, lab_agents)

Planning & Research Loop

Structured research plan with explicit hypotheses, success criteria, and stopping rules tied to novelty themes.
Multi‑stage literature reasoning (e.g., memory of past findings, cost/usage tracking).
Cross‑paper rationale for selecting baselines and datasets beyond ISIC.
Codegen & Verification

Broader codegen scope (custom layers/models/pipelines, training loops, data loaders).
Auto‑debug loop: compile/run error capture, targeted fixes, re‑run until green.
Unit/integration tests for generated code; smoke tests that gate execution.
Experiment Breadth & Rigor

Multi‑dataset evaluation (e.g., ISIC + small public benchmarks), dataset bootstrap/downloaders.
Systematic ablation matrices (optimizer families, schedulers, augment suites, heads).
Sweeps with small search spaces and automatic selection of best config.
Statistical checks (e.g., repeated runs/CI bounds) and clear baselines.
Ops/Repro & UI

Rich experiment tracking (always‑on MLflow/W&B; charts, comparisons).
Artifact management (models, logs, plots, tables) with consistent run registry.
Containerization or env lockfiles (Docker/conda‑lock) for portable repro.
Parallel orchestration at the matrix level (queues, retries).
Writing & Reporting

Auto paper drafting (LaTeX/Markdown) with figures, tables, and BibTeX references.
Automatic result tables and plots derived from runs (with captions).
Abstraction for updating drafts after each iteration.
Why They Have These

AI‑Scientist/lab_agents aim for “research autonomy”: plan → code → run → analyze → refine, with strong ops and reporting so outputs are publishable and reproducible.
Lean Roadmap (no overengineering)

Phase 1: Research Plan + Paper Draft (high impact, low complexity)

Add a “Planner” step that turns novelty_report into a structured experiment plan with hypotheses and success metrics (JSON).
Add agents_write_paper.py to generate a Markdown/LaTeX draft using your runs:
Sections: Abstract, Intro/Related (from summaries), Methods (novelty), Experiments, Results (tables from runs), Discussion, Limitations, Conclusion.
Save to paper/paper.md and paper/main.tex; attempt pdflatex if available.
Hook into run_pipeline with WRITE_PAPER=1.
Phase 2: Dataset Bootstrap + Broader Ablations

Add simple dataset adapters and a bootstrap for a second dataset (e.g., CIFAR‑10) behind a `DATASET` flag. Prefer auto‑download only when enabled and fall back to `FakeData` if unavailable. Expand ablation grid lightly if budget allows. Keep budget guard and optional parallel to avoid bloat.
Phase 3: Codegen + Auto‑Debug Guardrails

Extend codegen to a small set of safe modules (augmentation and a simple head). Add auto‑debug guardrails:
Generate → py_compile → import → run a tiny smoke call (no heavy deps) → integrate only if all pass. Add 1–2 quick tests for guards.

Phase 2.5: Reporting Polish

Pick the best run across the matrix and save a short `runs/best.json`. Update the Markdown/LaTeX draft to include the best result and key hyperparameters. Add a small bar‑chart later (optional) once matplotlib is available.
Phase 4: Ops Defaults (still light)

Turn MLflow on by default if installed; otherwise no‑op.
Save plots (accuracy bars over runs) and embed in dashboard + LaTeX.
Optional: conda‑env export per run, minimal Dockerfile later.
Risks & Mitigations

LLM drift/cost: keep caching and caps; use smaller, deterministic prompts for plan/paper.
Dataset complexity: start with one extra small dataset (auto‑download + transforms).
Codegen safety: gate execution with compile/import/smoke tests; allow quick rollback (skip generated module).
If you want, I can:

Add agents_write_paper.py and integrate it into run_pipeline (Phase 1).
Add a planner step that writes a compact plan.json with hypotheses/metrics.
Wire a second small dataset adapter behind a flag to broaden results a notch.
