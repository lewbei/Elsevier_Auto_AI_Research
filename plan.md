Project: End-to-End LLM Research — Code Scan & Docs Update

Goal
- Perform a full-code scan of all Python files to understand the architecture and quality.
- Produce a detailed, actionable improvement report in improve_suggest.md.
- Update AGENTS.md to reflect a non-CLI workflow (no argparse, no offline fallbacks), oriented to end-to-end research.
- Validate changes by compiling sources and running tests.

Scope
- Read and analyze every .py file under repo root.
- Suggest changes; implement only documentation updates in this pass (no code rewrites unless required by tests/docs).
- Respect request to avoid CLI/argparse and offline fallback in documentation and future direction.
- Ignore find_papers implementation for improvement focus (it works well), but keep it documented as a pipeline stage.

Plan (Phases)
1) Inventory repo and map modules
2) Read core agent modules and orchestrator
3) Read lab utilities and helpers
4) Draft detailed improvement proposals (improve_suggest.md)
5) Update AGENTS.md (non-CLI, no offline fallback/argparse)
6) Compile sources and run tests

Execution Steps
1. List all .py files
2. Read files (agents → lab → utils → dev → tests)
3. Record observations and improvement items
4. Write improve_suggest.md with prioritized roadmap
5. Update AGENTS.md (usage patterns, env, runbook sans CLI/argparse/offline)
6. Compile all modules; run pytest; address doc-only issues if any

Risks
- Large surface area: missing subtle couplings or hidden behaviors
- Test failures unrelated to doc changes
- Environment-specific logic (Windows) may complicate suggestions
- Presence of generated files (lab/generated_*) could skew reading

Rollback
- Revert doc changes to previous state if needed (keep this commit isolated)
- No code behavior change in this pass; compile/tests verify safety
- If tests fail due to environment, run them selectively or annotate in suggestions

Notes
- No new CLI or argparse will be introduced.
- No offline fallback features will be added or emphasized in docs.
- Improvements will be specific and practical, with references to files and responsibilities.

---

Lab Folder Review and Refactor Plan

Goal
- Review all Python modules under lab/, fix a subtle transform composition bug, and add low-risk ergonomics (AMP, pin_memory, num_workers) to experiment_runner.

Plan
1) Inventory lab/*.py and read fully
2) Fix GeneratedAug double ToTensor issue
3) Improve experiment_runner data loaders and AMP
4) Add light type hints in generated_* files
5) Compile modules and run tests

Steps
- Standardize GeneratedAug to not include T.ToTensor; runner appends it.
- Update codegen prompt to discourage ToTensor in generated aug modules.
- In experiment_runner: add num_workers/pin_memory/persistent_workers; add optional AMP; add non_blocking .to(); log interval.
- Keep behavior identical by default; only enable when spec fields present.

Risks
- If GeneratedAug previously relied on its own ToTensor, missing conversion could occur; mitigated because runner appends ToTensor unconditionally.
- New spec toggles (amp, num_workers) may be misconfigured; guarded with safe defaults and try/except.

Rollback
- Revert edits to lab/generated_aug.py and lab/experiment_runner.py if any regressions appear.
- The changes are isolated; deleting generated_aug.py restores baseline behavior.

---

Production Hardening Plan (Lab)

Goal
- Move lab helpers toward production readiness: safer defaults, clear typing, optional performance toggles, and minimal tests.

Plan
1) Harden generated_train.update_spec with clamps and defaults
2) Make generated_train import-safe without torchvision
3) Add optional AMP, workers, pin_memory toggles (consumed by runner)
4) Add a focused pytest for update_spec normalization
5) Provide runbook and commands for CI usage

Risks
- Torch/torchvision may be absent; tests now skip if torch missing; transforms handle T=None.

Rollback
- Remove tests/test_generated_train_spec.py and revert lab/generated_train.py to prior version.
