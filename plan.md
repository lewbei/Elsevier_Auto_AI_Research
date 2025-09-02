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

