 Paper Finder Progress Reporting (Windows + Git‑Bash)

Checklist (what I will do)

- Add counters for skip reasons and totals
- Include baseline counts from CSV/PDFs
- Print remaining to reach MAX_KEPT
- Compile and run pytest on Windows Python
- Share usage notes and rollback

Plan

- Extend agents.paper_finder to: seed dedupe from existing CSV as before, but also count previously kept rows; compute remaining target as MAX_KEPT - (kept_csv + kept_this_run); print progress in [SKIP]/[JUDGED]/[DONE] lines.

Steps

1) Implement counters + baseline summary in paper_finder
2) Compile sources and run pytest
3) Validate messaging with a dry run (user env)

Risks

- Missing API keys stop paper_finder at runtime; tests do not cover it
- Windows path quirks when printing CSV/PDF paths

Rollback

- Revert agents/paper_finder.py to previous version via git if prints are too noisy

Repository Restructure Roadmap

Checklist (what I will do)

- Decide target locations for top‑level modules
- Move files with minimal import churn
- Add wrappers to preserve existing test imports
- Update run commands and docs consistently
- Compile sources and run pytest to verify
- Provide rollback and migration notes

Plan

- Consolidate utilities into a `utils/` package and move `llm_utils.py` and `pdf_utils.py` there.
- Move `paper_finder.py` into `agents/` so all pipeline entry points live under one package.
- Update imports across `agents/*` and add thin top‑level wrappers (`agents_iterate.py`, `agents_write_paper.py`) to keep tests stable.
- Update `run_pipeline.py` and `AGENTS.md` to call `python -m agents.paper_finder` instead of the previous top‑level script.

Steps

1) Create `utils/` package and move `llm_utils.py`/`pdf_utils.py`.
2) Move `paper_finder.py` to `agents/paper_finder.py`.
3) Fix imports in `agents/*` and docs; add top‑level wrappers for tests.
4) Update `run_pipeline.py` and `AGENTS.md` commands.
5) Compile all sources and run pytest.

Risks

- Import path regressions if any reference still points to old locations.
- Pipeline docs drift if commands are not updated consistently.
- Hidden references in notebooks or scripts not covered by tests.

Rollback

- Keep thin compatibility wrappers at the repo root so external users importing legacy names continue working.
- If needed, revert moves by restoring files to the top level and removing new package paths.
