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
