# Repository Guidelines

## Project Structure & Module Organization
- `agents/` contains the orchestrator and stage modules (summaries, novelty, planner, iterate, write, interactive); keep entrypoints importable and free from side effects.
- `lab/` houses the experiment runner, pytest sandbox, and LLM-assisted code tools that support pipeline development.
- `utils/` centralizes configuration, logging, and API helpers—extend these utilities before adding duplicate logic elsewhere.
- Generated artifacts live in `data/`, `runs/`, `logs/`, and `paper/`; avoid committing heavyweight outputs unless they are intentionally versioned.
- Support material and prototypes sit under `docs/`, `experiments/`, and `dev/`; promote stabilized code into `agents/` or `lab/` when it graduates.

## Build, Test, and Development Commands
- Define `PYTHON=/mnt/c/Users/lewka/miniconda3/envs/deep_learning/python.exe` and reuse it in commands.
- `$PYTHON -m agents.orchestrator` runs the end-to-end pipeline using the active `config.yaml` and environment overrides.
- Stage-level work: `$PYTHON -c "from agents.summarize import process_pdfs; process_pdfs('pdfs','data/summaries')"` (swap imports to call other stages).
- `$PYTHON -m pytest -q` executes the suite; append paths such as `tests/agents` for targeted runs or `-k keyword` for filters.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, expressive names, and type hints on public functions; prefer `pathlib.Path` and structured config accessors.
- Expose callable functions instead of CLI wrappers or argparse; orchestration scripts should import and reuse pipeline logic directly.
- Keep docstrings action-focused, describing key inputs, outputs, and side effects the pipeline depends on.

## Testing Guidelines
- Place tests under `tests/`, mirroring package layout (`tests/agents/test_planner.py`, `tests/lab/test_sandbox.py`).
- Keep cases deterministic and fast; use fixtures or lightweight sample files in `data/` rather than hitting external APIs.
- Update `pytest.ini` only when discovery rules must change, and flag any slow or flaky tests in pull requests.

## Commit & Pull Request Guidelines
- Write imperative, scope-first commit messages (e.g., `Enhance planner validation`), grouping related edits and summarizing behavioral impact.
- Pull requests should cover intent, primary modules touched, config or schema updates, and attach relevant run artifacts or log excerpts.
- Confirm tests or stage executions before review and note any skipped checks with justification.

## Configuration & Security Tips
- Manage defaults in root `config.yaml`, override via `.env`, and document environment changes reviewers need for reproducibility.
- Guard API keys (`ELSEVIER_KEY`, `DEEPSEEK_API_KEY`, `X_ELS_INSTTOKEN`); never log raw secrets or commit proprietary outputs without redaction.
