# Novelty Facets + Scoring Gate (Windows + Git‑Bash / CMD)

Checklist (what I will do)

- Add facet extractor to ground ideas in observed summaries.
- Add idea scorer/gate to demote bland/off‑facet ideas.
- Patch novelty to inject facets, require deltas, and multi‑temperature sample.
- Optionally tighten persona prompts to cite titles + falsify deltas.
- Compile sources and run pytest to validate.

Reasoning Protocol

- Plan: Mine facets from existing summaries, guide the LLM with those facets, enforce richer idea schema (derived_from_titles + delta_vs_prior + numeric specs), and filter weak ideas before writing the report.
- Steps: Facets module • Scoring module • Patch novelty imports + function + main • Optional persona prompt tweaks • Compile • Pytest.
- Risks: LLM calls need keys; prompt tightening may reduce idea count; scoring may over‑penalize if facets are sparse; Windows quoting for compile.
- Rollback: Disable via env (NOVELTY_DETAILED=0 or NOVELTY_ONLY_IDEAS=1); skip scoring (set keep_top high, min_score low); revert plan if needed.

Steps (execution)

1) Create `lab/novelty_facets.py` to extract facets from `data/summaries/*.json`.
2) Create `lab/novelty_scoring.py` to score and filter `new_ideas_detailed`.
3) Edit `agents/novelty.py`:
   - Imports for `extract_facets` and `score_and_filter_ideas`.
   - Update `group_novelty_and_ideas(..., facets=...)` to inject facets, require derived_from_titles + delta_vs_prior, and sample temperatures.
   - In `main()`, build facets, call the updated grouper, then score/filter before report.
   - Optional: tighten persona prompts to cite titles and include a falsifying ablation.
4) Compile and run tests.

Runbook (Windows CMD from Git‑Bash or PowerShell)

- Install deps: `cmd.exe /C "python -m pip install -r requirements.txt --disable-pip-version-check"`
- Compile: `cmd.exe /C "python dev\compile_all.py"`
- Tests: `cmd.exe /C "python -m pytest -q"`
- Novelty (after summarize): `cmd.exe /C "python -m agents.novelty"`

Notes

- Optional sampling: set YAML `pipeline.novelty.sampling.temperatures: [0.2, 0.6, 0.9]`.
- Optional ban list: `NOVELTY_BAN="resnet18,flip,jitter"`.
