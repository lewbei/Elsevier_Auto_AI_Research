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

---

# Planner Alignment with Novelty v2

Checklist (what I will do)

- Pass novelty outline (problems/objectives/contributions/research_questions) to planner.
- Provide top novelty_ideas (id/title/kind/spec_hint/eval_plan) as candidates.
- Update PI/Engineer/Reviewer prompts to ground on these fields.
- Improve offline fallback to pick a candidate and seed tasks.
- Compile sources and run pytest to validate.

Reasoning Protocol

- Plan: Make planner consume the richer novelty report by explicitly surfacing outline fields and filtered novelty_ideas, so plans choose a concrete novelty_focus and actionable tasks aligned with eval_plan/spec_hint.
- Steps: Inputs (outline+candidates) • Prompt updates • Offline fallback enrichment • Compile • Tests.
- Risks: LLM may ignore fields; candidate selection order may not reflect quality; Windows quoting for compile.
- Rollback: Revert to previous prompts; disable candidate injection via YAML flag if needed; fall back to offline minimal plan.

Runbook (Windows CMD from Git‑Bash)

- Compile: `cmd.exe /C "python dev\compile_all.py"`
- Tests: `cmd.exe /C "python -m pytest -q"`

---

# Autonomous Experimentation + Domain Templates + Novelty Check

Checklist (what I will do)

- Add domain templates and auto-select per goal/dataset
- Enable LLM-driven code edits to influence specs dynamically
- Add external novelty check (Semantic Scholar) with offline fallback
- Close the loop: analyze results → LLM suggests next changes
- Keep everything Windows + git-bash friendly and stub-safe

Reasoning Protocol

- Plan: Introduce minimal modules that hook into existing iterate/novelty paths without breaking current behavior. Use feature flags via YAML/env. Keep offline fallbacks.
- Steps: Templates • Iterate integration • External novelty check • Novelty wiring • Iterate update_spec hook • Compile/tests.
- Risks: Missing API keys, flaky network, no torchvision on Windows, brittle LLM outputs.
- Rollback: Disable via flags (CODEGEN_ENABLE=0, NOVELTY_EXT_CHECK=0), fall back to deterministic transforms, skip template detection.

Implementation Notes

- lab/domain_templates.py: detect_domain(goal, dataset) → {'cv'|'nlp'|'rl'|'generic'} + template hints. Exposed helpers for iterate prompts/spec defaults.
- lab/novelty_check.py: query Semantic Scholar (Graph API) for each idea title/keywords; annotate with hits and a novelty score; write data/novelty_search.jsonl; offline fallback.
- agents/novelty.py: optional external novelty filter controlled by config novelty.external_check.enable or env NOVELTY_EXT_CHECK=1. Blends ext novelty into tier1_score; filters low-novelty ideas.
- agents/iterate.py:
  - Include domain/template hints when proposing baseline/novelty/ablation.
  - Before running, call lab.generated_train.update_spec(spec) if present so LLM-written code can steer runs.
  - Journal each iteration under experiments/journal.jsonl with compact context + decisions.

Windows Runbook

- Python: use cmd.exe /C python ... from git-bash; no python3.
- Flags: CODEGEN_ENABLE=1 (LLM aug), CODEGEN_EDITOR=1 (training hooks), NOVELTY_EXT_CHECK=1 (external check).

Validation

- Compile all Python files to bytecode.
- Run pytest -q. Adjust only code touched if tests expose issues.

---

# Summaries: Related Work + Citations

Checklist (what I will do)

- Extend summarize schema to include related_work[].
- Capture citation strings (title/venue/year/doi/url) per item.
- Normalize and write related_work to each summary JSON.
- Compile sources and run pytest to validate.

Reasoning Protocol

- Plan: Teach the summarizer prompts (pass1/fill/finalize) to extract a concise related_work list from the paper text (usually the References section), keeping short citation strings and optional DOI/URL. Coerce to a stable object list in the final JSON to reduce novelty conflicts downstream.
- Steps: Update prompts/schemas • Normalize output • Compile • Pytest.
- Risks: PDFs missing explicit references; noisy extraction; longer payload size.
- Rollback: Remove field or cap size via YAML/env; ignore during novelty if noisy.

Runbook (Windows CMD from Git‑Bash)

- Compile: cmd.exe /C "python dev\compile_all.py"
- Tests:  cmd.exe /C "python -m pytest -q"
