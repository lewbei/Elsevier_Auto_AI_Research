Iterative Code Creation Integration (LLM-driven aug)

Checklist (what I will do)

- Inspect example folder + iterate flow
- Add LLM-driven augmentation codegen helper
- Wire config toggles and safe fallback
- Compile sources and run pytest
- Provide run instructions and rollback

Plan

- Introduce lab/codegen_llm.py with write_generated_aug_from_llm() that asks the LLM to output a tiny torchvision augmentation module (GeneratedAug) using a strict transform whitelist. Keep outputs minimal and safe.
- Hook codegen into agents/iterate.py engineer_refine_specs() and per-iteration setup. Gate with YAML config pipeline.codegen.enable or env CODEGEN_ENABLE=1. Fall back to the existing deterministic write_generated_aug() when LLM is unavailable or returns invalid code.
- Leave experiment runner unchanged except for its existing import of lab/generated_aug.py.

Steps

1) Add lab/codegen_llm.py (helper + safety checks + codeblock parsing)
2) Update agents/iterate.py to call LLM codegen when enabled; fallback otherwise
3) Document flags (CODEGEN_ENABLE and pipeline.codegen.enable) in AGENTS.md
4) Compile repository and run pytest
5) Validate iterate runs locally with CODEGEN_ENABLE=1

Risks

- Missing DEEPSEEK_API_KEY: codegen silently falls back to deterministic aug
- LLM emits invalid code: helper validates shape, else fallback triggers
- Excessive transforms: whitelist enforcement rejects risky code
- Runtime import failures on Windows: existing sanity checks mitigate

Rollback

- Disable via CODEGEN_ENABLE=0 or pipeline.codegen.enable=false (YAML)
- Remove lab/generated_aug.py to reset to default behavior
- Revert agents/iterate.py imports/lines referencing lab.codegen_llm

Advanced REPLACE/EDIT Editor (training hooks)

Checklist (what I will do)

- Add constrained REPLACE/EDIT loop to synthesize lab/generated_train.py
- Integrate optional hooks into experiment runner
- Gate behind YAML/env flags, with safe defaults
- Compile and test

Plan

- Implement lab/code_edit_loop.py with run_codegen_editor():
  Generates a tiny module defining build_train_transforms, build_model_head, update_spec,
  using only torchvision transforms and torch.nn layers, with strict safety filters.
- Update lab/experiment_runner.py to optionally import lab/generated_train and consume hooks:
  transform override/extend; head override; spec tweaks kept minimal. All inside try/except.
- Add flags: pipeline.codegen.editor.enable (YAML) and CODEGEN_EDITOR=1 (env).

Steps

1) Add lab/code_edit_loop.py (editor + safety + import checks)
2) Integrate hook attempt in agents/iterate.py when enabled
3) Update experiment_runner to consult generated_train hooks
4) Document flags in AGENTS.md
5) Compile and run pytest

Risks

- Missing DEEPSEEK_API_KEY: falls back silently
- LLM produces invalid/unsafe code: rejected by safety checks, loop retries, then defaults
- Hook import errors: guarded by try/except; experiment continues

Rollback

- Disable via CODEGEN_EDITOR=0 or pipeline.codegen.editor.enable=false
- Remove lab/generated_train.py

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


Verbose Mode + Detailed Logging

Checklist (what I will do)

- Add global verbose toggle (LOG_LEVEL/VERBOSE)
- Trace LLM payload/response to logs/llm
- Print experiment/spec details when verbose
- Update docs and .env example
- Compile and run pytest

Plan

- Introduce debug helpers in lab/logging_utils.py: get_log_level(), is_verbose(), vprint().
- Add optional LLM logging in utils/llm_utils.py gated by LLM_LOG or LOG_LEVEL=debug; write JSON files to logs/llm/ and echo paths when verbose.
- Surface details in runners/agents when verbose: spec summaries, stub reasons, metrics, and plan summaries.
- Document toggles in AGENTS.md and .env.example.

Steps

1) Add helpers to lab/logging_utils.py
2) Add LLM logging to utils/llm_utils.py
3) Instrument experiment runner and agents for verbose prints
4) Update .env.example and AGENTS.md
5) Compile repo and run pytest

Risks

- Excessive console noise if left on; default remains info
- Minor I/O overhead for LLM logs; gated behind flags
- Windows CRLF/encoding concerns; file I/O uses utf-8

Rollback

- Disable by setting LOG_LEVEL=info and LLM_LOG=0
- Remove logs/llm/ if space is a concern
