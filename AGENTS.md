End‑to‑End LLM Research: Init Plan (Windows + Git‑Bash)

Checklist (what I will do)

- Verify Windows Python and env (.env) are set.
- Install requirements with pip (Windows Python).
- Compile all Python sources to catch syntax issues.
- Run pytest to confirm repo health.
- Provide step‑by‑step run commands for the pipeline.
- Note risks and rollback switches.

Reasoning Protocol

- Plan: Configure env + dependencies, validate via compile/tests, then run the LLM pipeline end‑to‑end with clear toggles and budgets.
- Steps: Env • Deps • Compile • Tests • Run pipeline (find → novelty → planner → iterate → paper).
- Risks: Missing API keys; rate limits; dataset not present; Windows PyTorch not installed; network variability.
- Rollback: Use offline fallbacks, disable expensive steps, enable FakeData, or run only subsets; clear caches if needed.

Environment Setup (Windows + Git‑Bash)

- Windows Python: use cmd to run Python (no python3).
  - Quick check: cmd.exe /C python -V
  - If you must reference the absolute path, the Windows path is typically C:\Users\lewka\miniconda3\envs\deep_learning\python.exe
    and the Git‑Bash style is /c/Users/lewka/miniconda3/envs/deep_learning/python.exe, but using cmd.exe /C python is preferred.
- Required environment variables (in .env, loaded automatically):
  - ELSEVIER_KEY: your Elsevier API key (required for paper_finder)
  - X_ELS_INSTTOKEN: optional institutional token
  - DEEPSEEK_API_KEY: DeepSeek API key (required for LLM calls)
- Optional knobs:
  - WRITE_PAPER=1 to enable paper drafting step
  - HITL_CONFIRM=0/HITL_AUTO_APPROVE=1 to auto‑continue gates
  - DATASET=isic|cifar10 (default is isic)
  - ALLOW_FALLBACK_DATASET=true to use torchvision FakeData if real dataset missing (requires torchvision)
  - ALLOW_DATASET_DOWNLOAD=true to allow CIFAR10 download (requires torchvision)
  - TIME_BUDGET_SEC=0 (0 disables budgeting) • PARALLEL_RUNS=false • MUTATE_K=0 • REPEAT_N=1
  - LLM cache: LLM_CACHE=true • LLM_CACHE_DIR=.cache/llm

YAML Config Override (project‑level)

- Place a `config.yaml` (or `.yml` / `.json`) at repo root to override settings globally. YAML takes precedence over env/defaults.
- Example:
  
  dataset:
    name: cifar10
    path: data/cifar10
    allow_fallback: true     # allow torchvision FakeData when real data missing
    allow_download: true     # allow CIFAR10 download
  
- You can also point to a custom file with `CONFIG_FILE=path/to/config.yaml`.

Dependency Install (pip)

- Install core dependencies:
  - cmd.exe /C "python -m pip install -r requirements.txt --disable-pip-version-check"
  - Note: torch/torchvision are skipped by design on Windows in requirements.txt; experiments will gracefully run in stub mode without them.

Compile Sources (sanity)

- Compile entire repo to bytecode to catch syntax errors:
  - cmd.exe /C "python -c ""import sys,compileall; sys.exit(0 if compileall.compile_dir('.', force=True, quiet=1) else 1)"""
  - If quoting is finicky in your shell, you can instead run: cmd.exe /C "python dev\compile_all.py" after creating a small helper.

Run Tests (pytest)

- Execute unit tests:
  - cmd.exe /C "python -m pytest -q"
  - Tests do not hit external services and should pass locally.

Pipeline Runbook (step‑by‑step)

1) Find and download papers (Elsevier + DeepSeek relevance)
   - Pre‑req: set ELSEVIER_KEY and DEEPSEEK_API_KEY in .env
   - cmd.exe /C "python paper_finder.py"
   - Outputs: abstract_screen_deepseek.csv and pdfs/*.pdf
   - If you prefer to skip this and use your own PDFs, just place them under pdfs/ and move on.

2) Summarize, critique, and synthesize novelty
   - cmd.exe /C "python -m agents.novelty"
   - Output: data/novelty_report.json

3) Derive a compact research plan (multi‑agent with offline fallback)
   - cmd.exe /C "python -m agents.planner"
   - Output: data/plan.json (plus data/plan_session.jsonl logs)

4) Iterative experiments (baseline/novelty/ablation + variants)
   - Defaults run safely even without torch/torchvision (stub mode). To run minimally with synthetic data, set ALLOW_FALLBACK_DATASET=true and install torchvision.
   - cmd.exe /C "python -m agents.iterate"
   - Outputs: runs/, experiments/, runs/summary.json, runs/best.json, runs/dashboard.html, runs/accuracy.png

5) Write paper draft (optional)
   - Enable with WRITE_PAPER=1 (or set env for the command):
   - cmd.exe /C "set WRITE_PAPER=1&& python -m agents.write_paper"
   - Outputs: paper/paper.md and paper/main.tex (+ refs.bib built from CSV if present). Attempts pdflatex if available.

One‑shot pipeline

- Run everything (4 steps; paper drafting only if WRITE_PAPER=1):
  - cmd.exe /C "python run_pipeline.py"

Validation and Artifacts

- Key outputs:
  - PDFs: pdfs/
  - Summaries: data/summaries/
  - Novelty report: data/novelty_report.json
  - Plan: data/plan.json
  - Runs + reports: runs/ (report.md, dashboard.html, best.json)
  - Paper draft: paper/ (if WRITE_PAPER=1)

Risks and Mitigations

- Keys missing: LLM calls or Elsevier queries will fail; set .env. LLM utilities warn on missing vars.
- Rate limits (Elsevier/DeepSeek): built‑in backoff; re‑run later; keep MAX_KEPT modest in paper_finder.
- No dataset / no torchvision: agents_iterate falls back to stub mode; set ALLOW_FALLBACK_DATASET=true + install torchvision for FakeData.
- Windows path issues: prefer cmd.exe /C python … to avoid path translation; keep working directory at repo root.

Rollback / Safe switches

- Offline planning: agents_planner.make_plan_offline() auto‑engages on LLM error.
- Skip paper writing: omit WRITE_PAPER.
- Compute budget: set TIME_BUDGET_SEC to halt long iterations. Use REPEAT_N=1 and MUTATE_K=0 for minimal runs.
- Clear caches: remove .cache/llm if needed to re‑query LLM.

Notes

- Domain‑agnostic: Set your project `goal` in `config.yaml`; prompts adapt to it. No hard‑coded domain defaults.
- Dataset via YAML: `dataset.kind` imagefolder|cifar10|custom, `dataset.path`, and `dataset.splits` control train/val/test. CIFAR10 uses a train/val split (val_fraction) and official test for evaluation.
- Evaluation: selection uses validation; reports highlight test metrics by default.
- .env is loaded by python‑dotenv; do not commit secrets.
- LLM caching is enabled by default (LLM_CACHE=true); cached JSON lives under .cache/llm/.
- MLflow is optional and disabled by default. Enable with MLFLOW_ENABLED=true and set MLFLOW_TRACKING_URI if desired.
