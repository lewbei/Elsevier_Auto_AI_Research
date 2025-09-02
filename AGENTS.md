Agents Overview (Programmatic, No CLI)

Principles
- No CLI, no argparse, no offline fallback. Use Python entrypoints directly.
- Config via .env and optional YAML (config.yaml). YAML takes precedence; env vars override as needed.
- Windows + Git‑Bash friendly; paths use pathlib; no shell dependencies required.

Environment
- Keys: ELSEVIER_KEY (paper_finder), DEEPSEEK_API_KEY (LLM), optional X_ELS_INSTTOKEN.
- Common toggles (YAML preferred, env optional):
  - pipeline.skip.{find_papers|summaries|novelty|planner|iterate}
  - pipeline.always_fresh, pipeline.max_iters
  - pipeline.codegen.enable, pipeline.codegen.editor.enable
  - dataset.{name|path|kind|splits|allow_fallback|allow_download}
  - llm.{provider|model|default|custom|use}

Config File
- Place config.yaml at repo root (or set CONFIG_FILE). Example:

  dataset:
    name: isic
    path: data/isic
    kind: imagefolder
  pipeline:
    skip:
      find_papers: true
      novelty: false
      planner: false
      iterate: false
    max_iters: 2
    codegen:
      enable: true
      editor:
        enable: false

Stage Contracts (Python Usage)
- Summaries (agents/summarize.py)
  - Function: process_pdfs(pdf_dir, out_dir, max_pages=0, max_chars=0, chunk_size=20000, timeout=60, max_tries=4, model=None, profile=None, verbose=True, skip_existing=True) -> int
  - Input: PDFs under pdf_dir
  - Output: JSON files under data/summaries/

- Novelty (agents/novelty.py)
  - Function: main() reads data/summaries/*.json, writes data/novelty_report.json
  - Optional: persona discussion and facets mining (guarded by config)

- Planner (agents/planner.py)
  - Function: main() reads data/novelty_report.json, writes data/plan.json
  - Strict JSON normalization and validation; logs a transcript under data/plan_session.jsonl

- Iterate (agents/iterate.py)
  - Function: iterate(novelty_dict, max_iters=2) performs baseline/novelty/ablation + variants
  - Outputs: runs/summary.json, runs/best.json, runs/dashboard.html, accuracy.png

- Interactive (agents/interactive.py) [optional]
  - Function: main() uses personas + editor to synthesize training hooks and run small tests

- Write Paper (agents/write_paper.py) [optional]
  - Function: main() composes paper/paper.md and paper/main.tex from artifacts (LLM drafting optional)

Programmatic Orchestration
- Preferred: import agents.orchestrator and call main() to run staged pipeline with programmatic internals (no CLI behavior required).
- Alternatively, call stages individually in your own script to customize control flow.

Artifacts
- PDFs: pdfs/
- Summaries: data/summaries/
- Novelty report: data/novelty_report.json
- Plan: data/plan.json (+ data/plan_session.jsonl)
- Runs + reports: runs/ (report.md, dashboard.html, best.json, summary.json)
- Paper: paper/ (paper.md, main.tex)

Notes
- GPT‑5 family omits temperature by design (handled in utils/llm_utils).
- No offline fallbacks are performed by default. If you need deterministic placeholders, add them explicitly behind opt-in flags.

