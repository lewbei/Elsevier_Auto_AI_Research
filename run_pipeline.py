import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from lab.config import get_bool


HERE = Path(__file__).parent.resolve()


def win_python_path() -> str:
    # Prefer user-specified path via env; else default to requested path; else sys.executable
    env_path = os.getenv("WIN_PYTHON")
    if env_path and Path(env_path).exists():
        return env_path
    default = r"C:\Users\lewka\miniconda3\envs\deep_learning\python.exe"
    if Path(default).exists():
        return default
    return sys.executable


def run_step(py: str, args: list[str]) -> None:
    cmd = [py] + args
    print(f"\n[RUN] {' '.join(cmd)}")
    start = time.time()
    try:
        subprocess.run(cmd, cwd=str(HERE), check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Step failed with exit code {exc.returncode}: {' '.join(cmd)}")
    finally:
        dur = time.time() - start
        print(f"[DONE] {' '.join(args)} in {dur:.1f}s")


def main() -> None:
    py = win_python_path()
    # Quick env checks (don’t fail early on missing insttoken)
    missing = []
    if not os.getenv("ELSEVIER_KEY"):
        missing.append("ELSEVIER_KEY")
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        print(f"[WARN] Missing environment vars: {', '.join(missing)}. Steps may fail.")

    # Ensure output dirs exist
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "downloads").mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    pdf_dir = HERE / "pdfs"
    try:
        pdf_count = sum(1 for p in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else 0
    except Exception:
        pdf_count = 0

    # 0) Optional full orchestrator: runs all phases end-to-end and returns
    orch_enable = get_bool("pipeline.orchestrator.enable", False) or (str(os.getenv("ORCHESTRATOR_ENABLE", "")).lower() in {"1", "true", "yes"})
    if orch_enable:
        run_step(py, ["-m", "agents.orchestrator"])  # orchestrates all phases
        print("[PIPELINE COMPLETE] (orchestrator)")
        return

    # 1) Find + download papers (cap 40 kept, dedupe seeded from CSV)
    skip_find = get_bool("pipeline.skip.find_papers", False) or (str(os.getenv("SKIP_FIND_PAPERS", "")).lower() in {"1", "true", "yes"})
    auto_skip = pdf_count >= 40
    if skip_find or auto_skip:
        reason = "env SKIP_FIND_PAPERS" if skip_find else f"found {pdf_count} PDFs under pdfs/"
        print(f"[SKIP] Step 1 (paper_finder) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.paper_finder"])  # packaged agent

    # 2) Summaries + novelty synthesis
    novelty_path = HERE / "data" / "novelty_report.json"
    skip_novelty = get_bool("pipeline.skip.novelty", False) or (str(os.getenv("SKIP_NOVELTY", "")).lower() in {"1", "true", "yes"})
    if skip_novelty or novelty_path.exists():
        reason = "env SKIP_NOVELTY" if skip_novelty else f"found {novelty_path}"
        print(f"[SKIP] Step 2 (novelty) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.novelty"])  # packaged agent

    # 2.5) Planner: derive a compact plan.json from novelty report
    plan_path = HERE / "data" / "plan.json"
    skip_planner = get_bool("pipeline.skip.planner", False) or (str(os.getenv("SKIP_PLANNER", "")).lower() in {"1", "true", "yes"})
    if skip_planner or plan_path.exists():
        reason = "env SKIP_PLANNER" if skip_planner else f"found {plan_path}"
        print(f"[SKIP] Step 2.5 (planner) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.planner"])  # packaged agent

    # 3) Iterative experiments (baseline/novelty/ablation + refine)
    # 3a) Optional interactive multi-persona + executable code loop
    skip_interactive = get_bool("pipeline.skip.interactive", False) or (str(os.getenv("SKIP_INTERACTIVE", "")).lower() in {"1", "true", "yes"})
    enable_interactive = get_bool("pipeline.interactive.enable", False) or (str(os.getenv("INTERACTIVE_ENABLE", "")).lower() in {"1", "true", "yes"})
    if enable_interactive and not skip_interactive:
        run_step(py, ["-m", "agents.interactive"])  # optional interactive stage
    skip_iter = get_bool("pipeline.skip.iterate", False) or (str(os.getenv("SKIP_ITERATE", "")).lower() in {"1", "true", "yes"})
    if skip_iter:
        print("[SKIP] Step 3 (iterate) skipped (env SKIP_ITERATE).")
    else:
        run_step(py, ["-m", "agents.iterate"])  # packaged agent

    # 4) Optional: write paper draft (enable with WRITE_PAPER=1)
    write_paper = get_bool("pipeline.write_paper", False) or (str(os.getenv("WRITE_PAPER", "")).lower() in {"1", "true", "yes"})
    if write_paper:
        run_step(py, ["-m", "agents.write_paper"])  # packaged agent

    print("\n[PIPELINE COMPLETE] Outputs:\n- PDFs: pdfs/\n- Summaries: data/summaries/\n- Novelty report: data/novelty_report.json\n- Plan: data/plan.json\n- Runs+reports: runs/\n- Paper draft: paper/ (if WRITE_PAPER=1)")


if __name__ == "__main__":
    main()
