import os
import sys
import time
import subprocess
import shutil
from pathlib import Path


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

    # 1) Find + download papers (cap 40 kept, dedupe seeded from CSV)
    run_step(py, ["paper_finder.py"])  # call the script explicitly

    # 2) Summaries + novelty synthesis
    run_step(py, ["-m", "agents.novelty"])  # packaged agent

    # 2.5) Planner: derive a compact plan.json from novelty report
    run_step(py, ["-m", "agents.planner"])  # packaged agent

    # 3) Iterative experiments (baseline/novelty/ablation + refine)
    run_step(py, ["-m", "agents.iterate"])  # packaged agent

    # 4) Optional: write paper draft (enable with WRITE_PAPER=1)
    if str(os.getenv("WRITE_PAPER", "")).lower() in {"1", "true", "yes"}:
        run_step(py, ["-m", "agents.write_paper"])  # packaged agent

    print("\n[PIPELINE COMPLETE] Outputs:\n- PDFs: pdfs/\n- Summaries: data/summaries/\n- Novelty report: data/novelty_report.json\n- Plan: data/plan.json\n- Runs+reports: runs/\n- Paper draft: paper/ (if WRITE_PAPER=1)")


if __name__ == "__main__":
    main()
