import os
import sys
import time
import subprocess
import shutil
from pathlib import Path
from lab.config import get_bool, get
from lab.logging_utils import get_log_level
from dotenv import load_dotenv


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


def run_step(py: str, args: list[str], *, stage: str) -> None:
    cmd = [py] + args
    print(f"\n[RUN] {' '.join(cmd)}")
    start = time.time()
    # Inherit env but tag stage and run id
    env = dict(os.environ)
    env["LLM_STAGE"] = stage
    try:
        subprocess.run(cmd, cwd=str(HERE), check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Step failed with exit code {exc.returncode}: {' '.join(cmd)}")
    finally:
        dur = time.time() - start
        print(f"[DONE] {' '.join(args)} in {dur:.1f}s")


def _aggregate_llm_usage(run_id: str) -> dict:
    usage_path = HERE / "logs" / "llm" / "usage.jsonl"
    summary = {
        "run_id": run_id,
        "events": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "by_stage": {},
        "by_model": {},
    }
    if not usage_path.exists():
        return summary
    try:
        import json
        with usage_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("run_id") != run_id:
                    continue
                pt = int(rec.get("prompt_tokens") or 0)
                ct = int(rec.get("completion_tokens") or 0)
                cost = float(rec.get("cost") or 0.0)
                model = str(rec.get("model") or "")
                stage = str(rec.get("stage") or "")
                summary["events"] += 1
                summary["prompt_tokens"] += pt
                summary["completion_tokens"] += ct
                summary["total_tokens"] += int(rec.get("total_tokens") or (pt + ct))
                summary["cost"] += cost
                if stage:
                    s = summary["by_stage"].setdefault(stage, {"events": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0})
                    s["events"] += 1
                    s["prompt_tokens"] += pt
                    s["completion_tokens"] += ct
                    s["total_tokens"] += int(rec.get("total_tokens") or (pt + ct))
                    s["cost"] += cost
                if model:
                    m = summary["by_model"].setdefault(model, {"events": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0})
                    m["events"] += 1
                    m["prompt_tokens"] += pt
                    m["completion_tokens"] += ct
                    m["total_tokens"] += int(rec.get("total_tokens") or (pt + ct))
                    m["cost"] += cost
    except Exception:
        return summary
    return summary


def main() -> None:
    # Load environment variables from .env at repo root (and parents)
    # so that API keys and flags are visible to this orchestrator as well.
    try:
        # Be explicit about location so it works even when invoked from elsewhere
        load_dotenv(HERE / ".env")
    except Exception:
        pass
    py = win_python_path()
    # Tag this pipeline run for LLM cost tracking
    run_id = time.strftime("run_%Y%m%d-%H%M%S")
    os.environ["LLM_RUN_ID"] = run_id
    # Fresh-run override: when enabled, do not auto-skip based on existing artifacts
    fresh_run = get_bool("pipeline.always_fresh", False) or (str(os.getenv("ALWAYS_FRESH", "")).lower() in {"1", "true", "yes"})
    print(f"[INFO] LOG_LEVEL={get_log_level()}")
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
        run_step(py, ["-m", "agents.orchestrator"], stage="orchestrator")  # orchestrates all phases
        print("[PIPELINE COMPLETE] (orchestrator)")
        return

    # 1) Find + download papers (cap 40 kept, dedupe seeded from CSV)
    skip_find = get_bool("pipeline.skip.find_papers", False) or (str(os.getenv("SKIP_FIND_PAPERS", "")).lower() in {"1", "true", "yes"})
    auto_skip = (pdf_count >= 40) and (not fresh_run)
    if skip_find or auto_skip:
        reason = "env SKIP_FIND_PAPERS" if skip_find else (f"found {pdf_count} PDFs under pdfs/" + (" (fresh-run override disabled)" if not fresh_run else ""))
        print(f"[SKIP] Step 1 (paper_finder) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.paper_finder"], stage="paper_finder")  # packaged agent

    # 1.5) Per-paper summaries (extract + summarize + critic)
    sum_dir = HERE / "data" / "summaries"
    try:
        sum_count = sum(1 for p in sum_dir.glob("*.json")) if sum_dir.exists() else 0
    except Exception:
        sum_count = 0
    skip_summaries = get_bool("pipeline.skip.summaries", False) or (str(os.getenv("SKIP_SUMMARIES", "")).lower() in {"1", "true", "yes"})
    auto_skip_sum = (sum_count > 0) and (not fresh_run)
    if skip_summaries or auto_skip_sum:
        reason = "env SKIP_SUMMARIES" if skip_summaries else (f"found {sum_count} summaries under data/summaries/" + (" (fresh-run override disabled)" if not fresh_run else ""))
        print(f"[SKIP] Step 1.5 (summaries) skipped ({reason}).")
    else:
        # Programmatic summaries using stage configuration from config.yaml
        try:
            from agents.summarize import process_pdfs
            # Read stage config
            max_pages = int(get("pipeline.summarize.max_pages", 0) or 0)
            max_chars = int(get("pipeline.summarize.max_chars", 0) or 0)
            chunk_size = int(get("pipeline.summarize.pass1_chunk", 20000) or 20000)
            timeout = int(get("pipeline.summarize.timeout", 60) or 60)
            max_tries = int(get("pipeline.summarize.max_tries", 4) or 4)
            model = get("pipeline.summarize.model", None)
            if isinstance(model, str) and not model.strip():
                model = None
            profile = get("pipeline.summarize.llm", None)
            # Prefer full-text mode for GPT-5 mini
            default_model = get("llm.default", None)
            prof = str(profile or "").strip().lower()
            md = str(model or "").strip().lower()
            dm = str(default_model or "").strip().lower()
            if (md.startswith("gpt-5-")) or (not md and prof == "default" and dm.startswith("gpt-5-")):
                chunk_size = -1  # force single full-text pass
            # Verbose if progress or detail enabled
            verbose = bool(get("pipeline.summarize.progress", True) or get("pipeline.summarize.detail", False))

            # Decide whether to skip existing summaries
            skip_existing_cfg = get("pipeline.summarize.skip_existing", None)
            if skip_existing_cfg is None:
                skip_existing = (not fresh_run)
            else:
                skip_existing = bool(skip_existing_cfg)

            wrote = process_pdfs(
                pdf_dir=str(pdf_dir),
                out_dir=str(sum_dir),
                max_pages=max_pages,
                max_chars=max_chars,
                chunk_size=chunk_size,
                timeout=timeout,
                max_tries=max_tries,
                model=model,
                profile=profile,
                verbose=verbose,
                skip_existing=skip_existing,
            )
            print(f"[RUN] Summaries produced: {wrote}")
        except Exception as exc:
            raise SystemExit(f"Summaries step failed: {exc}")

    # 1.75) Literature review synthesized from summaries
    lit_path = HERE / "data" / "lit_review.md"
    skip_lit = get_bool("pipeline.skip.lit_review", False) or (str(os.getenv("SKIP_LIT_REVIEW", "")).lower() in {"1", "true", "yes"})
    auto_skip_lit = (lit_path.exists() and not fresh_run)
    if skip_lit or auto_skip_lit:
        reason = "env SKIP_LIT_REVIEW" if skip_lit else f"found {lit_path} (fresh-run override disabled)"
        print(f"[SKIP] Step 1.75 (lit_review) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.lit_review"], stage="lit_review")  # literature review from summaries

    # 2) Novelty synthesis from summaries
    novelty_path = HERE / "data" / "novelty_report.json"
    skip_novelty = get_bool("pipeline.skip.novelty", False) or (str(os.getenv("SKIP_NOVELTY", "")).lower() in {"1", "true", "yes"})
    if skip_novelty or (novelty_path.exists() and not fresh_run):
        reason = "env SKIP_NOVELTY" if skip_novelty else f"found {novelty_path} (fresh-run override disabled)"
        print(f"[SKIP] Step 2 (novelty) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.novelty"], stage="novelty")  # packaged agent

    # 2.5) Planner: derive a compact plan.json from novelty report
    plan_path = HERE / "data" / "plan.json"
    skip_planner = get_bool("pipeline.skip.planner", False) or (str(os.getenv("SKIP_PLANNER", "")).lower() in {"1", "true", "yes"})
    if skip_planner or (plan_path.exists() and not fresh_run):
        reason = "env SKIP_PLANNER" if skip_planner else f"found {plan_path} (fresh-run override disabled)"
        print(f"[SKIP] Step 2.5 (planner) skipped ({reason}).")
    else:
        run_step(py, ["-m", "agents.planner"], stage="planner")  # packaged agent

    # 3) Iterative experiments (baseline/novelty/ablation + refine)
    # 3a) Optional interactive multi-persona + executable code loop
    skip_interactive = get_bool("pipeline.skip.interactive", False) or (str(os.getenv("SKIP_INTERACTIVE", "")).lower() in {"1", "true", "yes"})
    enable_interactive = get_bool("pipeline.interactive.enable", False) or (str(os.getenv("INTERACTIVE_ENABLE", "")).lower() in {"1", "true", "yes"})
    if enable_interactive and not skip_interactive:
        run_step(py, ["-m", "agents.interactive"], stage="interactive")  # optional interactive stage
    skip_iter = get_bool("pipeline.skip.iterate", False) or (str(os.getenv("SKIP_ITERATE", "")).lower() in {"1", "true", "yes"})
    if skip_iter:
        print("[SKIP] Step 3 (iterate) skipped (env SKIP_ITERATE).")
    else:
        run_step(py, ["-m", "agents.iterate"], stage="iterate")  # packaged agent

    # 4) Optional: write paper draft (enable with WRITE_PAPER=1)
    write_paper = get_bool("pipeline.write_paper", False) or (str(os.getenv("WRITE_PAPER", "")).lower() in {"1", "true", "yes"})
    if write_paper:
        run_step(py, ["-m", "agents.write_paper"], stage="write_paper")  # packaged agent

    # Write LLM cost usage summary for this run
    summary = _aggregate_llm_usage(run_id)
    try:
        import json
        runs_dir = HERE / "runs"
        runs_dir.mkdir(exist_ok=True)
        # Per-run file and latest pointer
        (runs_dir / f"llm_cost_{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (runs_dir / "llm_cost.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[COST] LLM usage summary written: runs/llm_cost_{run_id}.json")
    except Exception:
        print("[COST] Failed to write LLM usage summary.")

    print("\n[PIPELINE COMPLETE] Outputs:\n- PDFs: pdfs/\n- Summaries: data/summaries/\n- Novelty report: data/novelty_report.json\n- Plan: data/plan.json\n- Runs+reports: runs/\n- Paper draft: paper/ (if WRITE_PAPER=1)\n- LLM cost summary: runs/llm_cost.json")


if __name__ == "__main__":
    main()
