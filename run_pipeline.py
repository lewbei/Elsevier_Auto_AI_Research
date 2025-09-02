import os
import time
from pathlib import Path
from dotenv import load_dotenv
from lab.logging_utils import get_log_level


HERE = Path(__file__).parent.resolve()


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
    # Load .env explicitly from repo root
    try:
        load_dotenv(HERE / ".env")
    except Exception:
        pass
    # Tag this pipeline run for LLM cost tracking
    run_id = time.strftime("run_%Y%m%d-%H%M%S")
    os.environ["LLM_RUN_ID"] = run_id
    print(f"[INFO] LOG_LEVEL={get_log_level()}")
    # Ensure output dirs exist
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "downloads").mkdir(exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)

    # Programmatic orchestration (no CLI): leverage agents.orchestrator
    from agents import orchestrator as _orch
    _orch.main()

    # Write LLM cost usage summary for this run
    summary = _aggregate_llm_usage(run_id)
    try:
        import json
        runs_dir = HERE / "runs"
        runs_dir.mkdir(exist_ok=True)
        (runs_dir / f"llm_cost_{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (runs_dir / "llm_cost.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[COST] LLM usage summary written: runs/llm_cost_{run_id}.json")
    except Exception:
        print("[COST] Failed to write LLM usage summary.")

    print("\n[PIPELINE COMPLETE] Outputs:\n- PDFs: pdfs/\n- Summaries: data/summaries/\n- Novelty report: data/novelty_report.json\n- Plan: data/plan.json\n- Runs+reports: runs/\n- Paper draft: paper/ (if WRITE_PAPER=1)\n- LLM cost summary: runs/llm_cost.json")


if __name__ == "__main__":
    main()

