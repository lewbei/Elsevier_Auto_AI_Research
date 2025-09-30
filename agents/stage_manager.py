import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from lab.logging_utils import append_jsonl
from agents.iterate import iterate
from agents.stage_manifest import RUNS as RUNS_DIR, DATA as DATA_DIR, NOVELTY_REPORT, stage_ready

def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summary_goal() -> bool:
    s = _read_json(RUNS_DIR / "summary.json")
    return bool(s.get("goal_reached", False))


def _load_novelty() -> Dict[str, Any]:
    ready, reason = stage_ready("planner")
    if not ready:
        raise SystemExit(reason)
    return _read_json(NOVELTY_REPORT)


def run_stage(name: str, novelty: Dict[str, Any], max_iters: int, env_overrides: Dict[str, str] | None = None) -> None:
    """Execute a single stage of the research pipeline with environment overrides.
    
    Args:
        name: Stage name for logging
        novelty: Novelty report data
        max_iters: Maximum iterations for this stage
        env_overrides: Temporary environment variable overrides
    """
    session_log = RUNS_DIR / "stage_session.jsonl"
    start = time.time()
    append_jsonl(session_log, {"event": "stage_start", "name": name, "ts": start})
    # Apply env overrides (volatile)
    bak: Dict[str, str | None] = {}
    try:
        if env_overrides:
            for k, v in env_overrides.items():
                bak[k] = os.getenv(k)
                os.environ[k] = str(v)
        iterate(novelty, max_iters=max_iters)
    finally:
        # Restore environment
        for k, v in bak.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    dur = time.time() - start
    append_jsonl(session_log, {"event": "stage_end", "name": name, "duration_sec": dur, "goal_reached": _summary_goal()})


def main() -> None:
    """Execute the complete staged research pipeline."""
    novelty = _load_novelty()
    global_budget = float(os.getenv("STAGE_BUDGET_SEC", "0") or 0)
    stage_start = time.time()

    # Stage 1: setup (sanity run, small)
    run_stage("setup", novelty, max_iters=1, env_overrides={"REPEAT_N": "1", "MUTATE_K": "0"})
    if _summary_goal():
        return

    # Stage 2: refine (beam mutations, small repeats if requested)
    run_stage("refine", novelty, max_iters=1, env_overrides={"REPEAT_N": os.getenv("REPEAT_N", "1"), "MUTATE_K": os.getenv("MUTATE_K", "2")})
    if _summary_goal():
        return

    # Stage 3: ablate (ablation is embedded in iterate; one more pass)
    run_stage("ablate", novelty, max_iters=1, env_overrides={"REPEAT_N": os.getenv("REPEAT_N", "1"), "MUTATE_K": "0"})

    # Budget check
    if global_budget and (time.time() - stage_start) > global_budget:
        append_jsonl(RUNS_DIR / "stage_session.jsonl", {"event": "budget_exhausted", "elapsed": time.time() - stage_start})


if __name__ == "__main__":
    main()
