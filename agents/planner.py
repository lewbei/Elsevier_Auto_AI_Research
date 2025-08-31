import json
import pathlib
from typing import Any, Dict
import os

from dotenv import load_dotenv
from utils.llm_utils import chat_json_cached, LLMError
from lab.logging_utils import append_jsonl
from lab.prompt_overrides import load_prompt
from lab.config import dataset_name, dataset_path_for, get, get_bool


DATA_DIR = pathlib.Path("data")
REPORT_PATH = DATA_DIR / "novelty_report.json"
PLAN_PATH = DATA_DIR / "plan.json"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_plan(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM to turn the novelty report into a compact, actionable plan.
    The plan is intentionally small so it is cheap and deterministic.
    """
    topic = str(get("project.goal", "your task") or "your task")
    system = (
        "You are a Principal Investigator. Convert the provided novelty report into a compact research plan "
        f"for an iterative experiment loop on {topic}. Keep it strictly JSON and small."
    )
    user_payload = {
        "novelty_report": novelty,
        "constraints": {
            "budget": "<= 1 epoch per run, <= 100 steps",
            "dataset_default": f"{dataset_name('ISIC').upper()} under {dataset_path_for()} with train/ and val/",
        },
        "output_schema": {
            "objective": "string",
            "hypotheses": ["string"],
            "success_criteria": [
                {"metric": "val_accuracy", "delta_vs_baseline": 0.005}
            ],
            "datasets": [
                {"name": "ISIC", "path": "data/isic"}
            ],
            "baselines": ["string"],
            "novelty_focus": "string",
            "stopping_rules": ["string"],
        }
    }
    js = chat_json_cached(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0)

    # Minimal shape enforcement
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]

    plan = {
        "objective": str(js.get("objective") or "Evaluate a simple novelty"),
        "hypotheses": _as_list(js.get("hypotheses")),
        "success_criteria": js.get("success_criteria") or [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
        "datasets": js.get("datasets") or [{"name": dataset_name("DATA").upper(), "path": dataset_path_for()}],
        "baselines": _as_list(js.get("baselines") or ["resnet18 minimal"]),
        "novelty_focus": str(js.get("novelty_focus") or "augmentation and/or classifier head"),
        "stopping_rules": _as_list(js.get("stopping_rules") or ["stop if novelty beats baseline by >=0.5pp"]),
    }
    return plan


def make_plan_offline(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Offline fallback that derives a compact plan without LLM access."""
    themes = novelty.get("themes") or []
    objective = "Evaluate a simple novelty"
    novelty_focus = "augmentation and/or classifier head"
    if themes and isinstance(themes, list):
        try:
            first = themes[0]
            if isinstance(first, dict) and first.get("name"):
                novelty_focus = str(first.get("name"))
        except Exception:
            pass
    return {
        "objective": objective,
        "hypotheses": ["A small augmentation or head change improves val acc by ~0.5pp"],
        "success_criteria": [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
        "datasets": [{"name": dataset_name("DATA").upper(), "path": dataset_path_for()}],
        "baselines": ["resnet18 minimal"],
        "novelty_focus": novelty_focus,
        "stopping_rules": ["stop if novelty beats baseline by >=0.5pp"],
    }


def run_planning_session(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Run a small multi-agent planning session and return the final plan.
    Logs conversation turns into data/plan_session.jsonl. Falls back offline on error.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_log = DATA_DIR / "plan_session.jsonl"
    try:
        # PI draft
        pi_system = load_prompt("planner_pi_system") or "You are the Principal Investigator. Propose a compact research plan JSON."
        pi_user = json.dumps({"novelty_report": novelty}, ensure_ascii=False)
        pi = chat_json_cached(pi_system, pi_user, temperature=0.0)
        append_jsonl(session_log, {"role": "PI", "content": pi})

        # Engineer refinement
        eng_system = load_prompt("planner_engineer_system") or (
            "You are the Engineer. Tighten the plan to runnable specs and constraints (<=1 epoch, small steps). "
            "Return JSON with objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules."
        )
        eng_user = json.dumps({"pi_plan": pi}, ensure_ascii=False)
        eng = chat_json_cached(eng_system, eng_user, temperature=0.0)
        append_jsonl(session_log, {"role": "Engineer", "content": eng})

        # Reviewer check
        rev_system = load_prompt("planner_reviewer_system") or (
            "You are the Reviewer. Validate the plan for clarity, risks, and minimality. "
            "Return the final plan JSON (same schema), making only necessary edits."
        )
        rev_user = json.dumps({"engineer_plan": eng}, ensure_ascii=False)
        rev = chat_json_cached(rev_system, rev_user, temperature=0.0)
        append_jsonl(session_log, {"role": "Reviewer", "content": rev})

        # Shape to our expected fields
        final = {
            "objective": str(rev.get("objective") or ""),
            "hypotheses": rev.get("hypotheses") or [],
            "success_criteria": rev.get("success_criteria") or [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
            "datasets": rev.get("datasets") or [{"name": dataset_name("DATA").upper(), "path": dataset_path_for()}],
            "baselines": rev.get("baselines") or ["resnet18 minimal"],
            "novelty_focus": str(rev.get("novelty_focus") or ""),
            "stopping_rules": rev.get("stopping_rules") or ["stop if novelty beats baseline by >=0.5pp"],
        }
        return final
    except LLMError:
        # Offline fallback
        fb = make_plan_offline(novelty)
        append_jsonl(session_log, {"role": "offline_fallback", "content": fb})
        return fb


def main() -> None:
    _ensure_dirs()
    if not REPORT_PATH.exists():
        print(f"[ERR] Missing novelty report at {REPORT_PATH}. Run agents_novelty.py first.")
        return
    try:
        novelty = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] Failed to read novelty report: {exc}")
        return

    try:
        plan = run_planning_session(novelty)
    except Exception as exc:
        print(f"[WARN] Planning session failed: {exc}. Falling back to offline.")
        plan = make_plan_offline(novelty)

    # HITL confirmation gate
    hitl = get_bool("pipeline.hitl.confirm", False) or (str(os.getenv("HITL_CONFIRM", "")).lower() in {"1", "true", "yes"})
    auto = get_bool("pipeline.hitl.auto_approve", False) or (str(os.getenv("HITL_AUTO_APPROVE", "")).lower() in {"1", "true", "yes"})
    if hitl and not auto:
        pending = DATA_DIR / "plan_pending.json"
        pending.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[PENDING] Plan written to {pending}. Set HITL_AUTO_APPROVE=1 to finalize.")
        return

    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote plan to {PLAN_PATH}")


if __name__ == "__main__":
    main()
