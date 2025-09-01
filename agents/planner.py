import json
import pathlib
from typing import Any, Dict
import os

from dotenv import load_dotenv
from utils.llm_utils import chat_json_cached, LLMError
from lab.logging_utils import append_jsonl, is_verbose, vprint
from lab.prompt_overrides import load_prompt
from lab.config import dataset_name, dataset_path_for, get, get_bool
try:
    # Optional: multi‑persona advice to inform planning
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore


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
        "You are a Principal Investigator preparing an actionable, resource-aware research plan from a novelty report.\n"
        f"Target: iterative experiment loop for {topic}.\n"
        "Requirements: return STRICT JSON with the schema below; keep runnable on CPU or single GPU with <=1 epoch per run and small step budgets.\n"
        "Include: a crisp objective, explicit hypotheses, concrete success criteria (metric + delta vs baseline), datasets (name + path), minimal baselines, a tightly scoped novelty focus (one main lever), and practical stopping rules.\n"
        "Style: concise but unambiguous; avoid vague language; prefer values and thresholds.\n"
        "Self-check: Before responding, validate keys/types/constraints match the schema; fix mismatches and return JSON only."
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
            "tasks": [{"name": "string", "why": "string", "steps": ["string"]}],
            "risks": [{"risk": "string", "mitigation": "string"}],
            "assumptions": ["string"],
            "milestones": ["string"],
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
        # Optional transparency fields
        "tasks": js.get("tasks") or [],
        "risks": js.get("risks") or [],
        "assumptions": _as_list(js.get("assumptions") or []),
        "milestones": _as_list(js.get("milestones") or []),
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
    transcript_path = DATA_DIR / "plan_transcript.md"
    try:
        # Optional multi‑persona advisory pass (Professor/PhD/SW/ML)
        persona_notes = []
        try:
            enable_personas = (
                get_bool("pipeline.planner.personas.enable", False)
                or (str(os.getenv("PLANNER_PERSONAS", "")).lower() in {"1", "true", "yes"})
            )
        except Exception:
            enable_personas = False
        if enable_personas and DialogueManager is not None:
            try:
                dm = DialogueManager()
                # Prime with a single user message summarizing novelty
                dm.post("User", "Please provide concise planning notes for this novelty report.")
                # Collect short advice from each role
                for role in ["PhD", "Professor", "SW", "ML"]:
                    resp = dm.step_role(role, prompt="Provide 3 short bullet points to guide the plan. Be concrete.")
                    persona_notes.append(f"[{role}] {resp.get('text','')}")
            except Exception:
                persona_notes = []
        if persona_notes:
            append_jsonl(session_log, {"role": "Personas", "notes": persona_notes})

        # PI draft
        pi_system = load_prompt("planner_pi_system") or (
            "You are the PI (Principal Investigator). Draft a clear, minimal, and resource-aware plan from the novelty report and persona notes.\n"
            "Balance novelty with feasibility under tight compute (<=1 epoch, <=100 steps).\n"
            "Output must include: objective, concise hypotheses, success_criteria (val_accuracy delta vs baseline), datasets (name + path), minimal baselines, a focused novelty component, stopping_rules, and optional tasks/risks/assumptions/milestones.\n"
            "Be concrete, specify thresholds, and avoid ambiguous phrasing.\n"
            "Self-check: Before responding, validate keys/types/constraints; return only JSON."
        )
        pi_user = json.dumps({"novelty_report": novelty, "persona_notes": persona_notes}, ensure_ascii=False)
        append_jsonl(session_log, {"role": "PI_input", "system": pi_system, "user": json.loads(pi_user)})
        pi = chat_json_cached(pi_system, pi_user, temperature=0.0)
        append_jsonl(session_log, {"role": "PI", "content": pi})

        # Engineer refinement
        eng_system = load_prompt("planner_engineer_system") or (
            "You are the Engineer. Refine the PI plan into runnable specifications with exact parameters and safe ranges.\n"
            "Validate dataset paths, choose realistic metrics, ensure baselines are minimal, and convert tasks to short step-by-step actions.\n"
            "Respect constraints (<=1 epoch, small steps). Return JSON with objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules, tasks, risks, assumptions, milestones.\n"
            "Self-check: Before responding, validate keys/types/constraints; return only JSON."
        )
        eng_user = json.dumps({"pi_plan": pi, "persona_notes": persona_notes}, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Engineer_input", "system": eng_system, "user": json.loads(eng_user)})
        eng = chat_json_cached(eng_system, eng_user, temperature=0.0)
        append_jsonl(session_log, {"role": "Engineer", "content": eng})

        # Reviewer check
        rev_system = load_prompt("planner_reviewer_system") or (
            "You are the Reviewer. Check for clarity, feasibility, minimalism, and internal consistency.\n"
            "Remove ambiguous or non-actionable elements, flag risks and unstated assumptions, and finalize a lean plan that fits the compute budget.\n"
            "Return final plan JSON with the same keys; keep thresholds and dataset paths explicit.\n"
            "Self-check: Before responding, validate keys/types/constraints; return only JSON."
        )
        rev_user = json.dumps({"engineer_plan": eng, "persona_notes": persona_notes}, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Reviewer_input", "system": rev_system, "user": json.loads(rev_user)})
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
        # Write a human-readable transcript for transparency
        try:
            parts = [
                "# Planning Transcript",
                "",
                "## Persona Notes",
                *(persona_notes or ["(none)"]),
                "",
                "## PI Plan (draft)",
                json.dumps(pi, ensure_ascii=False, indent=2),
                "",
                "## Engineer Plan (refined)",
                json.dumps(eng, ensure_ascii=False, indent=2),
                "",
                "## Reviewer Plan (final)",
                json.dumps(final, ensure_ascii=False, indent=2),
                "",
            ]
            transcript_path.write_text("\n".join(parts), encoding="utf-8")
        except Exception:
            pass
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
    if is_verbose():
        try:
            vprint("Plan: " + json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception:
            pass
    print(f"[DONE] Wrote plan to {PLAN_PATH}")


if __name__ == "__main__":
    main()
