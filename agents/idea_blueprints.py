import json
import pathlib
from typing import Any, Dict, List
import os

from dotenv import load_dotenv
from lab.config import get, get_bool
from utils.llm_utils import chat_json, LLMError


load_dotenv()

DATA_DIR = pathlib.Path("data")
NOVELTY_PATH = DATA_DIR / "novelty_report.json"
IDEAS_DIR = DATA_DIR / "ideas"
BLUEPRINTS_PATH = IDEAS_DIR / "blueprints.json"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    IDEAS_DIR.mkdir(parents=True, exist_ok=True)


def _pick_ideas(final: Dict[str, Any], k: int) -> List[str]:
    ideas = []
    # Prefer uniqueness-filtered ideas when present
    for key in ("unique_ideas", "new_ideas"):
        xs = final.get(key) or []
        if isinstance(xs, list):
            ideas.extend([str(x) for x in xs])
        if len(ideas) >= k:
            break
    return ideas[:k]


def _expand_one_idea(idea: str, goal: str) -> Dict[str, Any]:
    system = (
        "You are a research planner. Convert a novelty idea into a compact, runnable blueprint under tight compute. "
        "Return strictly JSON with the schema provided."
    )
    user = {
        "goal": goal,
        "idea": idea,
        "constraints": {"epochs": 1, "max_train_steps": 100, "compute": "CPU/single GPU"},
        "schema": {
            "idea": "string",
            "problem": "string",
            "objectives": ["string"],
            "contributions": ["string"],
            "research_questions": ["string"],
            "methodology": {
                "novelty_kind": "architecture|training_objective|data|evaluation|augmentation|optimizer",
                "spec_hint": "string",
                "datasets": ["string"],
                "metrics": ["string"],
                "baselines": ["string"],
                "ablations": ["string"],
                "compute": {"epochs": 1, "max_train_steps": 100}
            },
            "risks": ["string"],
            "success_criteria": [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
            "stopping_rules": ["string"]
        },
        "rules": [
            "Scalars stay strings; lists stay arrays of strings.",
            "Keep items concise (<240 chars).",
            "Spec_hint should be an actionable one-liner to seed experiments.",
        ],
    }
    model = get("pipeline.idea_blueprints.model", None)
    profile = get("pipeline.idea_blueprints.llm", None)
    js = chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)
    # Ensure required keys and defaults
    bp = {
        "idea": js.get("idea") or idea,
        "problem": js.get("problem") or "",
        "objectives": js.get("objectives") or [],
        "contributions": js.get("contributions") or [],
        "research_questions": js.get("research_questions") or [],
        "methodology": js.get("methodology") or {
            "novelty_kind": "",
            "spec_hint": "",
            "datasets": [],
            "metrics": ["val_accuracy"],
            "baselines": ["resnet18 minimal"],
            "ablations": [],
            "compute": {"epochs": 1, "max_train_steps": 100},
        },
        "risks": js.get("risks") or [],
        "success_criteria": js.get("success_criteria") or [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
        "stopping_rules": js.get("stopping_rules") or ["stop if novelty beats baseline by >=0.5pp"],
    }
    return bp


def main() -> None:
    _ensure_dirs()
    if not NOVELTY_PATH.exists():
        print(f"[ERR] Missing novelty report at {NOVELTY_PATH}")
        return
    try:
        final = json.loads(NOVELTY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] Failed to read novelty report: {exc}")
        return

    top_k = int(get("pipeline.idea_blueprints.top_k", 3) or 3)
    ideas = _pick_ideas(final, top_k)
    if not ideas:
        print("[INFO] No ideas found in novelty report to expand.")
        return
    goal = str(get("project.goal", "your goal") or "your goal")

    blueprints: List[Dict[str, Any]] = []
    for i, idea in enumerate(ideas, start=1):
        try:
            bp = _expand_one_idea(idea, goal)
        except LLMError as exc:
            print(f"[WARN] LLM blueprint failed for idea {i}: {exc}")
            bp = {"idea": idea, "problem": "", "objectives": [], "contributions": [], "research_questions": [],
                  "methodology": {"novelty_kind": "", "spec_hint": "", "datasets": [], "metrics": ["val_accuracy"],
                                  "baselines": ["resnet18 minimal"], "ablations": [], "compute": {"epochs": 1, "max_train_steps": 100}},
                  "risks": [], "success_criteria": [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
                  "stopping_rules": ["stop if novelty beats baseline by >=0.5pp"]}
        blueprints.append(bp)
        # Write per-idea markdown for readability
        md_lines = [f"# Idea {i}", "", f"Novelty: {bp.get('idea','')}", "", "## Problem", bp.get("problem", ""),
                    "", "## Objectives", *[f"- {x}" for x in (bp.get("objectives") or [])],
                    "", "## Contributions", *[f"- {x}" for x in (bp.get("contributions") or [])],
                    "", "## Research Questions", *[f"- {x}" for x in (bp.get("research_questions") or [])],
                    "", "## Methodology",
                    f"- Kind: {bp.get('methodology',{}).get('novelty_kind','')}",
                    f"- Spec hint: {bp.get('methodology',{}).get('spec_hint','')}",
                    f"- Datasets: {', '.join(bp.get('methodology',{}).get('datasets',[]))}",
                    f"- Metrics: {', '.join(bp.get('methodology',{}).get('metrics',[]))}",
                    f"- Baselines: {', '.join(bp.get('methodology',{}).get('baselines',[]))}",
                    f"- Ablations: {', '.join(bp.get('methodology',{}).get('ablations',[]))}",
                    "", "## Risks", *[f"- {x}" for x in (bp.get("risks") or [])],
                    "", "## Success Criteria", *[f"- {x.get('metric','')}: +{x.get('delta_vs_baseline','')} vs baseline" for x in (bp.get("success_criteria") or [])],
                    "", "## Stopping Rules", *[f"- {x}" for x in (bp.get("stopping_rules") or [])]
                    ]
        (IDEAS_DIR / f"idea_{i}.md").write_text("\n".join(md_lines), encoding="utf-8")

    BLUEPRINTS_PATH.write_text(json.dumps({"blueprints": blueprints}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote idea blueprints to {BLUEPRINTS_PATH}")


if __name__ == "__main__":
    main()
