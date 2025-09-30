import json
import pathlib
import time
import os
from typing import Any, Dict, List


from lab.config import get, get_bool
from lab.logging_utils import append_jsonl, is_verbose, vprint
from lab.codegen_llm import write_generated_aug_from_llm  # type: ignore
from lab.code_edit_loop import run_codegen_editor  # type: ignore
from lab.experiment_runner import run_experiment

try:
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore



DATA_DIR = pathlib.Path("data")
RUNS_DIR = pathlib.Path("runs")
NOVELTY_PATH = DATA_DIR / "novelty_report.json"
PLAN_PATH = DATA_DIR / "plan.json"


def _now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _load_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ensure_dirs() -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def _persona_notes(novelty: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    enable = get_bool("pipeline.interactive.personas.enable", False) or (
        str(os.getenv("INTERACTIVE_PERSONAS", "")).lower() in {"1", "true", "yes"}
    )
    if not enable or DialogueManager is None:
        return []
    try:
        dm = DialogueManager()
        dm.post(
            "User",
            (
                "We will generate tiny training hooks (transforms/head/spec tweaks) under tight compute.\n"
                "Provide role-specific, concrete guidance: key transforms or head tweaks to try first, safe ranges (lr/steps), and quick validation checks."
            ),
        )
        ctx = json.dumps({"novelty": novelty, "plan": plan}, ensure_ascii=False)
        dm.post("User", f"Context: {ctx}")
        notes: List[str] = []
        for role in ["PhD", "Professor", "SW", "ML"]:
            r = dm.step_role(role, prompt="Give 3 short actionable bullets to guide code updates. Be concrete and within constraints.")
            notes.append(f"[{role}] {r.get('text','')}")
        return notes
    except Exception:
        return []


def main() -> None:
    _ensure_dirs()
    if not NOVELTY_PATH.exists():
        print(f"[SKIP] Interactive: missing novelty report at {NOVELTY_PATH}")
        return

    novelty = _load_json(NOVELTY_PATH)
    plan = _load_json(PLAN_PATH) if PLAN_PATH.exists() else {}

    run_id = f"interactive_{_now_str()}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    session_log = DATA_DIR / "interactive_session.jsonl"

    # Persona advice (optional)
    notes = _persona_notes(novelty, plan)
    if notes:
        append_jsonl(session_log, {"role": "Personas", "content": notes})

    # Novelty description for codegen
    desc = ""
    try:
        desc = str(((novelty.get("themes") or [{}])[0]).get("name") or "")
    except Exception:
        desc = ""
    if not desc:
        desc = str(plan.get("novelty_focus") or "jitter rotate")

    # Try LLM-generated augmentation for novelty
    try:
        write_generated_aug_from_llm(desc, extra_context=json.dumps({"plan": plan}, ensure_ascii=False))
    except Exception:
        pass

    # Interactive code loop settings
    steps = int(get("pipeline.interactive.max_steps", 2) or 2)
    target_delta = float(get("pipeline.interactive.target_delta", 0.005) or 0.005)

    context = {
        "plan": plan,
        "notes": notes,
    }
    last_err = ""
    results: List[Dict[str, Any]] = []

    for i in range(1, steps + 1):
        ctx = dict(context)
        if last_err:
            ctx["last_error"] = last_err
        append_jsonl(session_log, {"role": "system", "content": {"iteration": i, "ctx": ctx}})

        # Generate/patch training hooks file and validate
        ok = run_codegen_editor(description=desc, extra_context=json.dumps(ctx, ensure_ascii=False), max_steps=1)
        if not ok:
            append_jsonl(session_log, {"role": "editor", "content": "generation_failed"})
        else:
            append_jsonl(session_log, {"role": "editor", "content": "generation_ok"})

        # Build a minimal spec and execute experiment
        spec = {
            "title": f"interactive_step_{i}",
            "dataset_path": get("dataset.path", "data/isic") or "data/isic",
            "input_size": int(get("interactive.input_size", 224) or 224),
            "model": "resnet18",
            "epochs": 1,
            "batch_size": int(get("interactive.batch_size", 16) or 16),
            "lr": float(get("interactive.lr", 1e-3) or 1e-3),
            "max_train_steps": int(get("interactive.max_train_steps", 50) or 50),
            "seed": 42,
            "novelty_component": {"description": desc, "enabled": True},
        }
        try:
            result = run_experiment(spec)
            results.append({"iter": i, "spec": spec, "result": result})
            (out_dir / f"step_{i}_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
            (out_dir / f"step_{i}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            append_jsonl(session_log, {"role": "runner", "content": {"iter": i, "metrics": result.get("metrics", {})}})
            if is_verbose():
                try:
                    vprint(f"Interactive step {i} metrics: {result.get('metrics', {})}")
                except Exception:
                    pass
            # Early stop if accuracy improved significantly vs 0 baseline
            acc = float(result.get("metrics", {}).get("val_accuracy", 0.0) or 0.0)
            if acc >= target_delta:
                break
            last_err = ""
        except Exception as exc:
            last_err = f"Experiment error: {exc}"
            append_jsonl(session_log, {"role": "runner", "content": {"iter": i, "error": last_err}})

    # Summary
    summary = {
        "run_id": run_id,
        "steps": len(results),
        "target_delta": target_delta,
        "results": results,
    }
    (RUNS_DIR / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Interactive session written under {out_dir}")


if __name__ == "__main__":
    main()
