import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import re
import tempfile
import shutil
from datetime import datetime

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

# Optional dependency: jsonschema for strict validation
try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore

# Schemas: narrow but realistic; allow additionalProperties to avoid brittleness
NOVELTY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "themes": {"type": "array"},
        "novelty_ideas": {"type": "array"},
        "unique_ideas": {"type": "array"},
        "new_ideas": {"type": "array"},
        "new_ideas_detailed": {"type": "array"},
        "problems": {"type": "array"},
        "objectives": {"type": "array"},
        "contributions": {"type": "array"},
        "research_questions": {"type": "array"},
        "citations": {"type": "array"},
    },
    "additionalProperties": True,
}

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective": {"type": "string", "minLength": 1},
        "hypotheses": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "success_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "delta_vs_baseline": {"type": ["number", "integer"]},
                },
                "required": ["metric", "delta_vs_baseline"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
        "datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "path": {"type": "string"}},
                "required": ["name", "path"],
                "additionalProperties": True,
            },
            "minItems": 1,
        },
        "baselines": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "novelty_focus": {"type": "string", "minLength": 1},
        "stopping_rules": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "why": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "why", "steps"],
                "additionalProperties": True,
            },
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"risk": {"type": "string"}, "mitigation": {"type": "string"}},
                "required": ["risk", "mitigation"],
                "additionalProperties": True,
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "milestones": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "objective",
        "hypotheses",
        "success_criteria",
        "datasets",
        "baselines",
        "novelty_focus",
        "stopping_rules",
        "tasks",
        "risks",
        "assumptions",
        "milestones",
    ],
    "additionalProperties": True,
}


def _strip_json_comments(s: str) -> str:
    # remove /* ... */ and // ... comments without touching strings
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(^|\s)//.*?$", r"\1", s, flags=re.M)
    return s


def load_json_tolerant(path: pathlib.Path) -> Dict[str, Any]:
    """Strict JSON first; on failure strip comments and trailing commas and retry."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except Exception:
        cleaned = _strip_json_comments(raw)
        cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)  # trailing commas
        try:
            return json.loads(cleaned)
        except Exception as exc:
            raise ValueError(f"Invalid JSON at {path}: {exc}") from exc


def validate_json(data: Dict[str, Any], schema: Dict[str, Any], name: str) -> None:
    """Use jsonschema if present; else shallow required-keys check; fail fast."""
    if jsonschema is not None:
        jsonschema.validate(instance=data, schema=schema)  # type: ignore[attr-defined]
        return
    required = set(schema.get("required", []))
    missing = [k for k in required if k not in data or data.get(k) in (None, "", [], {})]
    if missing:
        raise ValueError(f"{name} missing required keys: {missing}")


def atomic_write_json(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically with timestamped backup of previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="plan_write_"))
    tmppath = tmpdir / path.name
    tmppath.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, backup)
    shutil.move(str(tmppath), str(path))
    shutil.rmtree(tmpdir, ignore_errors=True)


def _extract_outline(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact outline block from the novelty report (robust to missing keys)."""
    def _as_list(x):
        xs: List[str] = []
        if isinstance(x, list):
            xs = [str(i).strip() for i in x]
        elif x:
            xs = [str(x).strip()]
        # drop empties & dedupe preserving order
        seen = set()
        out: List[str] = []
        for s in xs:
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
        return out
    return {
        "problems": _as_list(novelty.get("problems")),
        "objectives": _as_list(novelty.get("objectives")),
        "contributions": _as_list(novelty.get("contributions")),
        "research_questions": _as_list(novelty.get("research_questions")),
        "citations": novelty.get("citations") or [],
    }


def _pick_candidates(novelty: Dict[str, Any], k: int = 3) -> list[Dict[str, Any]]:
    """Pick top-k novelty idea candidates with dedupe and normalized shape."""
    cands: list[Dict[str, Any]] = []
    seen = set()

    def _push(d: Dict[str, Any]) -> None:
        title = (d.get("title") or "").strip()
        if not title:
            return
        key = title.lower()
        if key in seen:
            return
        seen.add(key)
        cands.append({
            "id": str(d.get("id") or len(cands) + 1),
            "title": title,
            "novelty_kind": str(d.get("novelty_kind") or "").strip(),
            "spec_hint": str(d.get("spec_hint") or "").strip(),
            "method": str(d.get("method") or "").strip(),
            "risks": str(d.get("risks") or "").strip(),
            "eval_plan": [str(x).strip() for x in (d.get("eval_plan") or []) if str(x).strip()],
            "compute_budget": str(d.get("compute_budget") or "").strip(),
            "derived_from_titles": [str(x).strip() for x in (d.get("derived_from_titles") or []) if str(x).strip()],
            "delta_vs_prior": str(d.get("delta_vs_prior") or "").strip(),
        })

    # Prefer structured ideas first
    for key in ("novelty_ideas", "new_ideas_detailed"):
        xs = novelty.get(key) or []
        if isinstance(xs, list):
            for it in xs:
                if isinstance(it, dict):
                    _push(it)
        if cands:
            break

    # Fallback to strings if no structured ideas present
    if not cands:
        for key in ("unique_ideas", "new_ideas"):
            xs = novelty.get(key) or []
            if isinstance(xs, list):
                for s in xs:
                    _push({"title": str(s)})
            if cands:
                break

    return cands[: max(0, int(k))]


def build_plan(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the LLM to turn the novelty report into a compact, actionable plan.
    The plan is intentionally small so it is cheap and deterministic.
    """
    topic = str(get("project.goal", "your task") or "your task")
    outline = _extract_outline(novelty)
    candidates = _pick_candidates(novelty, k=3)
    system = (
        "You are a Principal Investigator preparing an actionable, resource-aware research plan from a novelty report.\n"
        f"Target: iterative experiment loop for {topic}.\n"
        "Requirements: return STRICT JSON with the schema below; keep runnable on CPU or single GPU with <=1 epoch per run and small step budgets.\n"
        "Grounding: use the provided novelty_outline (problems/objectives/contributions/research_questions) and choose ONE novelty_focus from novelty_candidates.\n"
        "Map the chosen candidate's eval_plan/spec_hint into concrete tasks/steps suitable for our environment.\n"
        "Include ALL of the following keys, each NON-EMPTY: objective, hypotheses, success_criteria (metric + delta vs baseline), datasets (name + path), baselines, novelty_focus, stopping_rules, tasks (name/why/steps), risks (risk/mitigation), assumptions, milestones.\n"
        "Metrics may include mAP@0.5, FP_per_image, precision@IoU0.5, IoU, Dice, AUC, ECE, or val_accuracy — choose metrics that match the novelty.\n"
        "Style: concise but unambiguous; avoid vague language; prefer values and thresholds.\n"
        "Self-check: validate keys/types/constraints match the schema; fix mismatches and return JSON only."
    )
    user_payload = {
        "novelty_report": novelty,
        "novelty_outline": outline,
        "novelty_candidates": candidates,
        "constraints": {
            "budget": "<= 1 epoch per run, <= 100 steps",
            "dataset_default": f"{dataset_name('ISIC').upper()} under {dataset_path_for()} with train/ and val/",
        },
        "output_schema": {
            "objective": "string",
            "hypotheses": ["string"],
            "success_criteria": [
                {"metric": "mAP@0.5", "delta_vs_baseline": 0.03}
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
            "milestones": ["string"]
        }
    }
    profile = get("pipeline.planner.llm", None)
    js = chat_json_cached(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0, profile=profile)

    if not isinstance(js, dict):
        raise ValueError("LLM returned non-dict JSON for plan.")
    # STRICT: validate raw response shape — no coercion/fallback
    validate_json(js, PLAN_SCHEMA, "plan_from_llm")

    # Minimal shape enforcement
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]

    # Suggested novelty_focus from first candidate (no defaults). Empty is allowed and will be caught by validation.
    suggested_focus = ""
    try:
        if candidates:
            c0 = candidates[0]
            parts = [str(c0.get("novelty_kind") or "").strip(), str(c0.get("spec_hint") or "").strip()]
            parts = [p for p in parts if p]
            suggested_focus = ": ".join(parts) if parts else str(c0.get("title") or "").strip()
    except Exception:
        suggested_focus = ""

    plan = {
        "objective": str(js.get("objective") or ""),
        "hypotheses": js.get("hypotheses") or [],
        "success_criteria": js.get("success_criteria") or [],
        "datasets": js.get("datasets") or [],
        "baselines": js.get("baselines") or [],
        "novelty_focus": str(js.get("novelty_focus") or suggested_focus or ""),
        "stopping_rules": js.get("stopping_rules") or [],
        "tasks": js.get("tasks") or [],
        "risks": js.get("risks") or [],
        "assumptions": js.get("assumptions") or [],
        "milestones": js.get("milestones") or [],
    }
    validate_json(plan, PLAN_SCHEMA, "final_plan_from_build")
    return plan


def make_plan_offline(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Offline fallback that derives a compact plan without LLM access."""
    themes = novelty.get("themes") or []
    outline = _extract_outline(novelty)
    candidates = _pick_candidates(novelty, k=1)
    objective = "Evaluate a simple novelty"
    novelty_focus = "augmentation and/or classifier head"
    tasks = []
    # Prefer first structured candidate to seed focus and tasks
    if candidates:
        c0 = candidates[0]
        nk = c0.get("novelty_kind") or ""
        sh = c0.get("spec_hint") or ""
        ttl = c0.get("title") or ""
        novelty_focus = (f"{nk}: {sh}" if nk and sh else (ttl or nk or sh or novelty_focus))
        ev = c0.get("eval_plan") or []
        if isinstance(ev, list) and ev:
            tasks = [{"name": f"Eval step {i}", "why": "from novelty eval_plan", "steps": [str(s)]} for i, s in enumerate(ev, start=1)]
    # Lightly use themes if no candidate
    if not candidates and themes and isinstance(themes, list):
        try:
            first = themes[0]
            if isinstance(first, dict) and first.get("name"):
                novelty_focus = str(first.get("name"))
        except Exception:
            pass
    # Hypotheses seeded from research_questions/objectives when available
    hyps = outline.get("research_questions") or outline.get("objectives") or [
        "A small augmentation or head change improves val acc by ~0.5pp"
    ]
    return {
        "objective": objective,
        "hypotheses": hyps[:3],
        "success_criteria": [{"metric": "val_accuracy", "delta_vs_baseline": 0.005}],
        "datasets": [{"name": dataset_name("DATA").upper(), "path": dataset_path_for()}],
        "baselines": ["resnet18 minimal"],
        "novelty_focus": novelty_focus,
        "stopping_rules": ["stop if novelty beats baseline by >=0.5pp"],
        "tasks": tasks,
        "assumptions": [],
        "milestones": [],
        "risks": [],
    }


def run_planning_session(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Run a small multi-agent planning session and return the final plan.
    Logs conversation turns into data/plan_session.jsonl. Falls back offline on error.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_log = DATA_DIR / "plan_session.jsonl"
    transcript_path = DATA_DIR / "plan_transcript.md"
    # Progress printing control (default ON unless explicitly disabled)
    try:
        _progress = True
        try:
            _progress = bool(get_bool("pipeline.planner.print_progress", True))
        except Exception:
            _progress = True
        envp = str(os.getenv("PLANNER_PROGRESS", "")).lower()
        if envp in {"0", "false", "no", "off"}:
            _progress = False
        elif envp in {"1", "true", "yes", "on"}:
            _progress = True
    except Exception:
        _progress = True
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
        if _progress:
            try:
                print(f"[PLANNER] Persona notes recorded: {len(persona_notes)}")
            except Exception:
                pass

        # Prepare outline/candidates for all roles
        outline = _extract_outline(novelty)
        candidates = _pick_candidates(novelty, k=3)

        # PI draft
        pi_system = load_prompt("planner_pi_system") or (
            "You are the PI (Principal Investigator). Draft a clear, minimal, and resource-aware plan grounded in the novelty report.\n"
            "Use novelty_outline (problems/objectives/contributions/research_questions) and choose ONE novelty from novelty_candidates as the novelty_focus.\n"
            "Translate the chosen candidate's eval_plan/spec_hint into concrete tasks/steps that fit <=1 epoch and <=100 steps.\n"
            "Output MUST include ALL keys, each non-empty: objective, hypotheses, success_criteria (metric + delta vs baseline), datasets (name + path), baselines, novelty_focus, stopping_rules, tasks (name/why/steps), risks (risk/mitigation), assumptions, milestones.\n"
            "Metrics may include mAP@0.5, FP_per_image, precision@IoU0.5, IoU, Dice, AUC, ECE, or val_accuracy — pick those matching the novelty.\n"
            "Be concrete, specify thresholds, and avoid ambiguous phrasing.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        pi_user = json.dumps({
            "novelty_report": novelty,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "PI_input", "system": pi_system, "user": json.loads(pi_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        pi = chat_json_cached(pi_system, pi_user, temperature=0.0, model=model, profile=profile)
        append_jsonl(session_log, {"role": "PI", "content": pi})
        if _progress:
            try:
                print(
                    "[PLANNER] PI plan: objective='%s', hypotheses=%d, datasets=%d, tasks=%d"
                    % (
                        str(pi.get("objective") or "").strip()[:80],
                        len(pi.get("hypotheses") or []),
                        len(pi.get("datasets") or []),
                        len(pi.get("tasks") or []),
                    )
                )
            except Exception:
                pass

        # Engineer refinement
        eng_system = load_prompt("planner_engineer_system") or (
            "You are the Engineer. Refine the PI plan into runnable specifications with exact parameters and safe ranges.\n"
            "Validate dataset paths, choose realistic metrics, ensure baselines are minimal, and convert the selected novelty candidate into short step-by-step tasks.\n"
            "Respect constraints (<=1 epoch, small steps). Return JSON with ALL keys present and non-empty: objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules, tasks, risks, assumptions, milestones.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        eng_user = json.dumps({
            "pi_plan": pi,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Engineer_input", "system": eng_system, "user": json.loads(eng_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        eng = chat_json_cached(eng_system, eng_user, temperature=0.0, model=model, profile=profile)
        append_jsonl(session_log, {"role": "Engineer", "content": eng})
        if _progress:
            try:
                print(
                    "[PLANNER] Engineer plan: baselines=%d, tasks=%d, risks=%d"
                    % (
                        len(eng.get("baselines") or []),
                        len(eng.get("tasks") or []),
                        len(eng.get("risks") or []),
                    )
                )
            except Exception:
                pass

        # Reviewer check
        rev_system = load_prompt("planner_reviewer_system") or (
            "You are the Reviewer. Check for clarity, feasibility, minimalism, and internal consistency.\n"
            "Ensure the novelty_focus corresponds to one candidate and tasks map to its eval_plan/spec_hint.\n"
            "Finalize a lean plan that fits the compute budget; keep thresholds and dataset paths explicit.\n"
            "Return final plan JSON with ALL keys present and non-empty: objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules, tasks, risks, assumptions, milestones.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        rev_user = json.dumps({
            "engineer_plan": eng,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Reviewer_input", "system": rev_system, "user": json.loads(rev_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        rev = chat_json_cached(rev_system, rev_user, temperature=0.0, model=model, profile=profile)

        # Strict validation warnings (no mutation here)
        def _warn_if_invalid(tag: str, obj: Any) -> None:
            try:
                if isinstance(obj, dict):
                    validate_json(obj, PLAN_SCHEMA, f"{tag} plan")
            except Exception as e:
                print(f"[WARN] {tag} produced invalid plan shape: {e}")

        _warn_if_invalid("PI", pi)
        _warn_if_invalid("Engineer", eng)
        _warn_if_invalid("Reviewer", rev)
        append_jsonl(session_log, {"role": "Reviewer", "content": rev})
        if _progress:
            try:
                print(
                    "[PLANNER] Reviewer plan: objective='%s', baselines=%d, stopping_rules=%d"
                    % (
                        str(rev.get("objective") or "").strip()[:80],
                        len(rev.get("baselines") or []),
                        len(rev.get("stopping_rules") or []),
                    )
                )
            except Exception:
                pass

        # carry over fields from Engineer/PI if Reviewer misses them (no offline fallback; same session only)
        def _carry(key, default):
            if rev.get(key):
                return rev.get(key)
            if eng.get(key):
                return eng.get(key)
            if pi.get(key):
                return pi.get(key)
            return default

        final = {
            "objective": str(_carry("objective", "")),
            "hypotheses": _carry("hypotheses", []),
            "success_criteria": _carry("success_criteria", []),
            "datasets": _carry("datasets", []),
            "baselines": _carry("baselines", []),
            "novelty_focus": str(_carry("novelty_focus", "")),
            "stopping_rules": _carry("stopping_rules", []),
            "tasks": _carry("tasks", []),
            "risks": _carry("risks", []),
            "assumptions": _carry("assumptions", []),
            "milestones": _carry("milestones", []),
        }
        # Strict validation of the final composed plan
        validate_json(final, PLAN_SCHEMA, "final_plan")
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
    except LLMError as e:
        # No offline fallback: fail loudly so the caller can see the cause
        append_jsonl(session_log, {"role": "error", "content": f"LLM planning failed: {e}"})
        raise


def main() -> None:
    _ensure_dirs()
    if not REPORT_PATH.exists():
        print(f"[ERR] Missing novelty report at {REPORT_PATH}. Run agents.novelty first.")
        return
    try:
        novelty = load_json_tolerant(REPORT_PATH)
        validate_json(novelty, NOVELTY_SCHEMA, "novelty_report")
    except Exception as exc:
        print(f"[ERR] Failed to read/validate novelty report: {exc}")
        return

    try:
        # Optional high-level novelty stats before planning
        try:
            themes = len(novelty.get("themes") or [])
            ideas = len(novelty.get("novelty_ideas") or [])
            rq = len(novelty.get("research_questions") or [])
            print(f"[PLANNER] Novelty report: themes={themes}, ideas={ideas}, research_q={rq}")
        except Exception:
            pass
        plan = run_planning_session(novelty)
    except Exception as exc:
        print(f"[ERR] Planning session failed without fallback: {exc}")
        return

    # HITL confirmation gate
    hitl = get_bool("pipeline.hitl.confirm", False) or (str(os.getenv("HITL_CONFIRM", "")).lower() in {"1", "true", "yes"})
    auto = get_bool("pipeline.hitl.auto_approve", False) or (str(os.getenv("HITL_AUTO_APPROVE", "")).lower() in {"1", "true", "yes"})
    if hitl and not auto:
        pending = DATA_DIR / "plan_pending.json"
        pending.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[PENDING] Plan written to {pending}. Set HITL_AUTO_APPROVE=1 to finalize.")
        return

    try:
        atomic_write_json(PLAN_PATH, plan)
    except Exception as exc:
        print(f"[ERR] Failed to write plan.json atomically: {exc}")
        return
    if is_verbose():
        try:
            vprint("Plan: " + json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception:
            pass
    print(f"[DONE] Wrote plan to {PLAN_PATH}")


if __name__ == "__main__":
    main()
