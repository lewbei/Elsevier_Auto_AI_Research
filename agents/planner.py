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
from utils.llm_utils import LLM_PROVIDER, LLM_MODEL
from utils.llm_utils import LLM_CHAT_URL
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


def _print_schema_errors(tag: str, obj: Any, schema: Dict[str, Any]) -> None:
    """Best-effort detailed schema diagnostics for invalid objects."""
    try:
        if jsonschema is None:
            print(f"[SCHEMA] {tag}: jsonschema not installed; limited diagnostics. Offending value type={type(obj).__name__}.")
            return
        from jsonschema import Draft7Validator  # type: ignore
        v = Draft7Validator(schema)
        errs = list(v.iter_errors(obj))
        if not errs:
            print(f"[SCHEMA] {tag}: no detailed errors collected.")
            return
        print(f"[SCHEMA] {tag}: {len(errs)} issue(s) found. Showing first 5:")
        for i, e in enumerate(errs[:5], start=1):
            path = ".".join([str(p) for p in list(e.path)]) or "<root>"
            print(f"  {i}) path={path} message={e.message}")
    except Exception:
        try:
            print(f"[SCHEMA] {tag}: failed to produce detailed diagnostics.")
        except Exception:
            pass


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


def _normalize_plan_shape(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce a loosely-typed LLM plan object into the strict PLAN_SCHEMA types.
    This does NOT drop information; it concatenates rich fields into strings
    where the schema expects strings, and preserves original structure in logs.
    """
    if not isinstance(js, dict):
        return {}

    out: Dict[str, Any] = dict(js)  # shallow copy

    # ---- baselines: array[string] ----
    baselines = out.get("baselines")
    norm_baselines: List[str] = []
    if isinstance(baselines, list):
        for b in baselines:
            if isinstance(b, str):
                s = b.strip()
                if s:
                    norm_baselines.append(s)
            elif isinstance(b, dict):
                name = str(b.get("name") or "").strip()
                desc = str(b.get("description") or "").strip()
                notes = str(b.get("notes") or "").strip()
                exact_params = b.get("exact_params")
                param_str = ""
                if isinstance(exact_params, dict) and exact_params:
                    try:
                        # compact, stable ordering
                        param_str = " | params=" + json.dumps(exact_params, sort_keys=True, ensure_ascii=False)
                    except Exception:
                        param_str = ""
                concat = " : ".join([x for x in [name, desc] if x]).strip(" :")
                concat = (concat + param_str + ((" | " + notes) if notes else "")).strip(" |")
                if concat:
                    norm_baselines.append(concat)
            else:
                try:
                    s = json.dumps(b, ensure_ascii=False)
                    if s:
                        norm_baselines.append(s)
                except Exception:
                    pass
    elif isinstance(baselines, str):
        s = baselines.strip()
        if s:
            norm_baselines.append(s)
    out["baselines"] = norm_baselines

    # ---- novelty_focus: string ----
    nf = out.get("novelty_focus")
    if isinstance(nf, dict):
        title = str(nf.get("title") or "").strip()
        nid = str(nf.get("id") or "").strip()
        kind = str(nf.get("novelty_kind") or "").strip()
        spec = str(nf.get("short_spec") or nf.get("spec_hint") or "").strip()
        # keep your voice and specificity
        pieces = [p for p in [nid, title, kind, spec] if p]
        out["novelty_focus"] = " — ".join(pieces) if pieces else ""
    elif nf is None:
        out["novelty_focus"] = ""
    else:
        out["novelty_focus"] = str(nf)

    # ---- risks: array[{risk, mitigation}] ----
    risks = out.get("risks")
    norm_risks: List[Dict[str, str]] = []
    if isinstance(risks, dict):
        for k, v in risks.items():
            if isinstance(v, dict):
                r_txt = str(v.get("risk") or k or "").strip()
                m_txt = str(v.get("mitigation") or "").strip()
            else:
                r_txt = str(k or "").strip()
                m_txt = str(v or "").strip()
            if r_txt or m_txt:
                norm_risks.append({"risk": r_txt, "mitigation": m_txt})
    elif isinstance(risks, list):
        for v in risks:
            if isinstance(v, dict):
                r_txt = str(v.get("risk") or "").strip()
                m_txt = str(v.get("mitigation") or "").strip()
                if r_txt or m_txt:
                    norm_risks.append({"risk": r_txt, "mitigation": m_txt})
            elif isinstance(v, str):
                norm_risks.append({"risk": v.strip(), "mitigation": ""})
    out["risks"] = norm_risks

    # ---- assumptions: array[string] ----
    assumptions = out.get("assumptions")
    if not isinstance(assumptions, list):
        # salvage common misplacement under 'risks'
        if isinstance(risks, dict) and isinstance(risks.get("assumptions"), list):
            assumptions = risks.get("assumptions")
        else:
            assumptions = []
    out["assumptions"] = [str(x).strip() for x in assumptions if str(x).strip()]

    # ---- milestones: array[string] ----
    ms = out.get("milestones")
    norm_ms: List[str] = []
    if isinstance(ms, list):
        for m in ms:
            if isinstance(m, str):
                s = m.strip()
                if s:
                    norm_ms.append(s)
            elif isinstance(m, dict):
                name = str(m.get("name") or "").strip()
                criteria = str(m.get("criteria") or "").strip()
                s = " : ".join([x for x in [name, criteria] if x]).strip(" :")
                if s:
                    norm_ms.append(s)
            else:
                try:
                    s = json.dumps(m, ensure_ascii=False)
                    if s:
                        norm_ms.append(s)
                except Exception:
                    pass
    elif isinstance(ms, str):
        s = ms.strip()
        if s:
            norm_ms.append(s)
    out["milestones"] = norm_ms

    # ---- datasets: array[{name, path}] ----
    ds = out.get("datasets")
    norm_ds: List[Dict[str, str]] = []
    if isinstance(ds, list):
        for d in ds:
            if isinstance(d, dict):
                name = str(d.get("name") or "").strip()
                path = str(d.get("path") or "").strip()
                if name or path:
                    norm_ds.append({"name": name, "path": path})
            elif isinstance(d, str):
                norm_ds.append({"name": d.strip(), "path": ""})
    elif isinstance(ds, dict):
        name = str(ds.get("name") or "").strip()
        path = str(ds.get("path") or "").strip()
        if name or path:
            norm_ds.append({"name": name, "path": path})
    out["datasets"] = norm_ds

    # ---- success_criteria: array[{metric, delta_vs_baseline:number}] ----
    sc = out.get("success_criteria")
    norm_sc: List[Dict[str, Union[str, float]]] = []
    if isinstance(sc, list):
        for s in sc:
            if not isinstance(s, dict):
                continue
            metric = str(s.get("metric") or "").strip()
            dvb = s.get("delta_vs_baseline")
            if isinstance(dvb, str):
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", dvb)
                dvb_num = float(m.group()) if m else 0.0
            else:
                try:
                    dvb_num = float(dvb)
                except Exception:
                    dvb_num = 0.0
            if metric:
                norm_sc.append({"metric": metric, "delta_vs_baseline": dvb_num})
    out["success_criteria"] = norm_sc

    # ---- hypotheses: array[string] ----
    hyps = out.get("hypotheses")
    if isinstance(hyps, list):
        out["hypotheses"] = [str(h).strip() for h in hyps if str(h).strip()]
    elif isinstance(hyps, str):
        out["hypotheses"] = [hyps.strip()]
    else:
        out["hypotheses"] = []

    # ---- stopping_rules: array[string] ----
    sr = out.get("stopping_rules")
    if isinstance(sr, list):
        out["stopping_rules"] = [str(x).strip() for x in sr if str(x).strip()]
    elif isinstance(sr, str):
        out["stopping_rules"] = [sr.strip()]
    else:
        out["stopping_rules"] = []

    # ---- tasks: array[{name, why, steps:[string]}] ----
    ts = out.get("tasks")
    norm_tasks: List[Dict[str, Any]] = []
    if isinstance(ts, list):
        for t in ts:
            if isinstance(t, dict):
                name = str(t.get("name") or "").strip()
                why = str(t.get("why") or "").strip()
                steps = t.get("steps")
                if isinstance(steps, list):
                    steps_list = [str(s).strip() for s in steps if str(s).strip()]
                elif isinstance(steps, str):
                    steps_list = [steps.strip()]
                elif steps is None:
                    steps_list = []
                else:
                    try:
                        steps_list = [json.dumps(steps, ensure_ascii=False)]
                    except Exception:
                        steps_list = []
                norm_tasks.append({"name": name, "why": why, "steps": steps_list})
            elif isinstance(t, str):
                norm_tasks.append({"name": t.strip(), "why": "", "steps": []})
    out["tasks"] = norm_tasks

    # ---- objective: string ----
    obj = out.get("objective")
    out["objective"] = str(obj or "").strip()

    return out


def _shape_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a structural diff focusing on types and presence.
    Example output: {"risks": {"from": "object", "to": "array"}, "baselines[0]": {"from": "object", "to": "string"}}
    """
    def t(x):  # type name
        if isinstance(x, list): return "array"
        if isinstance(x, dict): return "object"
        if isinstance(x, str): return "string"
        if isinstance(x, bool): return "boolean"
        if isinstance(x, (int, float)) and not isinstance(x, bool): return "number"
        if x is None: return "null"
        return type(x).__name__

    diff: Dict[str, Any] = {}

    # top-level keys
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        b = before.get(k)
        a = after.get(k)
        tb, ta = t(b), t(a)
        if tb != ta:
            diff[k] = {"from": tb, "to": ta}
        # spot check arrays first element
        if isinstance(b, list) and b:
            tb0 = t(b[0])
        else:
            tb0 = None
        if isinstance(a, list) and a:
            ta0 = t(a[0])
        else:
            ta0 = None
        if tb0 != ta0 and (tb0 is not None or ta0 is not None):
            diff[f"{k}[0]"] = {"from": tb0, "to": ta0}
    return diff
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
    """Pick top-k novelty idea candidates with dedupe and normalized shape, preferring higher tier1_score when present."""
    bucket: list[Dict[str, Any]] = []
    seen = set()

    def _push(d: Dict[str, Any]) -> None:
        title = (d.get("title") or "").strip()
        if not title:
            return
        key = title.lower()
        if key in seen:
            return
        seen.add(key)
        bucket.append({
            "id": str(d.get("id") or len(bucket) + 1),
            "title": title,
            "novelty_kind": str(d.get("novelty_kind") or "").strip(),
            "spec_hint": str(d.get("spec_hint") or "").strip(),
            "method": str(d.get("method") or "").strip(),
            "risks": str(d.get("risks") or "").strip(),
            "eval_plan": [str(x).strip() for x in (d.get("eval_plan") or []) if str(x).strip()],
            "compute_budget": str(d.get("compute_budget") or "").strip(),
            "derived_from_titles": [str(x).strip() for x in (d.get("derived_from_titles") or []) if str(x).strip()],
            "delta_vs_prior": str(d.get("delta_vs_prior") or "").strip(),
            "domain_tags": [str(x).strip().lower() for x in (d.get("domain_tags") or []) if str(x).strip()],
            "task_tags": [str(x).strip().lower() for x in (d.get("task_tags") or []) if str(x).strip()],
            "tier1_score": float(d.get("tier1_score", 0.0) or 0.0),
        })

    # Prefer structured ideas first
    for key in ("novelty_ideas", "new_ideas_detailed"):
        xs = novelty.get(key) or []
        if isinstance(xs, list):
            for it in xs:
                if isinstance(it, dict):
                    _push(it)
        if bucket:
            break

    # Fallback to strings if no structured ideas present
    if not bucket:
        for key in ("unique_ideas", "new_ideas"):
            xs = novelty.get(key) or []
            if isinstance(xs, list):
                for s in xs:
                    _push({"title": str(s)})
            if bucket:
                break
    # Prefer by tier1_score desc, then by title
    bucket.sort(key=lambda d: (d.get("tier1_score", 0.0), d.get("title", "")), reverse=True)
    topk = max(0, int(k))
    cands = bucket[: topk]
    # Coverage: ensure at least one domain-agnostic (generic) or multi-domain candidate if available
    try:
        def _is_generic_or_multi(c: Dict[str, Any]) -> bool:
            tags = c.get("domain_tags", []) or []
            if any(t == "generic" for t in tags):
                return True
            return len(set([t for t in tags if t])) >= 2
        if cands and not any(_is_generic_or_multi(c) for c in cands):
            for extra in bucket[topk:]:
                if _is_generic_or_multi(extra):
                    cands[-1] = extra
                    break
    except Exception:
        pass
    return cands


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
        "Map the chosen candidate's eval_plan/spec_hint into concrete tasks/steps suitable for our environment. Use delta_vs_prior to set baselines and at least one ablation.\n"
        "Include ALL of the following keys, each NON-EMPTY: objective, hypotheses, success_criteria (metric + delta vs baseline), datasets (name + path), baselines, novelty_focus, stopping_rules, tasks (name/why/steps), risks (risk/mitigation), assumptions, milestones.\n"
        "STRICT success_criteria SHAPE: an ARRAY of OBJECTS, each {metric: string, delta_vs_baseline: number}.\n"
        "Do NOT return 'primary_metric'/'secondary_metrics' objects; do NOT return strings like '+0.03 absolute' — use 0.03 for +3%.\n"
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
            "domain": "generic",
            "task_tags": [],
            "allowed_datasets": [dataset_name('ISIC')],
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
    model = get("pipeline.planner.model", None)
    try:
        _req_timeout = int(get("pipeline.planner.request_timeout", 180) or 180)
    except Exception:
        _req_timeout = 180
    try:
        _max_tries = int(get("pipeline.planner.request_retries", 4) or 4)
    except Exception:
        _max_tries = 4
    def _omit_temp_for_gpt5() -> bool:
        try:
            prov = str((get("llm.provider", LLM_PROVIDER) or LLM_PROVIDER)).lower()
            mod = str((model or get("llm.model", LLM_MODEL) or LLM_MODEL)).lower()
            return prov == "openai" and (mod.startswith("gpt-5-") or mod == "gpt-5")
        except Exception:
            return False
    if _omit_temp_for_gpt5():
        js = chat_json_cached(
            system,
            json.dumps(user_payload, ensure_ascii=False),
            model=model,
            profile=profile,
            timeout=_req_timeout,
            max_tries=_max_tries,
        )
    else:
        js = chat_json_cached(
            system,
            json.dumps(user_payload, ensure_ascii=False),
            temperature=0.0,
            model=model,
            profile=profile,
            timeout=_req_timeout,
            max_tries=_max_tries,
        )

    if not isinstance(js, dict):
        raise ValueError("LLM returned non-dict JSON for plan.")
    # Normalize first, then strict validate
    raw_js = js
    js = _normalize_plan_shape(js)
    try:
        validate_json(js, PLAN_SCHEMA, "plan_from_llm")
    except Exception as e:
        print(f"[ERR] plan_from_llm schema violation: {e}")
        _print_schema_errors("plan_from_llm", js, PLAN_SCHEMA)
        # diagnostics: shape diff (raw -> normalized)
        try:
            sd = _shape_diff(raw_js, js)
            append_jsonl(DATA_DIR / "plan_session.jsonl", {"role": "shape_diff", "stage": "build_plan", "diff": sd})
        except Exception:
            pass
        raise
    # Persist normalized build-plan
    try:
        append_jsonl(DATA_DIR / "plan_session.jsonl", {"role": "build_plan_raw", "content": raw_js})
        append_jsonl(DATA_DIR / "plan_session.jsonl", {"role": "build_plan_normalized", "content": js})
    except Exception:
        pass

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
            parts = [
                str(c0.get("novelty_kind") or "").strip(),
                str(c0.get("spec_hint") or "").strip(),
                str(c0.get("delta_vs_prior") or "").strip(),
            ]
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
    try:
        validate_json(plan, PLAN_SCHEMA, "final_plan_from_build")
    except Exception as e:
        print(f"[ERR] final_plan_from_build schema violation: {e}")
        _print_schema_errors("final_plan_from_build", plan, PLAN_SCHEMA)
        raise
    return plan


 


def run_planning_session(novelty: Dict[str, Any]) -> Dict[str, Any]:
    """Run a small multi-agent planning session and return the final plan.
    Logs conversation turns into data/plan_session.jsonl. No offline fallback; fails loudly on error.
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
        # Lightweight constraints block with domain/task hints and dataset defaults
        try:
            dom_tags = []
            task_tags = []
            for c in candidates:
                dom_tags.extend([t for t in c.get("domain_tags", []) if t])
                task_tags.extend([t for t in c.get("task_tags", []) if t])
            dom = "generic"
            if "generic" in dom_tags:
                dom = "generic"
            elif dom_tags:
                dom = str(dom_tags[0])
        except Exception:
            dom = "generic"
            task_tags = []
        constraints = {
            "budget": "<= 1 epoch per run, <= 100 steps",
            "dataset_default": f"{dataset_name('ISIC').upper()} under {dataset_path_for()} with train/ and val/",
            "domain": dom,
            "task_tags": list({t for t in task_tags if t}),
            "allowed_datasets": [dataset_name('ISIC')],
        }
        # Planner request tuning (timeouts/retries)
        try:
            _req_timeout = int(get("pipeline.planner.request_timeout", 180) or 180)
        except Exception:
            _req_timeout = 180
        try:
            _max_tries = int(get("pipeline.planner.request_retries", 4) or 4)
        except Exception:
            _max_tries = 4
        # Log effective LLM configuration (for troubleshooting)
        try:
            eff_provider = str((get("llm.provider", LLM_PROVIDER) or LLM_PROVIDER)).lower()
            eff_model = str((get("pipeline.planner.model", None) or get("llm.model", LLM_MODEL) or LLM_MODEL))
            print(
                f"[PLANNER] LLM config: provider={eff_provider}, model={eff_model}, url={LLM_CHAT_URL}, timeout={_req_timeout}s, retries={_max_tries}"
            )
        except Exception:
            pass

        # PI draft
        pi_system = load_prompt("planner_pi_system") or (
            "You are the PI (Principal Investigator). Draft a clear, minimal, and resource-aware plan grounded in the novelty report.\n"
            "Use novelty_outline (problems/objectives/contributions/research_questions) and choose ONE novelty from novelty_candidates as the novelty_focus.\n"
            "Translate the chosen candidate's eval_plan/spec_hint into concrete tasks/steps that fit <=1 epoch and <=100 steps. Use delta_vs_prior to define baselines and at least one falsifying ablation.\n"
            "Output MUST include ALL keys, each non-empty: objective, hypotheses, success_criteria (metric + delta vs baseline), datasets (name + path), baselines, novelty_focus, stopping_rules, tasks (name/why/steps), risks (risk/mitigation), assumptions, milestones.\n"
            "STRICT success_criteria SHAPE: an ARRAY of OBJECTS, each {metric: string, delta_vs_baseline: number}. No 'primary_metric'/'secondary_metrics' maps. No string deltas.\n"
            "Metrics may include mAP@0.5, FP_per_image, precision@IoU0.5, IoU, Dice, AUC, ECE, or val_accuracy — pick those matching the novelty.\n"
            "Be concrete, specify thresholds, and avoid ambiguous phrasing.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        pi_user = json.dumps({
            "novelty_report": novelty,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "constraints": constraints,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "PI_input", "system": pi_system, "user": json.loads(pi_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        def _omit_temp_for_gpt5() -> bool:
            try:
                prov = str((get("llm.provider", LLM_PROVIDER) or LLM_PROVIDER)).lower()
                mod = str((model or get("llm.model", LLM_MODEL) or LLM_MODEL)).lower()
                return prov == "openai" and (mod.startswith("gpt-5-") or mod == "gpt-5")
            except Exception:
                return False
        try:
            print(
                f"[PLANNER] Calling LLM stage=PI (temp={'omit' if _omit_temp_for_gpt5() else '0.0'}) timeout={_req_timeout}s retries={_max_tries}"
            )
            if _omit_temp_for_gpt5():
                pi = chat_json_cached(pi_system, pi_user, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
            else:
                pi = chat_json_cached(pi_system, pi_user, temperature=0.0, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
        except LLMError as e:
            print(f"[ERR] LLM error at stage=PI: {e}")
            print("[HINT] Increase pipeline.planner.request_timeout or check network/API key. See config.yaml.")
            raise
        append_jsonl(session_log, {"role": "PI", "content": pi})
        # Normalize PI and log diff
        try:
            pi_raw = pi
            pi = _normalize_plan_shape(pi)
            append_jsonl(session_log, {"role": "PI_normalized", "content": pi})
            sd = _shape_diff(pi_raw, pi)
            append_jsonl(session_log, {"role": "shape_diff", "stage": "PI", "diff": sd})
        except Exception:
            pass
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
            "Respect constraints (<=1 epoch, small steps). Use delta_vs_prior to propose explicit ablations that isolate the change. Return JSON with ALL keys present and non-empty: objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules, tasks, risks, assumptions, milestones.\n"
            "STRICT success_criteria SHAPE: an ARRAY of OBJECTS, each {metric: string, delta_vs_baseline: number}. No 'primary_metric'/'secondary_metrics'. Use 0.03 for +3%.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        eng_user = json.dumps({
            "pi_plan": pi,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "constraints": constraints,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Engineer_input", "system": eng_system, "user": json.loads(eng_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        try:
            print(
                f"[PLANNER] Calling LLM stage=Engineer (temp={'omit' if _omit_temp_for_gpt5() else '0.0'}) timeout={_req_timeout}s retries={_max_tries}"
            )
            if _omit_temp_for_gpt5():
                eng = chat_json_cached(eng_system, eng_user, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
            else:
                eng = chat_json_cached(eng_system, eng_user, temperature=0.0, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
        except LLMError as e:
            print(f"[ERR] LLM error at stage=Engineer: {e}")
            print("[HINT] Increase pipeline.planner.request_timeout or reduce prompt size/personas to shorten calls.")
            raise
        append_jsonl(session_log, {"role": "Engineer", "content": eng})
        # Normalize Engineer and log diff
        try:
            eng_raw = eng
            eng = _normalize_plan_shape(eng)
            append_jsonl(session_log, {"role": "Engineer_normalized", "content": eng})
            sd = _shape_diff(eng_raw, eng)
            append_jsonl(session_log, {"role": "shape_diff", "stage": "Engineer", "diff": sd})
        except Exception:
            pass
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
            "Finalize a lean plan that fits the compute budget; keep thresholds and dataset paths explicit. Ensure at least one ablation based on delta_vs_prior is present.\n"
            "Return final plan JSON with ALL keys present and non-empty: objective, hypotheses, success_criteria, datasets, baselines, novelty_focus, stopping_rules, tasks, risks, assumptions, milestones.\n"
            "STRICT success_criteria SHAPE: an ARRAY of OBJECTS, each {metric: string, delta_vs_baseline: number}. No 'primary_metric'/'secondary_metrics'. Numeric deltas only.\n"
            "Self-check: validate keys/types/constraints; return only JSON."
        )
        rev_user = json.dumps({
            "engineer_plan": eng,
            "novelty_outline": outline,
            "novelty_candidates": candidates,
            "constraints": constraints,
            "persona_notes": persona_notes,
        }, ensure_ascii=False)
        append_jsonl(session_log, {"role": "Reviewer_input", "system": rev_system, "user": json.loads(rev_user)})
        model = get("pipeline.planner.model", None)
        profile = get("pipeline.planner.llm", None)
        try:
            print(
                f"[PLANNER] Calling LLM stage=Reviewer (temp={'omit' if _omit_temp_for_gpt5() else '0.0'}) timeout={_req_timeout}s retries={_max_tries}"
            )
            if _omit_temp_for_gpt5():
                rev = chat_json_cached(rev_system, rev_user, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
            else:
                rev = chat_json_cached(rev_system, rev_user, temperature=0.0, model=model, profile=profile, timeout=_req_timeout, max_tries=_max_tries)
        except LLMError as e:
            print(f"[ERR] LLM error at stage=Reviewer: {e}")
            print("[HINT] Increase pipeline.planner.request_timeout or re-run later to avoid transient rates/timeouts.")
            raise
        # Normalize Reviewer and log diff
        try:
            rev_raw = rev
            rev = _normalize_plan_shape(rev)
            append_jsonl(session_log, {"role": "Reviewer_normalized", "content": rev})
            sd = _shape_diff(rev_raw, rev)
            append_jsonl(session_log, {"role": "shape_diff", "stage": "Reviewer", "diff": sd})
        except Exception:
            pass

        # Strict validation warnings (no mutation here)
        def _warn_if_invalid(tag: str, obj: Any) -> None:
            try:
                if isinstance(obj, dict):
                    validate_json(obj, PLAN_SCHEMA, f"{tag} plan")
            except Exception as e:
                print(f"[WARN] {tag} produced invalid plan shape: {e}")

        _warn_if_invalid("PI", pi)
        if is_verbose():
            _print_schema_errors("PI", pi, PLAN_SCHEMA)
        _warn_if_invalid("Engineer", eng)
        if is_verbose():
            _print_schema_errors("Engineer", eng, PLAN_SCHEMA)
        _warn_if_invalid("Reviewer", rev)
        if is_verbose():
            _print_schema_errors("Reviewer", rev, PLAN_SCHEMA)
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
        # Normalize once more before strict validation
        final = _normalize_plan_shape(final)
        # Strict validation of the final composed plan (with diagnostics)
        try:
            validate_json(final, PLAN_SCHEMA, "final_plan")
        except Exception as e:
            print(f"[ERR] final_plan schema violation: {e}")
            _print_schema_errors("final_plan", final, PLAN_SCHEMA)
            raise
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
    # Ensure environment variables are available
    try:
        load_dotenv()
    except Exception:
        pass
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
