import os
import json
import pathlib
import time
from typing import Dict, Any, List, Tuple

from utils.llm_utils import chat_json, LLMError, LLM_PROVIDER, LLM_MODEL
from lab.config import get
from lab.logging_utils import is_verbose, vprint, append_jsonl
from lab.novelty_facets import extract_facets
from lab.novelty_scoring import score_and_filter_ideas
import numpy as _np

try:
    # Optional: multi‑persona discussion helpers
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore



DATA_DIR = pathlib.Path("data")
SUM_DIR = DATA_DIR / "summaries"
NOVELTY_DIR = DATA_DIR / "novelty"
REPORT_PATH = NOVELTY_DIR / "novelty_report.json"
 
# Files for optional multi‑agent discussion logs
NOVELTY_SESSION = NOVELTY_DIR / "novelty_session.jsonl"
NOVELTY_TRANSCRIPT = NOVELTY_DIR / "novelty_transcript.md"
NOVELTY_UNIQ_JSON = NOVELTY_DIR / "novelty_uniqueness.json"
NOVELTY_UNIQ_TRANSCRIPT = NOVELTY_DIR / "novelty_uniqueness.md"
NOVELTY_IDEA_CRITIQUES = NOVELTY_DIR / "idea_critiques.jsonl"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    NOVELTY_DIR.mkdir(parents=True, exist_ok=True)


# --- BEGIN: Tier-1 verbatim user-spec injection helper ---
def _inject_user_claims(final: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load ./data/user_spec.json (or env USER_SPEC_PATH) and inject verbatim fields
    into `final` so downstream validators and blueprint builders quote your sentences.
    Expected optional fields in the JSON: objective (str), hypotheses (list[str]),
    success_criteria (list[dict]). Permissive: does not delete when missing.
    """
    import os as _os, json as _json, pathlib as _pl
    p = _pl.Path(_os.getenv("USER_SPEC_PATH", "data/user_spec.json"))
    if not p.exists():
        return final
    try:
        spec = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return final

    problems = list(final.get("problems") or [])
    objectives = list(final.get("objectives") or [])
    contributions = list(final.get("contributions") or [])
    research_questions = list(final.get("research_questions") or [])

    # Verbatim mapping (do NOT rewrite the user's sentences)
    obj = spec.get("objective")
    if isinstance(obj, str) and obj.strip():
        problems = [obj.strip()] or problems
        objectives = [obj.strip()] or objectives
    for sc in (spec.get("success_criteria") or []):
        try:
            metric = str(sc.get("metric") or "").strip()
            dvb = sc.get("delta_vs_baseline")
            if metric and dvb is not None:
                contributions.append(f"{metric} : Δ_vs_baseline = {dvb}")
        except Exception:
            continue
    for h in (spec.get("hypotheses") or []):
        if isinstance(h, str) and h.strip():
            research_questions.append(h.strip())

    final["problems"] = problems
    final["objectives"] = objectives
    final["contributions"] = contributions
    final["research_questions"] = research_questions
    return final
# --- END: Tier-1 verbatim user-spec injection helper ---


def _load_citations(csv_path: pathlib.Path) -> List[Dict[str, str]]:
    """Load citations from an optional CSV (abstract_screen_deepseek.csv)."""
    cites: List[Dict[str, str]] = []
    if not csv_path.exists():
        return cites
    try:
        import csv
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                cites.append({
                    "title": str(row.get("title") or ""),
                    "year": str(row.get("year") or ""),
                    "doi": str(row.get("doi") or ""),
                })
    except Exception:
        return []
    return cites


def summarize_paper(text: str, title_hint: str = "") -> Dict[str, Any]:
    raise RuntimeError("summarize_paper is no longer available in agents.novelty; run agents.summarize first.")


def criticize_paper(summary: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("criticize_paper is no longer available in agents.novelty; run agents.summarize first.")


def group_novelty_and_ideas(summaries: List[Dict[str, Any]], persona_notes: List[str] | None = None) -> Dict[str, Any]:
    # Build compact input
    items = []
    for s in summaries:
        items.append({
            "title": str(s.get("title") or "")[:120],
            "novelty_claims": s.get("novelty_claims") or [],
            "methods": s.get("methods") or [],
            "limitations": s.get("limitations") or [],
        })
    goal = str(get("project.goal", "your goal") or "your goal")
    system = (
        "You are a meticulous panel moderator coordinating multiple expert personas to synthesize novelty across papers.\n"
        "Tasks: (a) robustly CLUSTER into 3–6 THEMES; (b) write a compact SUMMARY per theme with divergences/caveats and 2–4 representative paper titles; (c) propose 5–8 NEW RESEARCH IDEAS.\n"
        f"Orient the synthesis toward the goal: {goal}.\n"
        "For NEW RESEARCH IDEAS, return BOTH a short list (new_ideas) AND a detailed, numbered list (new_ideas_detailed) with the following fields for each idea:\n"
        "id (1-based string), title (<=100 chars), novelty_kind [architecture|training_objective|data|evaluation|augmentation|optimizer|other], why_novel (<=240 chars), spec_hint (<=240 chars minimal actionable change), method (<=280 chars concrete method/recipe), risks (<=240 chars), eval_plan (list of <=6 short steps), compute_budget (<=60 chars).\n"
        "If YOLO/detection-related, include yolo fields inside method/spec_hint where appropriate (yolo_version, backbone, img_size, anchors, conf_thresh, iou_thresh, nms, task_adapt).\n"
        "Avoid duplication across ideas; keep ideas feasible under tight budgets. Return strictly JSON.\n"
        "Self-check: validate keys/types per output_schema; fix mismatches and return JSON only."
    )
    model = get("pipeline.novelty.model", None)
    profile = get("pipeline.novelty.llm", None)
    max_tries = int(get("pipeline.novelty.retries", 3) or 3)
    max_attempts = int(get("pipeline.novelty.max_attempts", 3) or 3)
    min_ideas = int(get("pipeline.novelty.min_ideas", 3) or 3)

    attempt_feedback: List[str] = []
    js: Dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        augmented_payload = {
            "papers": items,
            "persona_notes": list(persona_notes or []),
            "attempt": attempt,
            "feedback": attempt_feedback,
            "output_schema": {
                "themes": [
                    {
                        "name": "string",
                        "summary": "string",
                        "representative_papers": ["string"],
                    }
                ],
                "new_ideas": ["string"],
                "new_ideas_detailed": [
                    {
                        "id": "string",
                        "title": "string",
                        "novelty_kind": "string",
                        "why_novel": "string",
                        "spec_hint": "string",
                        "method": "string",
                        "risks": "string",
                        "eval_plan": ["string"],
                        "compute_budget": "string"
                    }
                ]
            }
        }

        tries = 0
        while True:
            tries += 1
            try:
                js = chat_json(
                    system,
                    json.dumps(augmented_payload, ensure_ascii=False),
                    temperature=0.2,
                    model=model,
                    profile=profile,
                )
                break
            except LLMError as exc:
                if tries >= max_tries:
                    if attempt >= max_attempts:
                        print(f"[ERR] LLM group discussion failed after {tries} tries (attempt {attempt}/{max_attempts}): {exc}")
                        return {"themes": [], "new_ideas": [], "new_ideas_detailed": []}
                    wait = min(15 * tries, 60)
                    print(f"[WARN] LLM novelty synthesis failed (attempt {attempt}/{max_attempts}, try {tries}/{max_tries}): {exc}. Retrying in {wait}s…")
                    time.sleep(wait)
                    break  # break inner loop to reattempt outer payload if retries exhausted
                wait = min(15 * tries, 60)
                print(f"[WARN] LLM novelty synthesis failed (try {tries}/{max_tries}): {exc}. Waiting {wait}s before retry…")
                time.sleep(wait)
                continue
        else:
            continue

        themes = js.get("themes") or []
        ideas = js.get("new_ideas_detailed") or []
        if isinstance(ideas, list) and len(ideas) >= min_ideas:
            break
        attempt_feedback.append(
            f"Attempt {attempt} returned {len(ideas) if isinstance(ideas, list) else 0} detailed ideas; need at least {min_ideas}. Increase diversity and novelty."\
        )
        time.sleep(min(10 * attempt, 45))
    else:
        return {"themes": [], "new_ideas": [], "new_ideas_detailed": []}
    # light shape enforcement
    themes = js.get("themes") or []
    if not isinstance(themes, list):
        themes = []
    new_ideas = js.get("new_ideas") or []
    if not isinstance(new_ideas, list):
        new_ideas = []
    # Detailed ideas (objects)
    di = js.get("new_ideas_detailed") or []
    if not isinstance(di, list):
        di = []
    fixed_di: List[Dict[str, Any]] = []
    # Allow disable via config/env
    detailed_enable = True
    try:
        detailed_enable = bool(get("pipeline.novelty.detailed_ideas", True))
        env = str(os.getenv("NOVELTY_DETAILED", "")).lower()
        if env in {"0", "false", "no", "off"}:
            detailed_enable = False
        elif env in {"1", "true", "yes", "on"}:
            detailed_enable = True
    except Exception:
        detailed_enable = True
    if detailed_enable:
        for idx, item in enumerate(di, start=1):
            if not isinstance(item, dict):
                continue
            fixed_di.append({
                "id": str(item.get("id") or idx),
                "title": str(item.get("title") or "").strip(),
                "novelty_kind": str(item.get("novelty_kind") or "").strip(),
                "why_novel": str(item.get("why_novel") or "").strip(),
                "spec_hint": str(item.get("spec_hint") or "").strip(),
                "method": str(item.get("method") or "").strip(),
                "risks": str(item.get("risks") or "").strip(),
                "eval_plan": [str(x) for x in (item.get("eval_plan") or []) if str(x).strip()],
                "compute_budget": str(item.get("compute_budget") or "").strip(),
            })
    return {"themes": themes, "new_ideas": new_ideas, "new_ideas_detailed": fixed_di}


def group_novelty_and_ideas_v2(
    summaries: List[Dict[str, Any]],
    persona_notes: List[str] | None = None,
    facets: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    # Build compact input from summaries
    items: List[Dict[str, Any]] = []
    for s in summaries:
        rel = []
        try:
            for rw in (s.get("related_work") or []):
                if isinstance(rw, dict):
                    rel.append({
                        "citation": str(rw.get("citation") or ""),
                        "method_summary": str(rw.get("method_summary") or ""),
                        "methods": [str(x) for x in (rw.get("methods") or []) if str(x)],
                        "datasets": [str(x) for x in (rw.get("datasets") or []) if str(x)],
                        "metrics": [str(x) for x in (rw.get("metrics") or []) if str(x)],
                    })
        except Exception:
            rel = []
        items.append({
            "title": str(s.get("title") or "")[:120],
            "novelty_claims": s.get("novelty_claims") or [],
            "methods": s.get("methods") or [],
            "limitations": s.get("limitations") or [],
            "related_methods": rel,
        })

    goal = str(get("project.goal", "your goal") or "your goal")
    backbones = [str(b) for b in (facets or {}).get("backbones", []) if str(b)] if facets else []
    datasets_facets = [str(d) for d in (facets or {}).get("datasets", []) if str(d)] if facets else []
    metrics_facets = [str(m) for m in (facets or {}).get("metrics", []) if str(m)] if facets else []
    backbone_hint = f"Use model families from: {', '.join(backbones[:6])}." if backbones else "Use model families mentioned in the supplied papers."
    dataset_hint = f"Datasets should come from: {', '.join(datasets_facets[:6])}." if datasets_facets else "Datasets must be drawn from the provided papers."
    metric_hint = f"Metrics should come from: {', '.join(metrics_facets[:6])}." if metrics_facets else "Metrics must be drawn from the provided papers."
    facet_block = json.dumps(facets or {}, ensure_ascii=False)
    system_lines = [
        "You are a meticulous panel moderator coordinating multiple expert personas to synthesize novelty across papers.",
        "STRICT GROUNDING:",
        "- Use ONLY backbones/operators/datasets that appear in facets.backbones/ops/datasets OR that appear in related_methods from summaries. Do not invent generic backbones.",
        "- Every idea MUST include: derived_from_titles (>=2), delta_vs_prior (<=240 chars citing those titles), AND >=3 numeric spec tokens (e.g., img_size=640, lr=3e-4, iou_thresh=0.6).",
        "- The novelty must focus on MODEL/architecture innovations (set novelty_kind='architecture'); avoid purely optimization- or data-only tweaks.",
        f"- {backbone_hint}",
        f"- {dataset_hint}",
        f"- {metric_hint}",
        "- Avoid generic augmentation-only or hparam-only tweaks unless justified as non-trivial.",
        "- If detection/YOLO, include yolo_version, backbone, img_size, anchors, conf_thresh, iou_thresh, nms, task_adapt in method/spec_hint.",
        "- Tag each idea with domain_tags (e.g., ['derm','lung','retina'] or ['generic']) and task_tags (e.g., ['segmentation','classification','detection']).",
        "Tasks: (a) robustly CLUSTER into 3–6 THEMES; (b) write a compact SUMMARY per theme with divergences/caveats and 2–4 representative paper titles; (c) propose 5–8 NEW RESEARCH IDEAS.",
        "Return strictly JSON and self-validate against the output_schema.",
    ]
    system = "\n".join(line for line in system_lines if line)

    user_payload = {
        "papers": items,
        "persona_notes": list(persona_notes or []),
        "facets": json.loads(facet_block),
        "output_schema": {
            "themes": [{"name": "string", "summary": "string", "representative_papers": ["string"]}],
            "new_ideas": ["string"],
            "new_ideas_detailed": [
                {
                    "id": "string",
                    "title": "string",
                    "novelty_kind": "string",
                    "why_novel": "string",
                    "spec_hint": "string",
                    "method": "string",
                    "risks": "string",
                    "eval_plan": ["string"],
                    "compute_budget": "string",
                    "derived_from_titles": ["string"],
                    "delta_vs_prior": "string",
                    "domain_tags": ["string"],
                    "task_tags": ["string"],
                }
            ],
        },
    }

    model = get("pipeline.novelty.model", None)
    profile = get("pipeline.novelty.llm", None)

    # diversify a bit, then merge (but do not multi-sample for GPT-5)
    temps = get("pipeline.novelty.sampling.temperatures", [0.2]) or [0.2]
    if isinstance(temps, str):
        try:
            temps = [float(t) for t in str(temps).split(",") if t.strip()]
        except Exception:
            temps = [0.2]
    try:
        provider = str(get("llm.provider", LLM_PROVIDER) or LLM_PROVIDER).lower()
        sel_model = str((model or get("llm.model", LLM_MODEL) or LLM_MODEL)).lower()
        if provider == "openai" and (sel_model.startswith("gpt-5-") or sel_model == "gpt-5"):
            temps = [0.0]
    except Exception:
        pass
    candidates: List[Dict[str, Any]] = []
    themes_agg: List[Dict[str, Any]] = []
    # Allow a longer request timeout via YAML/env; default to 180s for robustness
    try:
        _req_timeout = int(get("pipeline.novelty.request_timeout", 180) or 180)
    except Exception:
        _req_timeout = 180
    for tval in temps:
        js = chat_json(
            system,
            json.dumps(user_payload, ensure_ascii=False),
            temperature=float(tval),
            model=model,
            profile=profile,
            timeout=_req_timeout,
        )
        th = js.get("themes") or []
        di = js.get("new_ideas_detailed") or []
        if isinstance(th, list):
            themes_agg.extend([x for x in th if isinstance(x, dict)])
        if isinstance(di, list):
            candidates.extend([x for x in di if isinstance(x, dict)])

    # light shape enforcement + dedupe
    fixed_di: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(candidates, start=1):
        dft = {
            "id": str(item.get("id") or idx),
            "title": str(item.get("title") or "").strip(),
            "novelty_kind": "architecture",
            "why_novel": str(item.get("why_novel") or "").strip(),
            "spec_hint": str(item.get("spec_hint") or "").strip(),
            "method": str(item.get("method") or "").strip(),
            "risks": str(item.get("risks") or "").strip(),
            "eval_plan": [str(x) for x in (item.get("eval_plan") or []) if str(x).strip()],
            "compute_budget": str(item.get("compute_budget") or "").strip(),
            "derived_from_titles": [str(x).strip() for x in (item.get("derived_from_titles") or []) if str(x).strip()],
            "delta_vs_prior": str(item.get("delta_vs_prior") or "").strip(),
            "domain_tags": [str(x).strip().lower() for x in (item.get("domain_tags") or []) if str(x).strip()],
            "task_tags": [str(x).strip().lower() for x in (item.get("task_tags") or []) if str(x).strip()],
        }
        key = (dft["title"].lower(), dft["delta_vs_prior"].lower())
        if key in seen:
            continue
        seen.add(key)
        fixed_di.append(dft)

    themes = [t for t in themes_agg if isinstance(t, dict)]
    new_ideas = [it.get("title", "") for it in fixed_di]
    # Local duplicate similarity against available novelty/methods text (no external models)
    try:
        import difflib
        corpus: List[str] = []
        for s in summaries:
            try:
                cl = " ".join([" ".join(s.get("novelty_claims") or []), " ".join(s.get("methods") or [])])
                cl = cl.strip()
                if cl:
                    corpus.append(cl)
            except Exception:
                continue
        def _max_dup_sim(text: str) -> float:
            best = 0.0
            for c in corpus:
                try:
                    r = difflib.SequenceMatcher(None, text.lower(), c.lower()).ratio()
                except Exception:
                    r = 0.0
                if r > best:
                    best = r
            return best
        # Tier‑1 scorecard inspired scoring
        def _tier1_score(it: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
            w = {"delta": 0.25, "ablation": 0.25, "compute": 0.20, "general": 0.15, "rigor": 0.15}
            s = {k: 0.0 for k in w}
            delta = it.get("delta_vs_prior", "").strip()
            s["delta"] = 1.0 if len(delta) >= 12 else (0.4 if delta else 0.0)
            eval_plan = " ".join(it.get("eval_plan", []))
            abl_terms = ["ablation", "sweep", "off", "tau", "sigma", "dropout", "without"]
            ablation_hits = sum(term in eval_plan.lower() for term in abl_terms)
            s["ablation"] = min(1.0, 0.5 + 0.25 * ablation_hits) if ablation_hits else 0.0
            budget = (it.get("compute_budget", "") + " " + it.get("spec_hint", "") + " " + it.get("method", "")).lower()
            model_terms = ["transformer", "backbone", "convnext", "swin", "vision transformer", "vit", "architecture", "cnn"]
            s["compute"] = 1.0 if any(term in budget for term in model_terms) else (0.5 if "model" in budget else 0.0)
            spec_txt = (it.get("spec_hint", "") + " " + it.get("method", "")).lower()
            domain_words = ["isic", "derm", "skin", "retina", "lidc", "lung", "chestxray", "nih-chest"]
            s["general"] = 1.0 if not any(dw in spec_txt for dw in domain_words) else 0.4
            rigor = " ".join(it.get("eval_plan", [])).lower()
            s["rigor"] = 1.0 if any(m in rigor for m in ["iou", "dice", "map", "auc", "ece", "precision"]) else 0.5
            total = sum(w[k] * s[k] for k in w)
            return total, s
        for it in fixed_di:
            try:
                idea_txt = " ".join([
                    it.get("title", ""),
                    it.get("why_novel", ""),
                    it.get("spec_hint", ""),
                    it.get("method", ""),
                ]).strip()
                it["dup_sim"] = round(_max_dup_sim(idea_txt), 3) if idea_txt else 0.0
            except Exception:
                it["dup_sim"] = 0.0
            score, _ = _tier1_score(it)
            # Small penalty for high duplication similarity
            if it.get("dup_sim", 0.0) >= 0.7:
                score *= 0.75
            elif it.get("dup_sim", 0.0) >= 0.6:
                score *= 0.9
            it["tier1_score"] = round(float(score), 3)
        # Sort by tier1_score descending for downstream selection
        fixed_di.sort(key=lambda d: d.get("tier1_score", 0.0), reverse=True)
    except Exception:
        pass
    return {"themes": themes, "new_ideas": new_ideas, "new_ideas_detailed": fixed_di}


def derive_research_outline(themes_and_ideas: Dict[str, Any], citations: List[Dict[str, str]], persona_notes: List[str] | None = None) -> Dict[str, Any]:
    """Use LLM to derive problems/objectives/contributions/research_questions with citations."""
    n_probs = int(get("research.num_problems", 2) or 2)
    n_objs = int(get("research.num_objectives", 2) or 2)
    n_cont = int(get("research.num_contributions", 2) or 2)
    n_rqs = int(get("research.num_questions", 1) or 1)
    goal = str(get("project.goal", "your goal") or "your goal")
    system = (
        "You are a meticulous research planner translating synthesized themes into a concrete research outline.\n"
        f"Produce EXACTLY {n_probs} PROBLEMS (crisp, specific), {n_objs} OBJECTIVES (SMART-style), {n_cont} CONTRIBUTIONS (testable, minimal), and {n_rqs} RESEARCH_QUESTIONS (focused, empirically answerable).\n"
        "Cite when appropriate using (Title, DOI) for entries present in the provided citations list only; do not invent citations.\n"
        "Ensure scope fits <=1 epoch per run and small data budgets; prefer incremental, verifiable claims. Return strictly JSON with the requested keys.\n"
        "Self-check: Before responding, validate keys/types per output_schema; fix mismatches and return JSON only."
    )
    user_payload = {
        "goal": goal,
        "themes": themes_and_ideas.get("themes", []),
        "new_ideas": themes_and_ideas.get("new_ideas", []),
        "citations": citations,
        "persona_notes": list(persona_notes or []),
        "output_schema": {
            "problems": ["string"],
            "objectives": ["string"],
            "contributions": ["string"],
            "research_questions": ["string"],
        }
    }
    model = get("pipeline.novelty.model", None)
    profile = get("pipeline.novelty.llm", None)
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0, model=model, profile=profile)
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]
    return {
        "problems": (_as_list(js.get("problems")) or [])[:n_probs],
        "objectives": (_as_list(js.get("objectives")) or [])[:n_objs],
        "contributions": (_as_list(js.get("contributions")) or [])[:n_cont],
        "research_questions": (_as_list(js.get("research_questions")) or [])[:n_rqs],
        "citations": citations,
    }


def _persona_discussion(summaries: List[Dict[str, Any]], facets: Dict[str, List[str]] | None = None) -> List[str]:
    """Run an optional multi‑persona discussion across paper summaries.

    Returns list of short notes while logging full transcript to JSONL/Markdown.
    Controlled by config/env: pipeline.novelty.personas.enable or NOVELTY_PERSONAS=1.
    """
    try:
        enable = (
            bool(get("pipeline.novelty.personas.enable", False))
            or (str(os.getenv("NOVELTY_PERSONAS", "")).lower() in {"1", "true", "yes"})
        )
        if not enable or DialogueManager is None:
            return []
    except Exception:
        return []

    # Prepare compact context from summaries to keep tokens bounded
    try:
        ctx_items: List[Dict[str, Any]] = []
        allowed_titles: List[str] = []
        for s in summaries[:30]:  # cap to 30 papers for brevity
            ctx_items.append({
                "title": str(s.get("title") or "")[:120],
                "novelty_claims": (s.get("novelty_claims") or [])[:5],
                "methods": (s.get("methods") or [])[:5],
                "limitations": (s.get("limitations") or [])[:5],
            })
            t = str(s.get("title") or "").strip()
            if t:
                allowed_titles.append(t)
        # Facets summary passed to personas to avoid "missing inputs" loops
        fsum = {
            "backbones": (facets or {}).get("backbones", [])[:10],
            "ops": (facets or {}).get("ops", [])[:10],
            "datasets": (facets or {}).get("datasets", [])[:10],
        }
        dm = DialogueManager()
        dm.post(
            "User",
            (
                "We will discuss novelty themes across the provided papers and propose novel, actionable ideas under tight compute.\n"
                "STRICT GROUNDING: Use ONLY the provided facets (backbones/ops/datasets) and the listed paper titles. Where you cite titles, select from allowed_titles.\n"
                "VALIDATION: When asked for decisions, respond in concise, structured form; keep within length budgets.\n"
                "CONSENSUS: When prompted, reply with 'AGREE_NOVELTY: yes|no' and a one-line SPEC_HINT; be explicit.\n"
            ),
        )
        dm.post("User", json.dumps({"papers": ctx_items, "facets": fsum, "allowed_titles": allowed_titles}, ensure_ascii=False))

        steps = 3
        # Prefer YAML, but allow env override via NOVELTY_STEPS
        try:
            steps = int(get("pipeline.novelty.personas.steps", 3) or 3)
        except Exception:
            steps = 3
        try:
            env_steps = os.getenv("NOVELTY_STEPS")
            if env_steps:
                steps = max(1, int(env_steps))
        except Exception:
            pass

        # Console printing control (default on): env NOVELTY_PRINT=0 to silence
        try:
            print_flag = (
                (str(os.getenv("NOVELTY_PRINT", "")).lower() in {"1", "true", "yes"})
                or (get("pipeline.novelty.personas.print", True) is True)
            )
        except Exception:
            print_flag = True

        notes: List[str] = []
        # Log header
        append_jsonl(NOVELTY_SESSION, {"role": "system", "content": {"stage": "novelty_personas", "steps": steps}})
        if print_flag:
            try:
                print(f"[NOVELTY CHAT] Personas discussion enabled (steps={steps})")
                print("[User] We will discuss novelty themes across papers and propose novel ideas.")
            except Exception:
                pass

        # Debate mode toggle
        debate_enable = False
        try:
            debate_enable = bool(get("pipeline.novelty.personas.debate.enable", False)) or (
                str(os.getenv("NOVELTY_DEBATE", "")).lower() in {"1", "true", "yes"}
            )
        except Exception:
            debate_enable = False

        # Custom roles/order if provided
        try:
            roles_cfg = get("pipeline.novelty.personas.debate.roles", None)
        except Exception:
            roles_cfg = None
        env_roles = os.getenv("NOVELTY_ROLES", "")
        if roles_cfg or env_roles:
            try:
                order = roles_cfg if isinstance(roles_cfg, list) else [s.strip() for s in env_roles.split(",") if s.strip()]
                if order:
                    # Recreate DialogueManager with explicit order
                    dm = DialogueManager(order=order)
            except Exception:
                pass

        if not debate_enable:
            # Simple round-robin turns
            # Build stable alias mapping for duplicates in dm.order
            try:
                rr_order = getattr(dm, "order")  # type: ignore[attr-defined]
            except Exception:
                rr_order = ["PhD", "Professor", "SW", "ML"]
            rr_counts: Dict[str, int] = {}
            rr_alias_total: Dict[str, int] = {}
            for rname in rr_order:
                rr_alias_total[rname] = rr_alias_total.get(rname, 0) + 1
            # Per-role turn counter cycles through 1..total for that role
            for i in range(max(1, steps)):
                r = dm.step_auto()
                role_name = str(r.get('role',''))
                total = rr_alias_total.get(role_name, 1)
                prev = rr_counts.get(role_name, 0)
                idx = (prev % total) + 1
                rr_counts[role_name] = prev + 1
                alias = f"{role_name}#{idx}" if total > 1 else role_name
                note = f"[{alias}] {r.get('text','')}"
                notes.append(note)
                append_jsonl(NOVELTY_SESSION, {"role": role_name, "alias": alias, "content": r.get("text", ""), "phase": "round"})
                if print_flag:
                    try:
                        print(note)
                    except Exception:
                        pass
        else:
            # Structured debate: 3–4 personas, 2 proposers, critiques, consensus
            try:
                order = getattr(dm, "order")  # type: ignore[attr-defined]
            except Exception:
                order = ["PhD", "Professor", "SW", "ML"]
            proposers = order[:2] if len(order) >= 2 else order

            # Stable aliases for whole debate (Professor#1, Professor#2, ...)
            alias_counts: Dict[str, int] = {}
            alias_list: List[str] = []
            for rname in order:
                alias_counts[rname] = alias_counts.get(rname, 0) + 1
                alias_list.append(f"{rname}#{alias_counts[rname]}" if alias_counts[rname] > 0 and order.count(rname) > 1 else rname)
            if print_flag:
                try:
                    print("[NOVELTY CHAT] Participants:", ", ".join(alias_list))
                except Exception:
                    pass

            def _alias_at(index: int) -> str:
                try:
                    return alias_list[index]
                except Exception:
                    return order[index] if 0 <= index < len(order) else ""

            # Proposals from proposers (force citations + deltas; titles subset of allowed_titles)
            for p_idx, role in enumerate(proposers):
                prompt = (
                    f"From the perspective of {_alias_at(p_idx)}, propose 2 concrete NOVELTY directions grounded in the provided papers and facets.\n"
                    "Each must include:\n"
                    "TITLES:[t1,t2] (subset of allowed_titles), DELTA_VS_PRIOR:<one sentence referencing t1/t2>, SPEC_HINT:<numbers>, RISK:<one line>.\n"
                    "Use ONLY backbones from facets.backbones. If detection, include yolo_version, iou_thresh, nms.\n"
                    "Do NOT ask for inputs; proceed using provided facets and context.\n"
                    "Prefix each with 'PROPOSAL_NOVELTY:'."
                )
                r = dm.step_role(role, prompt=prompt)
                alias = _alias_at(p_idx)
                note = f"[{alias}] {r.get('text','')}"
                notes.append(note)
                append_jsonl(NOVELTY_SESSION, {"role": role, "alias": alias, "content": r.get("text", ""), "phase": "proposal"})
                if print_flag:
                    try:
                        print(note)
                    except Exception:
                        pass

            # Debate rounds (questions + answers + critique + consensus)
            rounds = 2
            try:
                rounds = int(get("pipeline.novelty.personas.debate.rounds", 2) or 2)
            except Exception:
                rounds = 2
            try:
                if os.getenv("NOVELTY_ROUNDS"):
                    rounds = max(1, int(os.getenv("NOVELTY_ROUNDS", "2") or 2))
            except Exception:
                pass

            # Unlimited until agree? Use a high safety cap if enabled
            until_agree = False
            try:
                until_agree = bool(get("pipeline.novelty.personas.debate.until_agree", False)) or (
                    str(os.getenv("NOVELTY_UNTIL_AGREE", "")).lower() in {"1", "true", "yes"}
                )
            except Exception:
                until_agree = False
            max_rounds = rounds
            try:
                mr_env = os.getenv("NOVELTY_MAX_ROUNDS", "")
                if mr_env:
                    max_rounds = max(1, int(mr_env))
            except Exception:
                pass
            if until_agree:
                # If no explicit max, choose a generous cap to avoid infinite loops
                if not os.getenv("NOVELTY_MAX_ROUNDS"):
                    max_rounds = max(max_rounds, 20)

            agreed = False
            round_idx = 0
            while round_idx < max_rounds and not agreed:
                round_idx += 1
                # Cross-examination: each role asks questions
                for idx, role in enumerate(order):
                    prompt = (
                        f"From the perspective of {_alias_at(idx)}, ask 2 critical QUESTIONS about NOVELTY.\n"
                        "Each QUESTION_NOVELTY must cite one of the TITLES and challenge originality vs that prior or the delta isolation."
                    )
                    r = dm.step_role(role, prompt=prompt)
                    alias = _alias_at(idx)
                    note = f"[{alias}] {r.get('text','')}"
                    notes.append(note)
                    append_jsonl(NOVELTY_SESSION, {"role": role, "alias": alias, "content": r.get("text", ""), "phase": "questions"})
                    if print_flag:
                        try:
                            print(note)
                        except Exception:
                            pass
                # Short answers: each role answers at least one outstanding question
                for idx, role in enumerate(order):
                    prompt = (
                        f"From the perspective of {_alias_at(idx)}, answer one QUESTION_NOVELTY concisely.\n"
                        "Prefix with 'ANSWER_NOVELTY:'. Reference the DELTA_VS_PRIOR and give one falsifying ablation with numeric spec."
                    )
                    r = dm.step_role(role, prompt=prompt)
                    alias = _alias_at(idx)
                    note = f"[{alias}] {r.get('text','')}"
                    notes.append(note)
                    append_jsonl(NOVELTY_SESSION, {"role": role, "alias": alias, "content": r.get("text", ""), "phase": "answers"})
                    if print_flag:
                        try:
                            print(note)
                        except Exception:
                            pass
                # Critiques by non-proposers
                for idx, role in enumerate(order):
                    if role in proposers:
                        continue
                    prompt = (
                        f"From the perspective of {_alias_at(idx)}, critique the NOVELTY of the current proposals (not process): assess uniqueness vs common baselines, plausibility under constraints, and clarity of COMPONENT_CHANGE. "
                        "Then propose a merged improvement that strengthens novelty or isolation. Prefix with 'CRITIQUE_NOVELTY:' and 'SUGGEST_NOVELTY:'; include a SPEC_HINT line."
                    )
                    r = dm.step_role(role, prompt=prompt)
                    alias = _alias_at(idx)
                    note = f"[{alias}] {r.get('text','')}"
                    notes.append(note)
                    append_jsonl(NOVELTY_SESSION, {"role": role, "alias": alias, "content": r.get("text", ""), "phase": "critique"})
                    if print_flag:
                        try:
                            print(note)
                        except Exception:
                            pass
                # Consensus vote (quorum-based)
                agrees = 0
                for idx, role in enumerate(order):
                    r = dm.step_role(role, prompt=(
                        "Do you AGREE on a single improved NOVEL proposal (not process)? Reply 'AGREE_NOVELTY: yes' or 'AGREE_NOVELTY: no' and a one-line merged SPEC_HINT."
                    ))
                    txt = str(r.get("text", ""))
                    alias = _alias_at(idx)
                    note = f"[{alias}] {txt}"
                    notes.append(note)
                    append_jsonl(NOVELTY_SESSION, {"role": role, "alias": alias, "content": txt, "phase": "consensus"})
                    if print_flag:
                        try:
                            print(note)
                        except Exception:
                            pass
                    if "agree_novelty:" in txt.lower() and "yes" in txt.lower():
                        agrees += 1
                # Record consensus outcome (best-effort visibility in logs/transcript)
                try:
                    from math import ceil as _ceil
                    quorum = _ceil(0.66 * max(1, len(order)))
                    append_jsonl(NOVELTY_SESSION, {
                        "role": "system",
                        "phase": "consensus_result",
                        "yes_votes": int(agrees),
                        "total": int(len(order)),
                        "quorum": int(quorum),
                        "agreed": bool(agrees >= quorum),
                    })
                except Exception:
                    pass
                # Also add a concise note line
                try:
                    from math import ceil as _ceil
                    quorum = _ceil(0.66 * max(1, len(order)))
                    notes.append(f"[Consensus] yes={agrees}/{len(order)} quorum={quorum} agreed={agrees >= quorum}")
                except Exception:
                    pass
                from math import ceil as _ceil
                if agrees >= _ceil(0.66 * max(1, len(order))):
                    agreed = True

            # Final consensus summary (ask for concise structured info)
            leader = order[1] if len(order) > 1 else order[0]
            r = dm.step_role(leader, prompt=(
                "Provide a CONSENSUS_NOVELTY_SUMMARY: one paragraph merging the best novelty proposal, with lines for NOVELTY_KIND and SPEC_HINT; cite at least two titles from allowed_titles."
            ))
            # Alias is second role's alias if exists; else first
            leader_idx = 1 if len(order) > 1 else 0
            leader_alias = _alias_at(leader_idx)
            note = f"[{leader_alias}] {r.get('text','')}"
            notes.append(note)
            append_jsonl(NOVELTY_SESSION, {"role": leader, "alias": leader_alias, "content": r.get("text", ""), "phase": "summary"})
            if print_flag:
                try:
                    print(note)
                except Exception:
                    pass

        # Best‑effort human‑readable transcript
        try:
            parts = ["# Novelty Discussion (Personas)", ""]
            parts.extend(notes)
            NOVELTY_TRANSCRIPT.write_text("\n\n".join(parts), encoding="utf-8")
        except Exception:
            pass
        if print_flag:
            try:
                print("[NOVELTY CHAT END]")
            except Exception:
                pass
        # Deduplicate near-identical lines to reduce repetition in logs
        try:
            import difflib as _difflib
            uniq: List[str] = []
            for n in notes:
                if not uniq:
                    uniq.append(n)
                    continue
                if max((_difflib.SequenceMatcher(None, n.lower(), u.lower()).ratio() for u in uniq), default=0.0) < 0.9:
                    uniq.append(n)
            return uniq
        except Exception:
            return notes
    except Exception:
        return []


def _uniqueness_reflection(panel: Dict[str, Any], persona_notes: List[str] | None = None) -> Dict[str, Any]:
    """Optional pass to nudge themes/ideas toward more novel, distinct directions.

    Controlled by config/env: pipeline.novelty.uniqueness.enable or NOVELTY_UNIQUENESS=1.
    Produces a dict: { unique_ideas: [str], critique: [str] } and logs a short transcript.
    """
    try:
        enabled = (
            bool(get("pipeline.novelty.uniqueness.enable", False))
            or (str(os.getenv("NOVELTY_UNIQUENESS", "")).lower() in {"1", "true", "yes"})
        )
        if not enabled:
            return {"unique_ideas": [], "critique": []}
    except Exception:
        return {"unique_ideas": [], "critique": []}

    # Compose a compact reflection prompt
    themes = panel.get("themes", [])
    ideas = panel.get("new_ideas", [])
    payload = {
        "themes": themes,
        "ideas": ideas,
        "persona_notes": list(persona_notes or []),
        # Lightweight generic prior-art hints to avoid vanilla moves
        "avoid": [
            "trivial augmentation tweaks (flip/jitter only)",
            "hyperparameter-only changes without methodological shift",
            "standard resnet18 head-only swaps without justification",
            "paper-writing loops (REPLACE/EDIT) or citation-only improvements",
            "generic ensemble without rationale",
        ],
        "output_schema": {
            "unique_ideas": ["string"],
            "critique": ["string"],
        }
    }
    system = (
        "You are a novelty reviewer. Critique the ideas for originality and propose an improved list of 5–10 UNIQUE ideas that are clearly distinct, specific, and testable under tight budgets.\n"
        "Avoid generic augmentation/hparam tropes; prefer combinations, constraints-aware designs, or measurement/label innovations. Return JSON."
    )
    try:
        model = get("pipeline.novelty.model", None)
        profile = get("pipeline.novelty.llm", None)
        js = chat_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.2, model=model, profile=profile)
    except LLMError:
        return {"unique_ideas": [], "critique": []}
    # light shape
    u = js.get("unique_ideas") or []
    if not isinstance(u, list):
        u = []
    c = js.get("critique") or []
    if not isinstance(c, list):
        c = []
    # Log transcript (best-effort)
    try:
        NOVELTY_UNIQ_JSON.write_text(json.dumps({"input": payload, "output": {"unique_ideas": u, "critique": c}}, ensure_ascii=False, indent=2), encoding="utf-8")
        parts = ["# Novelty Uniqueness Reflection", "", "## Critique", *[f"- {x}" for x in c], "", "## Unique Ideas", *[f"- {x}" for x in u]]
        NOVELTY_UNIQ_TRANSCRIPT.write_text("\n".join(parts), encoding="utf-8")
    except Exception:
        pass
    # Optional console print
    try:
        pf = str(os.getenv("NOVELTY_UNIQUENESS_PRINT", "")).lower() in {"1", "true", "yes"}
        if pf:
            print("[NOVELTY UNIQUE] Critique:")
            for x in c[:10]:
                print("- ", x)
            print("[NOVELTY UNIQUE] Unique ideas:")
            for x in u[:10]:
                print("- ", x)
            print("[NOVELTY UNIQUE END]")
    except Exception:
        pass
    return {"unique_ideas": u, "critique": c}


def _normalize_idea_fields(
    original: Dict[str, Any],
    updates: Dict[str, Any] | None,
    *,
    fallback_id: str,
) -> Dict[str, Any]:
    merged = dict(original)
    if isinstance(updates, dict):
        for key, value in updates.items():
            if value is None:
                continue
            if isinstance(value, list):
                merged[key] = [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(value, dict):
                merged[key] = value
            else:
                merged[key] = str(value).strip()
    if not str(merged.get("id", "")).strip():
        merged["id"] = str(fallback_id)
    for key in [
        "title",
        "novelty_kind",
        "why_novel",
        "spec_hint",
        "method",
        "risks",
        "delta_vs_prior",
        "compute_budget",
    ]:
        merged[key] = str(merged.get(key, "") or "").strip()
    for key in ["eval_plan", "derived_from_titles", "domain_tags", "task_tags"]:
        vals = merged.get(key)
        if isinstance(vals, list):
            norm = [str(v).strip() for v in vals if str(v).strip()]
        elif vals in (None, ""):
            norm = []
        else:
            norm = [str(vals).strip()]
        merged[key] = norm
    try:
        merged["tier1_score"] = float(merged.get("tier1_score", original.get("tier1_score", 0.0)) or 0.0)
    except Exception:
        merged["tier1_score"] = float(original.get("tier1_score", 0.0) or 0.0)
    try:
        merged["dup_sim"] = float(merged.get("dup_sim", original.get("dup_sim", 0.0)) or 0.0)
    except Exception:
        merged["dup_sim"] = float(original.get("dup_sim", 0.0) or 0.0)
    return merged


def _critique_and_upgrade_ideas(
    panel: Dict[str, Any],
    summary_objs: List[Dict[str, Any]],
    *,
    facets: Dict[str, Any] | None = None,
    persona_notes: List[str] | None = None,
) -> Dict[str, Any]:
    ideas = list(panel.get("new_ideas_detailed") or [])
    if not ideas:
        return panel
    facets = facets or {}
    persona_notes = persona_notes or []
    try:
        NOVELTY_IDEA_CRITIQUES.unlink(missing_ok=True)
    except Exception:
        pass
    goal = str(get("project.goal", "your goal") or "your goal")
    allowed_models: List[str] = []
    if facets:
        allowed_models.extend([str(b) for b in facets.get("backbones", []) if str(b)])
    try:
        quality_cutoff = float(get("pipeline.novelty.quality_cutoff", 0.68) or 0.68)
    except Exception:
        quality_cutoff = 0.68
    try:
        min_ideas = int(get("pipeline.novelty.min_ideas", 3) or 3)
    except Exception:
        min_ideas = 3
    model = get("pipeline.novelty.model", None)
    profile = get("pipeline.novelty.llm", None)

    title_index: Dict[str, Dict[str, Any]] = {}
    for summary in summary_objs:
        try:
            title = str(summary.get("title") or "").strip()
            if title:
                title_index[title.lower()] = summary
                allowed_models.extend([str(m) for m in summary.get("methods", []) if str(m)])
        except Exception:
            continue
    allowed_models = sorted({m for m in allowed_models if m})

    review_summaries: List[Dict[str, Any]] = []
    upgraded: List[Dict[str, Any]] = []
    for idx, idea in enumerate(ideas, start=1):
        support: List[Dict[str, Any]] = []
        support_methods: List[str] = []
        support_datasets: List[str] = []
        for title in idea.get("derived_from_titles") or []:
            key = str(title or "").strip().lower()
            if not key:
                continue
            match = title_index.get(key)
            if match:
                support.append({
                    "title": match.get("title", ""),
                    "novelty_claims": match.get("novelty_claims") or [],
                    "methods": match.get("methods") or [],
                    "limitations": match.get("limitations") or [],
                    "datasets": match.get("datasets") or [],
                })
                support_methods.extend([str(m) for m in match.get("methods") or [] if str(m)])
                support_datasets.extend([str(d) for d in match.get("datasets") or [] if str(d)])
        payload = {
            "goal": goal,
            "quality_cutoff": quality_cutoff,
            "idea": idea,
            "score_estimate": float(idea.get("tier1_score", 0.0) or 0.0),
            "dup_sim": float(idea.get("dup_sim", 0.0) or 0.0),
            "persona_notes": persona_notes,
            "facets": facets,
            "support": support,
            "allowed_models": sorted({m for m in (allowed_models + support_methods) if m}),
            "allowed_datasets": sorted({d for d in support_datasets if d}),
            "output_schema": {
                "decision": "keep|revise|drop",
                "headline": "string",
                "rationale": ["string"],
                "improvements": ["string"],
                "upgraded_idea": {
                    "title": "string",
                    "novelty_kind": "string",
                    "why_novel": "string",
                    "spec_hint": "string",
                    "method": "string",
                    "risks": "string",
                    "eval_plan": ["string"],
                    "compute_budget": "string",
                    "derived_from_titles": ["string"],
                    "delta_vs_prior": "string",
                    "domain_tags": ["string"],
                    "task_tags": ["string"],
                },
            },
        }
        system_lines = [
            "You are a ruthless novelty director vetting research ideas under tight budgets.",
            "Use only the provided support evidence—no external assumptions.",
            "KEEP only when the idea clearly exceeds quality_cutoff, cites >=2 support titles, and includes >=3 numeric spec tokens.",
            "Novelty must focus on MODEL/architecture design (set novelty_kind='architecture').",
        ]
        allowed_models_line = sorted({m for m in (allowed_models + support_methods) if m})
        if allowed_models_line:
            system_lines.append(f"Use model families drawn from these sources: {', '.join(allowed_models_line[:8])}.")
        allowed_datasets_line = sorted({d for d in support_datasets if d})
        if allowed_datasets_line:
            system_lines.append(f"Datasets must come from cited papers: {', '.join(allowed_datasets_line[:8])}.")
        system_lines.extend([
            "Otherwise choose REVISE and return an upgraded_idea sharpening novelty, grounding, and specs.",
            "Use DROP only if the idea is untenable AND you provide an upgraded_idea replacement.",
            "Return JSON matching output_schema exactly.",
        ])
        system = "\n".join(system_lines)
        try:
            review = chat_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.0, model=model, profile=profile)
        except LLMError as exc:
            notes = [f"LLM critique failed: {exc}"]
            idea_copy = dict(idea)
            idea_copy["idea_review"] = {
                "decision": "error",
                "headline": "critique error",
                "notes": notes,
                "improvements": [],
                "quality_cutoff": quality_cutoff,
            }
            upgraded.append(idea_copy)
            append_jsonl(NOVELTY_IDEA_CRITIQUES, {
                "idea_index": idx,
                "status": "error",
                "error": str(exc),
                "idea": idea,
            })
            review_summaries.append({
                "id": idea_copy.get("id"),
                "decision": "error",
                "score": float(idea.get("tier1_score", 0.0) or 0.0),
                "dup_sim": float(idea.get("dup_sim", 0.0) or 0.0),
                "headline": "critique error",
                "notes": notes,
            })
            continue

        decision = str(review.get("decision") or "").strip().lower()
        if decision not in {"keep", "revise", "drop", "upgrade", "replace"}:
            decision = "revise" if float(idea.get("tier1_score", 0.0) or 0.0) < quality_cutoff else "keep"
        notes = review.get("rationale") or review.get("critique") or []
        if isinstance(notes, str):
            notes = [notes]
        improvements = review.get("improvements") or []
        if isinstance(improvements, str):
            improvements = [improvements]
        updated_payload = review.get("upgraded_idea") if isinstance(review.get("upgraded_idea"), dict) else None
        if decision == "drop" and updated_payload is None:
            decision = "revise"

        merged = _normalize_idea_fields(
            idea,
            updated_payload,
            fallback_id=str(idea.get("id") or idx),
        )
        merged["novelty_kind"] = "architecture"
        if allowed_models:
            method_lower = merged.get("method", "").lower()
            if not any(am.lower() in method_lower for am in allowed_models):
                merged["method"] = (allowed_models[0] + "-based architecture: " + merged.get("method", "")).strip()
        merged["idea_review"] = {
            "decision": decision,
            "headline": str(review.get("headline") or ""),
            "notes": [str(n) for n in notes if str(n).strip()],
            "improvements": [str(n) for n in improvements if str(n).strip()],
            "quality_cutoff": quality_cutoff,
        }
        upgraded.append(merged)
        append_jsonl(NOVELTY_IDEA_CRITIQUES, {
            "idea_index": idx,
            "decision": decision,
            "score_before": float(idea.get("tier1_score", 0.0) or 0.0),
            "dup_sim": float(idea.get("dup_sim", 0.0) or 0.0),
            "review": review,
            "final_idea": merged,
        })
        review_summaries.append({
            "id": merged.get("id"),
            "decision": decision,
            "score": float(idea.get("tier1_score", 0.0) or 0.0),
            "dup_sim": float(idea.get("dup_sim", 0.0) or 0.0),
            "headline": merged["idea_review"].get("headline", ""),
            "notes": merged["idea_review"].get("notes", [])[:3],
        })

    existing_ids = {str(it.get("id")) for it in upgraded}
    if len(upgraded) < min_ideas:
        fallback = sorted(
            ideas,
            key=lambda it: float(it.get("tier1_score", 0.0) or 0.0),
            reverse=True,
        )
        for item in fallback:
            if len(upgraded) >= min_ideas:
                break
            iid = str(item.get("id") or len(upgraded) + 1)
            if iid in existing_ids:
                continue
            copy_item = dict(item)
            copy_item["idea_review"] = {
                "decision": "fallback",
                "headline": "added to satisfy min_ideas",
                "notes": [],
                "improvements": [],
                "quality_cutoff": quality_cutoff,
            }
            upgraded.append(copy_item)
            existing_ids.add(iid)
            review_summaries.append({
                "id": copy_item.get("id"),
                "decision": "fallback",
                "score": float(item.get("tier1_score", 0.0) or 0.0),
                "dup_sim": float(item.get("dup_sim", 0.0) or 0.0),
                "headline": "added to satisfy min_ideas",
                "notes": [],
            })

    panel["new_ideas_detailed"] = upgraded
    panel["new_ideas"] = [it.get("title", "") for it in upgraded]
    panel["idea_reviews"] = review_summaries
    return panel


def main() -> None:
    _ensure_dirs()
    try:
        NOVELTY_IDEA_CRITIQUES.unlink(missing_ok=True)
    except Exception:
        pass
    # Progress printing control (default ON unless explicitly disabled)
    try:
        _progress = True
        try:
            _progress = bool(get("pipeline.novelty.print_progress", True))
        except Exception:
            _progress = True
        envp = str(os.getenv("NOVELTY_PROGRESS", "")).lower()
        if envp in {"0", "false", "no", "off"}:
            _progress = False
        elif envp in {"1", "true", "yes", "on"}:
            _progress = True
    except Exception:
        _progress = True
    # Load existing summaries from disk
    if not SUM_DIR.exists():
        print(f"[ERR] Missing summaries dir: {SUM_DIR}. Run agents.summarize first.")
        return
    summary_files = sorted([p for p in SUM_DIR.glob("*.json") if p.is_file()])
    if not summary_files:
        print(f"[ERR] No summary JSONs found in {SUM_DIR}. Run agents.summarize first.")
        return
    summaries: List[Dict[str, Any]] = []
    for p in summary_files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and isinstance(rec.get("summary"), dict):
                summaries.append(rec)
        except Exception:
            continue
    if _progress:
        try:
            print(f"[NOVELTY] Loaded {len(summaries)} summaries from {SUM_DIR}")
        except Exception:
            pass

    # Optional: build FAISS-GPU index and compute neighbors among papers
    try:
        from lab.config import get as _cfg_get
        emb_enable = bool(_cfg_get("embeddings.enable", False))
        ret_enable = bool(_cfg_get("embeddings.retrieval.enable", False))
    except Exception:
        emb_enable = False
        ret_enable = False
    if emb_enable and ret_enable:
        from utils.embeddings import build_faiss_cpu_index, build_faiss_gpu_index
        model_name = str(_cfg_get("embeddings.model", "google/embeddinggemma-300m") or "google/embeddinggemma-300m")
        index_type = str(_cfg_get("embeddings.retrieval.index", "faiss-cpu") or "faiss-cpu").strip().lower()
        if _progress:
            try:
                print(f"[NOVELTY] Building {index_type.upper()} index for embeddings model '{model_name}' …")
            except Exception:
                pass
        if index_type == "faiss-cpu":
            index, vec_paths = build_faiss_cpu_index(model_name)
        elif index_type == "faiss-gpu":
            index, vec_paths = build_faiss_gpu_index(model_name)
        else:
            raise RuntimeError(f"unknown retrieval index: {index_type}")
        # Load vectors back (aligned with index IDs)
        X = _np.vstack([_np.load(p).astype(_np.float32) for p in vec_paths])
        D, I = index.search(X, int(_cfg_get("embeddings.retrieval.top_k", 8) or 8) + 1)
        # Build neighbor map excluding self (first neighbor)
        neighbors = {}
        keys = [pathlib.Path(p).stem for p in vec_paths]
        for row, key in enumerate(keys):
            ids = list(I[row])
            sims = list(D[row])
            out = []
            for j, idx in enumerate(ids):
                if idx == row:
                    continue  # skip self
                out.append({"key": keys[idx], "sim": float(sims[j])})
            neighbors[key] = out
        (NOVELTY_DIR / "retrieval_neighbors.json").write_text(json.dumps(neighbors, ensure_ascii=False, indent=2), encoding="utf-8")
        if _progress:
            try:
                print(f"[NOVELTY] Wrote retrieval neighbors: {NOVELTY_DIR / 'retrieval_neighbors.json'}")
            except Exception:
                pass

    # Build facets from your summaries and persist for audit (used by personas too)
    facets = extract_facets([s["summary"] for s in summaries], persist=True)
    if _progress:
        try:
            print(
                "[NOVELTY] Mined facets: backbones=%d, datasets=%d, ops=%d"
                % (
                    len(facets.get("backbones", [])),
                    len(facets.get("datasets", [])),
                    len(facets.get("ops", [])),
                )
            )
        except Exception:
            pass

    # Optional persona discussion to inform clustering (with facets context)
    persona_notes: List[str] = _persona_discussion([s["summary"] for s in summaries], facets=facets)
    if _progress:
        try:
            print(f"[NOVELTY] Persona notes recorded: {len(persona_notes)}")
        except Exception:
            pass

    # Temperature sampling info (suppress temps for GPT-5)
    try:
        _temps = get("pipeline.novelty.sampling.temperatures", [0.2]) or [0.2]
        if isinstance(_temps, str):
            _temps = [float(t) for t in str(_temps).split(",") if t.strip()]
    except Exception:
        _temps = [0.2]
    try:
        _prov = str(get("llm.provider", LLM_PROVIDER) or LLM_PROVIDER).lower()
        _mod = str((get("pipeline.novelty.model", None) or get("llm.model", LLM_MODEL) or LLM_MODEL)).lower()
        if _prov == "openai" and (_mod.startswith("gpt-5-") or _mod == "gpt-5"):
            _temps = [0.0]
            if _progress:
                try:
                    print("[NOVELTY] Synthesizing themes/ideas (GPT-5: no temperature)")
                except Exception:
                    pass
        else:
            if _progress:
                try:
                    print(f"[NOVELTY] Synthesizing themes/ideas (temps={_temps})")
                except Exception:
                    pass
    except Exception:
        if _progress:
            try:
                print(f"[NOVELTY] Synthesizing themes/ideas (temps={_temps})")
            except Exception:
                pass

    try:
        panel = group_novelty_and_ideas_v2([s["summary"] for s in summaries], persona_notes=persona_notes, facets=facets)
    except LLMError as exc:
        print(f"[ERR] LLM group discussion failed: {exc}")
        panel = {"themes": [], "new_ideas": [], "new_ideas_detailed": []}
    if _progress:
        try:
            print(
                "[NOVELTY] Synthesized: themes=%d, ideas_detailed=%d"
                % (len(panel.get("themes") or []), len(panel.get("new_ideas_detailed") or []))
            )
        except Exception:
            pass

    # Gate bad ideas
    try:
        _before = len(panel.get("new_ideas_detailed") or [])
        panel = score_and_filter_ideas(panel, facets)
        _after = len(panel.get("new_ideas_detailed") or [])
        if _progress:
            try:
                print(f"[NOVELTY] Ideas kept after scoring: {_after}/{_before}")
                if _after == 0:
                    print("[NOVELTY] No ideas remain after scoring (no fallback enabled)")
            except Exception:
                pass
    except Exception:
        pass

    try:
        panel = _critique_and_upgrade_ideas(
            panel,
            [s["summary"] for s in summaries],
            facets=facets,
            persona_notes=persona_notes,
        )
        if _progress:
            try:
                decision_counts: Dict[str, int] = {}
                for entry in panel.get("idea_reviews", []) or []:
                    key = str(entry.get("decision") or "unknown").strip() or "unknown"
                    decision_counts[key] = decision_counts.get(key, 0) + 1
                if decision_counts:
                    stats_line = ", ".join(f"{k}={v}" for k, v in sorted(decision_counts.items()))
                    print(f"[NOVELTY] Idea critique decisions: {stats_line}")
            except Exception:
                pass
        try:
            _before = len(panel.get("new_ideas_detailed") or [])
            panel = score_and_filter_ideas(panel, facets)
            kept_ids = {str(it.get("id")) for it in panel.get("new_ideas_detailed") or []}
            panel["idea_reviews"] = [
                entry for entry in (panel.get("idea_reviews") or [])
                if str(entry.get("id")) in kept_ids
            ]
            _after = len(panel.get("new_ideas_detailed") or [])
            if _progress:
                try:
                    print(f"[NOVELTY] Ideas kept post-critique: {_after}/{_before}")
                    if _after == 0:
                        print("[NOVELTY] No ideas remain after critique scoring (no fallback enabled)")
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARN] Idea critique stage failed: {exc}")

    # Research outline and citations
    try:
        cites = _load_citations(pathlib.Path("abstract_screen_deepseek.csv"))
        outline = derive_research_outline(panel, cites, persona_notes=persona_notes)
    except LLMError as exc:
        print(f"[WARN] LLM research outline failed: {exc}")
        outline = {"problems": [], "objectives": [], "contributions": [], "research_questions": [], "citations": []}
    if _progress:
        try:
            print(
                "[NOVELTY] Outline: problems=%d, objectives=%d, contributions=%d, questions=%d"
                % (
                    len(outline.get("problems") or []),
                    len(outline.get("objectives") or []),
                    len(outline.get("contributions") or []),
                    len(outline.get("research_questions") or []),
                )
            )
        except Exception:
            pass

    # Optional uniqueness reflection (novelty differentiation)
    uniq = _uniqueness_reflection(panel, persona_notes=persona_notes)
    if _progress:
        try:
            print(f"[NOVELTY] Uniqueness reflection: unique_ideas={len(uniq.get('unique_ideas') or [])}")
        except Exception:
            pass

    final = {
        "num_papers": len(summaries),
        "themes": panel.get("themes", []),
        "new_ideas": panel.get("new_ideas", []),
        "novelty_ideas": panel.get("new_ideas_detailed", []),
        "unique_ideas": uniq.get("unique_ideas", []),
        "problems": outline.get("problems", []),
        "objectives": outline.get("objectives", []),
        "contributions": outline.get("contributions", []),
        "research_questions": outline.get("research_questions", []),
        "citations": outline.get("citations", []),
    }
    # Ensure Tier-1 validator sees your sentences verbatim
    try:
        final = _inject_user_claims(final)
    except Exception:
        pass
    # Optional: produce an ideas-only report
    try:
        only_ideas = bool(get("pipeline.novelty.only_ideas", False))
        env_only = str(os.getenv("NOVELTY_ONLY_IDEAS", "")).lower()
        if env_only in {"1", "true", "yes", "on"}:
            only_ideas = True
        elif env_only in {"0", "false", "no", "off"}:
            pass
    except Exception:
        only_ideas = False
    if only_ideas:
        final = {"novelty_ideas": final.get("novelty_ideas", [])}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    if _progress:
        try:
            print(f"[NOVELTY] Report path: {REPORT_PATH}")
        except Exception:
            pass
    if is_verbose():
        try:
            theme_names = [str(t.get("name") or "") for t in (final.get("themes") or [])]
            vprint("Novelty summary: num_papers=" + str(final.get("num_papers")) + 
                   ", themes=" + ", ".join([t for t in theme_names if t]))
        except Exception:
            pass
    print(f"[DONE] Wrote novelty report: {REPORT_PATH}")

    # Optional Tier‑1 Validator
    try:
        from tier1_validator import run_tier1_validator  # type: ignore
    except Exception:
        run_tier1_validator = None  # type: ignore
    try:
        enable_tier1 = False
        try:
            # YAML: tier1.enable; Env: ENABLE_TIER1=1
            enable_tier1 = bool(get("tier1.enable", False)) or (str(os.getenv("ENABLE_TIER1", "")).lower() in {"1", "true", "yes"})
        except Exception:
            enable_tier1 = False
        if run_tier1_validator and enable_tier1:
            try:
                _ = run_tier1_validator(final)
            except Exception as e:
                print(f"[WARN] Tier‑1 validator failed: {e}")
    except Exception:
        pass

    # Write light literature stats
    try:
        stats = {
            "num_pdfs": len(summary_files),
            "num_summaries": len(summaries),
            "num_themes": len(final.get("themes") or []),
            "num_new_ideas": len(final.get("new_ideas") or []),
        }
        (NOVELTY_DIR / "lit_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
