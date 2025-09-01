import os
import json
import pathlib
from typing import Dict, Any, List
from dotenv import load_dotenv

from utils.llm_utils import chat_json, LLMError
from lab.config import get, get as get_cfg, get as _get
from lab.logging_utils import is_verbose, vprint, append_jsonl

try:
    # Optional: multi‑persona discussion helpers
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore


load_dotenv()

DATA_DIR = pathlib.Path("data")
SUM_DIR = DATA_DIR / "summaries"
REPORT_PATH = DATA_DIR / "novelty_report.json"
 
# Files for optional multi‑agent discussion logs
NOVELTY_SESSION = DATA_DIR / "novelty_session.jsonl"
NOVELTY_TRANSCRIPT = DATA_DIR / "novelty_transcript.md"
NOVELTY_UNIQ_JSON = DATA_DIR / "novelty_uniqueness.json"
NOVELTY_UNIQ_TRANSCRIPT = DATA_DIR / "novelty_uniqueness.md"


def _ensure_dirs() -> None:
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    user_payload = {
        "papers": items,
        # Optional extra guidance from multi‑persona discussion (if enabled)
        "persona_notes": list(persona_notes or []),
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
    model = get("pipeline.novelty.model", None)
    profile = get("pipeline.novelty.llm", None)
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.2, model=model, profile=profile)
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


def _persona_discussion(summaries: List[Dict[str, Any]]) -> List[str]:
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
        for s in summaries[:30]:  # cap to 30 papers for brevity
            ctx_items.append({
                "title": str(s.get("title") or "")[:120],
                "novelty_claims": (s.get("novelty_claims") or [])[:5],
                "methods": (s.get("methods") or [])[:5],
                "limitations": (s.get("limitations") or [])[:5],
            })
        dm = DialogueManager()
        dm.post(
            "User",
            (
                "We will discuss novelty themes across papers and propose novel, actionable ideas under tight compute.\n"
                "Provide role-aware, concrete observations: what clusters emerge, what is missing, risks, and specific crossovers to try."
            ),
        )
        dm.post("User", json.dumps({"papers": ctx_items}, ensure_ascii=False))

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

            # Proposals from proposers
            for p_idx, role in enumerate(proposers):
                prompt = (
                    f"From the perspective of {_alias_at(p_idx)}, propose 2 concrete NOVELTY directions grounded in the provided papers (NOT process or training logistics). "
                    "Each proposal must specify: NOVELTY_KIND:[architecture|training_objective|data|evaluation|augmentation|optimizer]; "
                    "COMPONENT_CHANGE:<one-sentence change at component level>; WHY_NOVEL:<why this is novel vs common baselines>; "
                    "SPEC_HINT:<minimal spec/code change to validate>. "
                    "Prefix each with 'PROPOSAL_NOVELTY:'. Constraints: <=1 epoch, small steps, feasible on CPU. Ask no questions yet."
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
                        f"From the perspective of {_alias_at(idx)}, ask 2 critical questions about the NOVELTY itself: originality vs prior work, overlap with common baselines, "
                        "risk of being only an augmentation/hparam tweak, and what ablation isolates the novelty effect. "
                        "Prefer questions that strengthen or falsify the novelty. Prefix each with 'QUESTION_NOVELTY:'. Do not propose new ideas in this step."
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
                        f"From the perspective of {_alias_at(idx)}, answer one outstanding QUESTION_NOVELTY concisely. "
                        "Prefix with 'ANSWER_NOVELTY:'. If none apply, clarify one assumption that strengthens novelty."
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
                # Consensus vote
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
                if agrees >= len(order):
                    agreed = True

            # Final consensus summary
            leader = order[1] if len(order) > 1 else order[0]
            r = dm.step_role(leader, prompt=(
                "Provide a CONSENSUS_NOVELTY_SUMMARY: one paragraph merging the best novelty proposal, with lines for NOVELTY_KIND and SPEC_HINT; must be constraints-aware."
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


def main() -> None:
    _ensure_dirs()
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

    # Optional persona discussion to inform clustering
    persona_notes: List[str] = _persona_discussion([s["summary"] for s in summaries])

    try:
        panel = group_novelty_and_ideas([s["summary"] for s in summaries], persona_notes=persona_notes)
    except LLMError as exc:
        print(f"[ERR] LLM group discussion failed: {exc}")
        panel = {"themes": [], "new_ideas": []}

    # Research outline and citations
    try:
        cites = _load_citations(pathlib.Path("abstract_screen_deepseek.csv"))
        outline = derive_research_outline(panel, cites, persona_notes=persona_notes)
    except LLMError as exc:
        print(f"[WARN] LLM research outline failed: {exc}")
        outline = {"problems": [], "objectives": [], "contributions": [], "research_questions": [], "citations": []}

    # Optional uniqueness reflection (novelty differentiation)
    uniq = _uniqueness_reflection(panel, persona_notes=persona_notes)

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
        (DATA_DIR / "lit_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
