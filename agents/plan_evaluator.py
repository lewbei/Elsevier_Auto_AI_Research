"""Plan evaluator: scores single or multiple research plans.

Lightweight heuristic scoring so downstream orchestration can select strongest plan
before committing compute. No external dependencies.

Outputs JSON structure like:
{
  "plans": [
     {"path": "data/plan.json", "scores": {"feasibility": 0.82, "novelty_alignment": 0.67, ...}, "composite": 0.74}
  ],
  "best": { ... }
}

Scoring dimensions (heuristic, 0..1):
- feasibility: penalize >8 tasks, missing risks, or no stopping rules
- clarity: average length qualities; reward presence of hypotheses & milestones
- risk_balance: presence of at least one risk + mitigation
- novelty_alignment: whether novelty_focus mentions terms from hypotheses or tasks
- conciseness: inverse of total token-ish count (soft)

Composite = weighted average (feasibility 0.3, clarity 0.25, risk_balance 0.15, novelty_alignment 0.2, conciseness 0.1)

Config flags:
  feedback.plan_eval.enable or feedback.enable
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Dict, List

try:
    from lab.config import get_bool
except Exception:
    def get_bool(key: str, default: bool=False) -> bool:  # type: ignore
        return bool(default)

DATA_DIR = Path('data')

WEIGHTS = {
    'feasibility': 0.30,
    'clarity': 0.25,
    'risk_balance': 0.15,
    'novelty_alignment': 0.20,
    'conciseness': 0.10,
}


def _norm(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _score_plan(plan: Dict[str, Any]) -> Dict[str, float]:
    tasks = plan.get('tasks') or []
    risks = plan.get('risks') or []
    hypotheses = plan.get('hypotheses') or []
    stopping = plan.get('stopping_rules') or []
    novelty_focus = str(plan.get('novelty_focus') or '')
    milestones = plan.get('milestones') or []

    # Feasibility
    n_tasks = len(tasks)
    feasibility = 1.0
    if n_tasks == 0:
        feasibility = 0.2
    elif n_tasks > 10:
        feasibility = 0.4
    elif n_tasks > 8:
        feasibility = 0.6
    elif n_tasks > 6:
        feasibility = 0.75
    if not stopping:
        feasibility *= 0.6

    # Clarity: presence ratio across key fields (objective, hypotheses, tasks, success_criteria)
    clarity_keys = ['objective','hypotheses','tasks','success_criteria','datasets','baselines']
    present = 0
    for k in clarity_keys:
        v = plan.get(k)
        if isinstance(v, list) and v:
            present += 1
        elif isinstance(v, str) and v.strip():
            present += 1
    clarity = present / len(clarity_keys)
    if milestones:
        clarity = min(1.0, clarity + 0.1)

    # Risk balance
    risk_balance = 1.0 if risks else 0.4

    # Novelty alignment: overlap of novelty_focus tokens with hypotheses/tasks tokens
    import re
    def toks(s: str) -> List[str]:
        return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
    nf_t = set(toks(novelty_focus))
    hyp_t = set(t for h in hypotheses for t in toks(str(h)))
    task_t = set()
    for t in tasks:
        if isinstance(t, dict):
            for key in ['name','why']:
                task_t.update(toks(str(t.get(key,''))))
            steps = t.get('steps') or []
            for s in steps:
                task_t.update(toks(str(s)))
        else:
            task_t.update(toks(str(t)))
    overlap = len(nf_t & (hyp_t | task_t))
    novelty_alignment = 0.2 if not nf_t else min(1.0, overlap / max(1, len(nf_t)))

    # Conciseness: Penalize excessive token count across key arrays
    total_tokens = sum(len(toks(str(h))) for h in hypotheses) + sum(len(toks(str(r.get('risk','')))) for r in risks if isinstance(r, dict))
    conciseness = 1.0
    if total_tokens > 400:
        conciseness = 0.4
    elif total_tokens > 250:
        conciseness = 0.6
    elif total_tokens > 150:
        conciseness = 0.75

    out = {
        'feasibility': _norm(feasibility),
        'clarity': _norm(clarity),
        'risk_balance': _norm(risk_balance),
        'novelty_alignment': _norm(novelty_alignment),
        'conciseness': _norm(conciseness),
    }
    composite = sum(out[k]*WEIGHTS[k] for k in WEIGHTS)
    out['composite'] = round(composite, 4)
    return out


def evaluate_plans(paths: List[Path]) -> Dict[str, Any]:
    records = []
    for p in paths:
        try:
            js = json.loads(p.read_text(encoding='utf-8'))
            scores = _score_plan(js)
            records.append({'path': str(p), 'scores': scores, 'composite': scores['composite']})
        except Exception as exc:
            records.append({'path': str(p), 'error': str(exc), 'scores': {}, 'composite': 0.0})
    best = None
    if records:
        best = sorted(records, key=lambda r: r.get('composite',0.0), reverse=True)[0]
    return {'plans': records, 'best': best}


def main() -> None:
    # Autodetect plans: data/plans/*.json else data/plan.json
    multi_dir = DATA_DIR / 'plans'
    plan_files: List[Path] = []
    if multi_dir.exists():
        plan_files.extend(sorted(multi_dir.glob('plan_*.json')))
    single = DATA_DIR / 'plan.json'
    if single.exists() and not plan_files:
        plan_files.append(single)
    if not plan_files:
        print('[PLAN_EVAL] No plan files found.')
        return
    res = evaluate_plans(plan_files)
    out_path = DATA_DIR / 'plan_eval.json'
    try:
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[PLAN_EVAL] Wrote scores to {out_path}")
    except Exception as exc:
        print(f"[PLAN_EVAL] Failed to write results: {exc}")

if __name__ == '__main__':
    main()
