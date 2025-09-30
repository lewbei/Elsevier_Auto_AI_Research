"""Result Analyzer: transforms raw run outputs into insights.

Reads runs/summary.json + (optional) runs/best.json to compute:
- baseline vs novelty delta
- effect size (Cohen's d) when repeats present
- simple hyperparameter sensitivity (correlation with accuracy)
- top variants ranking
Produces:
  runs/analysis.json
  runs/insights.md (human-readable summary)

Config flags:
  feedback.analysis.enable or feedback.enable

Robust to missing files; fails soft.
"""
from __future__ import annotations
import json, math, statistics, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from lab.config import get_bool
except Exception:
    def get_bool(key: str, default: bool=False) -> bool:  # type: ignore
        return bool(default)

RUNS_DIR = Path('runs')


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def _cohens_d(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    try:
        mean_a, mean_b = statistics.mean(a), statistics.mean(b)
        var_a = statistics.pvariance(a)
        var_b = statistics.pvariance(b)
        # pooled std
        pooled = math.sqrt(((len(a)-1)*var_a + (len(b)-1)*var_b) / max(1, (len(a)+len(b)-2)))
        if pooled == 0:
            return 0.0
        return (mean_b - mean_a) / pooled
    except Exception:
        return 0.0


def analyze(summary: Dict[str, Any]) -> Dict[str, Any]:
    runs = summary.get('runs') or []
    # Collect baseline/novelty groups (including repeats)
    base_acc: List[float] = []
    nov_acc: List[float] = []
    records: List[Tuple[str,float,Dict[str,Any]]] = []
    for r in runs:
        try:
            name = str(r.get('name'))
            acc = float(r.get('result',{}).get('metrics',{}).get('val_accuracy', 0.0) or 0.0)
            records.append((name, acc, r.get('spec', {})))
            if name.startswith('baseline'):
                base_acc.append(acc)
            if name.startswith('novelty'):
                nov_acc.append(acc)
        except Exception:
            continue
    delta = (max(nov_acc) - max(base_acc)) if (base_acc and nov_acc) else 0.0
    effect = _cohens_d(base_acc, nov_acc)
    # Hyperparameter sensitivity (simple Pearson on lr, max_train_steps, input_size)
    def collect_param(param: str) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for name, acc, spec in records:
            try:
                if param in spec:
                    xs.append(float(spec.get(param)))
                    ys.append(acc)
            except Exception:
                pass
        return xs, ys
    import math
    def corr(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 3 or len(xs) != len(ys):
            return 0.0
        try:
            mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
            num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
            denom = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
            if denom == 0:
                return 0.0
            return num/denom
        except Exception:
            return 0.0
    sensitivity = {}
    for p in ['lr','max_train_steps','input_size']:
        xs, ys = collect_param(p)
        sensitivity[p] = corr(xs, ys)
    top = sorted(records, key=lambda t: t[1], reverse=True)[:5]
    insights = {
        'runs_total': len(runs),
        'baseline_best': max(base_acc) if base_acc else 0.0,
        'novelty_best': max(nov_acc) if nov_acc else 0.0,
        'delta_best': delta,
        'effect_size_d': effect,
        'sensitivity': sensitivity,
        'top_variants': [ {'name': n, 'val_accuracy': a} for n,a,_ in top ],
    }
    return insights


def write_outputs(insights: Dict[str, Any]) -> None:
    try:
        (RUNS_DIR / 'analysis.json').write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    # Markdown companion
    try:
        lines = [
            '# Run Insights', '',
            f"Total runs: {insights.get('runs_total')}",
            f"Best baseline acc: {insights.get('baseline_best'):.4f}",
            f"Best novelty acc: {insights.get('novelty_best'):.4f}",
            f"Delta (best novelty - best baseline): {insights.get('delta_best'):.4f}",
            f"Effect size (Cohen d): {insights.get('effect_size_d'):.3f}",
            '## Hyperparameter Sensitivity',
        ]
        sens = insights.get('sensitivity') or {}
        for k,v in sens.items():
            lines.append(f"- {k}: corr={v:.3f}")
        lines.append('\n## Top Variants')
        for r in insights.get('top_variants', []):
            lines.append(f"- {r['name']}: acc={r['val_accuracy']:.4f}")
        (RUNS_DIR / 'insights.md').write_text('\n'.join(lines), encoding='utf-8')
    except Exception:
        pass


def main() -> None:
    enabled = get_bool('feedback.analysis.enable', get_bool('feedback.enable', False))
    if not enabled:
        print('[ANALYZER] Disabled by config.')
        return
    summ_path = RUNS_DIR / 'summary.json'
    if not summ_path.exists():
        print('[ANALYZER] summary.json missing.')
        return
    summary = _load_json(summ_path) or {}
    ins = analyze(summary)
    write_outputs(ins)
    print('[ANALYZER] Wrote analysis artifacts.')

if __name__ == '__main__':
    main()
