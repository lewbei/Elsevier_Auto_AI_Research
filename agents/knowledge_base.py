"""Knowledge Base Stub

Appends experiment outcome records for future meta-learning.
Records stored as JSONL: data/knowledge/episodes.jsonl

record_experiment_outcome(plan, results, insights, decision=None)
  plan: dict of specs used
  results: list of run result dicts (iter_results entries)
  insights: optional analysis artifacts
  decision: optional adaptation decision applied
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import time

KB_DIR = Path('data/knowledge')
KB_FILE = KB_DIR / 'episodes.jsonl'


def _spec_hash(spec: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(spec, sort_keys=True)[:10000]
    except Exception:
        blob = str(spec)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def record_experiment_outcome(plan: Dict[str, Any], results: List[Dict[str, Any]], insights: Optional[Dict[str, Any]] = None, decision: Optional[Dict[str, Any]] = None) -> None:
    try:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            'ts': time.time(),
            'specs': {k: _spec_hash(v) for k, v in plan.items() if isinstance(v, dict)},
            'decision': decision,
            'summary': _summarize_results(results),
            'insights': insights or {},
        }
        with KB_FILE.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    best = None
    best_acc = -1.0
    for r in results:
        try:
            acc = float(r.get('result', {}).get('metrics', {}).get('val_accuracy', 0.0))
            if acc > best_acc:
                best = r
                best_acc = acc
        except Exception:
            continue
    return {
        'total_runs': len(results),
        'best_val_accuracy': best_acc if best_acc >= 0 else None,
        'best_name': best.get('name') if best else None,
    }


__all__ = ['record_experiment_outcome']