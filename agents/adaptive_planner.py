"""Adaptive Planner

Transforms a normalized decision (from decision_engine) and existing specs into
new speculative variants to be scheduled in subsequent iteration cycles.

Core function: adapt_plan(original_specs, decision) -> List[Dict]

Where original_specs is a dict mapping role -> spec (e.g., {'baseline': {...}, 'novelty': {...}})
Decision schema: see decision_engine.make_experiment_decision

Heuristics (initial):
  - adjust_lr: apply factor to novelty + create lr_down and lr_up variants
  - regularize: increase dropout or add 'weight_decay'
  - increase_capacity: bump model width surrogate (e.g., add flag 'widen_head')
  - explore_hparam: produce small lr increase + minor batch size change

Outputs include provenance metadata under key '_adapted_from'.
"""
from __future__ import annotations

from typing import Dict, Any, List
import copy


def _with_provenance(spec: Dict[str, Any], decision: Dict[str, Any], tag: str) -> Dict[str, Any]:
    s = copy.deepcopy(spec)
    s.setdefault('_adapted_from', {})
    s['_adapted_from'].update({
        'decision_type': decision.get('type'),
        'tag': tag,
        'rationale': decision.get('rationale'),
    })
    return s


def adapt_plan(original_specs: Dict[str, Dict[str, Any]], decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not decision:
        return []
    atype = decision.get('type')
    params = decision.get('params', {})
    out: List[Dict[str, Any]] = []
    novelty = original_specs.get('novelty') or original_specs.get('baseline') or {}

    if atype == 'adjust_lr':
        factor = float(params.get('factor', 0.5))
        base_lr = float(novelty.get('lr', 1e-3))
        lr_down = _with_provenance(novelty, decision, 'lr_down')
        lr_down['lr'] = max(1e-6, base_lr * factor)
        lr_up = _with_provenance(novelty, decision, 'lr_up')
        lr_up['lr'] = base_lr * 1.2
        out.extend([lr_down, lr_up])
    elif atype == 'regularize':
        reg = _with_provenance(novelty, decision, 'reg_up')
        reg['dropout_p'] = float(novelty.get('dropout_p', 0.2)) + params.get('dropout_delta', 0.1)
        reg['weight_decay'] = float(novelty.get('weight_decay', 0.0)) + 1e-4
        out.append(reg)
    elif atype == 'increase_capacity':
        cap = _with_provenance(novelty, decision, 'capacity_plus')
        arch = dict(cap.get('arch') or {})
        # Default to simple_cnn if not specified
        arch.setdefault('type', arch.get('type', 'simple_cnn'))
        width = int(arch.get('width', 64))
        depth = int(arch.get('depth', 4))
        arch['width'] = min(1024, int(width * 1.25))
        arch['depth'] = min(32, depth + 1)
        cap['arch'] = arch
        # Optionally keep generated head if previously used
        cap['use_generated_head'] = True
        out.append(cap)
    elif atype == 'explore_hparam':
        h1 = _with_provenance(novelty, decision, 'lr_small_up')
        h1['lr'] = float(novelty.get('lr', 1e-3)) * 1.15
        h2 = _with_provenance(novelty, decision, 'batch_alt')
        h2['batch_size'] = int(novelty.get('batch_size', 16)) + 8
        out.extend([h1, h2])
    else:
        # Unknown action: no adaptation
        return []

    # Deduplicate by (tag, lr, batch_size) simple hash
    seen = set()
    unique: List[Dict[str, Any]] = []
    for s in out:
        key = (
            s.get('_adapted_from', {}).get('tag'),
            s.get('lr'),
            s.get('batch_size'),
            s.get('dropout_p'),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


__all__ = ['adapt_plan']