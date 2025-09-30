"""Decision Engine

Converts low-level monitor actions (heuristic signals) into enriched, scored decisions
that downstream adaptive planner can consume.

Design:
  - Stateless functional core (pure transformation of inputs)
  - Lightweight scoring (priority 1..5, confidence 0..1)
  - Action schema normalization

Schema (decision dict):
{
  'type': <string>,           # e.g., 'adjust_lr'
  'params': {...},            # implementation parameters
  'priority': int,            # 1=low .. 5=highest
  'confidence': float,        # 0..1 heuristic strength
  'rationale': str,           # human readable reason
  'evidence': {...},          # subset of trend metrics/signals
  'source': 'monitor',        # origin system
}
"""
from __future__ import annotations

from typing import Dict, Any, Optional


_BASE_PRIORITIES = {
    'adjust_lr': 4,
    'regularize': 4,
    'increase_capacity': 3,
    'explore_hparam': 2,
}


def _confidence_from_signals(signals: list[str]) -> float:
    if not signals:
        return 0.2
    score = 0.0
    for s in signals:
        if s.startswith('decline'):
            score += 0.4
        elif s.startswith('plateau'):
            score += 0.2
        elif s == 'potential_overfit':
            score += 0.3
    return max(0.1, min(1.0, score))


def make_experiment_decision(monitor_action: Dict[str, Any], trend: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Normalize monitor action -> enriched decision dict.

    monitor_action: {'type': 'adjust_lr', ...}
    trend: trend dict produced by monitor (optional)
    """
    if not monitor_action or 'type' not in monitor_action:
        return None
    atype = str(monitor_action.get('type'))
    params = {k: v for k, v in monitor_action.items() if k not in {'type', 'reason'}}
    rationale = monitor_action.get('reason', '')
    signals = []
    evidence: Dict[str, Any] = {}
    if trend:
        signals = list(trend.get('signals', []))
        # Keep a subset of numerical trend metrics
        for k in ['val_acc_slope', 'val_acc_last', 'val_acc_best', 'train_loss_slope', 'train_loss_last']:
            if k in trend:
                evidence[k] = trend[k]
    confidence = _confidence_from_signals(signals)
    priority = _BASE_PRIORITIES.get(atype, 1)
    # Escalate priority if multiple strong signals present
    if len(signals) >= 2 and any(s.startswith('decline') for s in signals):
        priority = min(5, priority + 1)
    decision = {
        'type': atype,
        'params': params,
        'priority': priority,
        'confidence': confidence,
        'rationale': rationale,
        'evidence': {'signals': signals, **evidence},
        'source': 'monitor',
    }
    return decision


__all__ = ['make_experiment_decision']