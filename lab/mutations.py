from __future__ import annotations
from typing import Dict, Any, List


def propose_mutations(spec: Dict[str, Any], max_k: int = 2) -> List[Dict[str, Any]]:
    """Generate small, safe mutations of a training spec.
    Does not clamp values; downstream verifiers/runners should apply bounds.
    """
    s = dict(spec)
    out: List[Dict[str, Any]] = []

    # 1) LR scaled variants
    try:
        lr = float(s.get("lr", 1e-3))
        for scale in (0.5, 1.5):
            m = dict(s)
            m["lr"] = max(1e-6, lr * scale)
            out.append(m)
    except Exception:
        pass

    # 2) Steps +/-
    try:
        steps = int(s.get("max_train_steps", 50))
        for delta in (-50, 50):
            m = dict(s)
            m["max_train_steps"] = max(10, steps + delta)
            out.append(m)
    except Exception:
        pass

    # 3) Input size +/-
    try:
        inp = int(s.get("input_size", 224))
        for delta in (-32, 32):
            m = dict(s)
            m["input_size"] = max(96, min(512, inp + delta))
            out.append(m)
    except Exception:
        pass

    # 4) Optimizer swap
    try:
        opt = str(s.get("optimizer", "adam")).lower()
        m = dict(s)
        m["optimizer"] = "sgd" if opt != "sgd" else "adam"
        out.append(m)
    except Exception:
        pass

    # De‑duplicate by (lr, steps, input_size, optimizer)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for m in out:
        key = (
            round(float(m.get("lr", 0.0)), 8),
            int(m.get("max_train_steps", 0)),
            int(m.get("input_size", 0)),
            str(m.get("optimizer", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq[: max(0, int(max_k))]

