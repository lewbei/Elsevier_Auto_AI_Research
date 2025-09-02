from __future__ import annotations

"""Spec normalization for training runs (no CLI required).

Design:
- If pydantic is available, use a lightweight model to coerce types and apply
  bounds/defaults. Otherwise, fall back to a manual sanitizer.
- Preserve unknown keys and novelty_component; keep behavior conservative.
"""

from typing import Any, Dict, Optional, List


def _manual_clamp(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(spec)

    def _truthy(x: Any) -> bool:
        return str(x).strip().lower() in {"1", "true", "yes", "on"}

    try:
        if "input_size" in s:
            s["input_size"] = int(max(96, min(512, int(s["input_size"]))))
    except Exception:
        s["input_size"] = 224
    try:
        if "lr" in s:
            s["lr"] = float(max(1e-5, min(1e-1, float(s["lr"]))))
        else:
            s["lr"] = 1e-3
    except Exception:
        s["lr"] = 1e-3
    try:
        if "max_train_steps" in s:
            s["max_train_steps"] = int(max(10, min(1000, int(s["max_train_steps"]))))
        else:
            s["max_train_steps"] = 50
    except Exception:
        s["max_train_steps"] = 50
    try:
        s["batch_size"] = int(max(1, min(1024, int(s.get("batch_size", 16)))))
    except Exception:
        s["batch_size"] = 16
    model = str(s.get("model", "resnet18") or "resnet18").strip().lower()
    s["model"] = model or "resnet18"
    opt = str(s.get("optimizer", "adam") or "adam").strip().lower()
    s["optimizer"] = opt if opt in {"adam", "sgd"} else "adam"
    # toggles
    s["deterministic"] = _truthy(s.get("deterministic", False))
    s["amp"] = _truthy(s.get("amp", False))
    try:
        nworkers = int(s.get("num_workers", 0) or 0)
        s["num_workers"] = max(0, min(16, nworkers))
    except Exception:
        s["num_workers"] = 0
    try:
        s["pin_memory"] = bool(s.get("pin_memory", False))
    except Exception:
        s["pin_memory"] = False
    try:
        s["persistent_workers"] = bool(s.get("persistent_workers", False))
    except Exception:
        s["persistent_workers"] = False
    try:
        if "log_interval" in s:
            s["log_interval"] = int(max(0, int(s["log_interval"])))
    except Exception:
        pass
    # novelty
    nc = s.get("novelty_component") or {}
    if not isinstance(nc, dict):
        nc = {"enabled": bool(nc)}
    nc.setdefault("enabled", False)
    s["novelty_component"] = nc
    # seed
    try:
        s["seed"] = int(s.get("seed", 42) or 42)
    except Exception:
        s["seed"] = 42
    return s


def normalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized copy of the spec. Optional Pydantic support.

    Keeps unknown keys; clamps common fields; ensures novelty_component exists.
    """
    try:
        # Optional validation path; keep dependency soft
        from pydantic import BaseModel, Field, validator  # type: ignore

        class _Spec(BaseModel):
            title: Optional[str] = None
            seed: int = 42
            input_size: int = Field(224, ge=96, le=512)
            batch_size: int = Field(16, ge=1, le=1024)
            epochs: int = Field(1, ge=1, le=100)
            lr: float = Field(1e-3, ge=1e-5, le=1e-1)
            max_train_steps: int = Field(50, ge=10, le=1000)
            optimizer: str = Field("adam")
            model: str = Field("resnet18")
            num_workers: int = Field(0, ge=0, le=16)
            pin_memory: bool = False
            persistent_workers: bool = False
            deterministic: bool = False
            amp: bool = False
            log_interval: int = Field(0, ge=0)
            novelty_component: Dict[str, Any] = Field(default_factory=dict)

            @validator("optimizer")
            def _opt_ok(cls, v: str) -> str:
                v = (v or "").strip().lower()
                return v if v in {"adam", "sgd"} else "adam"

        # Validate known fields; then merge unknowns back
        known_keys = set(_Spec.__fields__.keys())
        base = _Spec(**{k: v for k, v in spec.items() if k in known_keys}).dict()
        # Ensure novelty_component.enabled present
        nc = base.get("novelty_component", {}) or {}
        if not isinstance(nc, dict):
            nc = {"enabled": bool(nc)}
        nc.setdefault("enabled", nc.get("enabled", False))
        base["novelty_component"] = nc
        # Merge unknown keys back (pass-through)
        for k, v in spec.items():
            if k not in known_keys:
                base[k] = v
        return base
    except Exception:
        return _manual_clamp(spec)

