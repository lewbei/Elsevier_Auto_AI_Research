from __future__ import annotations
"""Lightweight YAML/JSON config loader with dotted-key access.

Config precedence (highest → lowest):
- YAML/JSON config file (CONFIG_FILE env or ./config.yaml|.yml|.json)
- Environment variables (for a few compatibility keys)
- Hardcoded defaults (only when both above are missing)

This keeps existing env-based behavior but allows a single project-level
YAML to override settings across agents. Parsing dependency (PyYAML) is
optional: if missing, JSON is supported; otherwise returns empty config.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import json

_CACHE: Optional[Dict[str, Any]] = None


def _find_config_file() -> Optional[Path]:
    cand = os.getenv("CONFIG_FILE") or os.getenv("CONFIG_PATH")
    if cand:
        p = Path(cand).expanduser().resolve()
        return p if p.exists() else None
    here = Path.cwd()
    for name in ("config.yaml", "config.yml", "config.json"):
        p = (here / name).resolve()
        if p.exists():
            return p
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        # YAML dependency missing; fall back to JSON if .json else empty
        if path.suffix.lower() == ".json":
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data or {}
    except Exception:
        return {}


def _load_config() -> Dict[str, Any]:
    p = _find_config_file()
    if not p:
        return {}
    if p.suffix.lower() in {".yaml", ".yml", ".json"}:
        return _load_yaml(p)
    # Unknown suffix; try JSON then YAML
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _load_yaml(p)


def get_config(refresh: bool = False) -> Dict[str, Any]:
    global _CACHE
    if refresh or _CACHE is None:
        _CACHE = _load_config()
    return _CACHE


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def get(key: str, default: Any = None) -> Any:
    """Get a config value from dotted key, or env fallback for a few keys."""
    cfg = get_config()
    if key:
        cur: Any = cfg
        for part in key.split('.'):
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        if cur is not None:
            return cur
    # selected env fallbacks for backward-compat
    if key == "dataset.name":
        return (os.getenv("DATASET") or default)
    if key == "dataset.path":
        return default
    if key == "dataset.allow_fallback":
        val = os.getenv("ALLOW_FALLBACK_DATASET")
        return _truthy(val) if val is not None else default
    if key == "dataset.allow_download":
        val = os.getenv("ALLOW_DATASET_DOWNLOAD")
        return _truthy(val) if val is not None else default
    return default


def get_bool(key: str, default: bool = False) -> bool:
    val = get(key, None)
    if val is None:
        return bool(default)
    return _truthy(val)


def dataset_name(default: str = "isic") -> str:
    name = str(get("dataset.name", "") or "").strip().lower()
    if name:
        return name
    env = str(os.getenv("DATASET", default) or default).strip().lower()
    return env if env in {"isic", "cifar10"} else default


def dataset_path_for(name: Optional[str] = None) -> str:
    n = (name or dataset_name()).lower()
    # YAML override takes precedence
    p = get("dataset.path", None)
    if p:
        return str(p)
    # defaults by dataset
    if n == "cifar10":
        return "data/cifar10"
    return "data/isic"


def dataset_kind(default: Optional[str] = None) -> Optional[str]:
    """Return dataset.kind from config if present (imagefolder|cifar10|custom)."""
    kind = get("dataset.kind", None)
    return str(kind).strip().lower() if kind else default


def dataset_splits() -> Dict[str, str]:
    """Return split names for imagefolder/custom kinds. Defaults to train/val[/test]."""
    splits = get("dataset.splits", None)
    if isinstance(splits, dict):
        out: Dict[str, str] = {}
        for k in ("train", "val", "test"):
            v = splits.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()
        if out:
            return out
    return {"train": "train", "val": "val"}
