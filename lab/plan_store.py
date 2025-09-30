"""Helpers for reading and writing plan artifacts."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
PLANS_DIR = DATA_DIR / "plans"
PLAN_PATH = DATA_DIR / "plan.json"


def ensure_plan_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, payload: Dict[str, Any]) -> Path:
    ensure_plan_dirs()
    tmpdir = Path(tempfile.mkdtemp(prefix="plan_store_"))
    tmppath = tmpdir / path.name
    tmppath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if path.exists():
        backup = path.with_suffix(path.suffix + f".bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        try:
            shutil.copy2(path, backup)
        except Exception:
            pass
    shutil.move(str(tmppath), str(path))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return path


def write_active_plan(plan: Dict[str, Any]) -> Path:
    return _atomic_write(PLAN_PATH, plan)


def write_named_plan(plan: Dict[str, Any], filename: str) -> Path:
    ensure_plan_dirs()
    target = PLANS_DIR / filename
    return _atomic_write(target, plan)


def list_plan_files(include_active: bool = False) -> List[Path]:
    ensure_plan_dirs()
    files: List[Path] = []
    if include_active and PLAN_PATH.exists():
        files.append(PLAN_PATH)
    if PLANS_DIR.exists():
        for p in sorted(PLANS_DIR.glob("plan_*.json")):
            files.append(p)
    # remove duplicates while preserving order
    seen = set()
    unique: List[Path] = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def active_plan_exists() -> bool:
    return PLAN_PATH.exists()


def set_active_plan_from_path(src: Path) -> Path:
    data = src.read_text(encoding="utf-8")
    ensure_plan_dirs()
    tmpdir = Path(tempfile.mkdtemp(prefix="plan_store_"))
    tmppath = tmpdir / PLAN_PATH.name
    tmppath.write_text(data, encoding="utf-8")
    if PLAN_PATH.exists():
        backup = PLAN_PATH.with_suffix(PLAN_PATH.suffix + f".bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        try:
            shutil.copy2(PLAN_PATH, backup)
        except Exception:
            pass
    shutil.move(str(tmppath), str(PLAN_PATH))
    shutil.rmtree(tmpdir, ignore_errors=True)
    return PLAN_PATH


def read_plan(path: Path | None = None) -> Dict[str, Any]:
    target = path or PLAN_PATH
    if not target.exists():
        raise FileNotFoundError(f"Plan file not found: {target}")
    text = target.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        cleaned = re.sub(r"(^|\s)//.*?$", r"\1", cleaned, flags=re.M)
        cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)
        return json.loads(cleaned)


def snapshot_name(prefix: str = "plan") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"


__all__ = [
    "PLAN_PATH",
    "PLANS_DIR",
    "write_active_plan",
    "write_named_plan",
    "list_plan_files",
    "active_plan_exists",
    "set_active_plan_from_path",
    "read_plan",
    "snapshot_name",
]
