"""Cache utilities for summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/summary_cache")


def cache_path_for(pdf_filename: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf_filename).stem
    return CACHE_DIR / f"{stem}.txt"


def load_cached_text(pdf_filename: str) -> str | None:
    path = cache_path_for(pdf_filename)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def save_cached_text(pdf_filename: str, text: str) -> None:
    try:
        path = cache_path_for(pdf_filename)
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def write_quality_report(out_path: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "count": len(records),
        "stats": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["load_cached_text", "save_cached_text", "write_quality_report"]
