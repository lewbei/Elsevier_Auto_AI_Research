"""Convenience helpers for inspecting persona transcripts and recent run logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
PERSONA_LOG = DATA_DIR / "persona_conversations.jsonl"
NOVELTY_SESSION = DATA_DIR / "novelty" / "novelty_session.jsonl"
RUNS_DIR = ROOT / "runs"


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                records.append(rec)
        if limit is not None:
            return records[-max(0, limit) :]
        return records
    except Exception:
        return []


def tail_persona_log(limit: int = 10, phase: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the most recent persona conversation entries."""
    records = _read_jsonl(PERSONA_LOG, limit=None)
    if phase:
        phase_l = phase.strip().lower()
        records = [r for r in records if str(r.get("phase", "")).lower() == phase_l]
    if limit:
        return records[-max(0, limit) :]
    return records


def format_persona_entry(entry: Dict[str, Any]) -> str:
    """Format a persona log entry into a compact string."""
    phase = str(entry.get("phase") or "")
    ts = str(entry.get("ts") or "")
    notes = entry.get("notes") or []
    if not isinstance(notes, list):
        notes = [notes]
    snippet = " | ".join(str(n)[:180] for n in notes[:4])
    return f"[{ts}] {phase}: {snippet}"


def tail_novelty_session(limit: int = 20, phase: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the tail of novelty persona session logs (agents.novelty)."""
    records = _read_jsonl(NOVELTY_SESSION, limit=None)
    if phase:
        phase_l = phase.strip().lower()
        records = [r for r in records if str(r.get("phase", "")).lower() == phase_l]
    if limit:
        return records[-max(0, limit) :]
    return records


def load_run_cost_reports(limit: int = 5) -> List[Dict[str, Any]]:
    """Load recent LLM cost summaries from runs/llm_cost_*.json."""
    if not RUNS_DIR.exists():
        return []
    reports: List[Dict[str, Any]] = []
    try:
        files = sorted(RUNS_DIR.glob("llm_cost_*.json"))
        if limit:
            files = files[-max(0, limit) :]
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["path"] = str(path)
                reports.append(data)
            except Exception:
                continue
    except Exception:
        return []
    return reports


def summarize_run_costs(limit: int = 5) -> Dict[str, Any]:
    """Aggregate recent LLM cost reports into a concise summary."""
    reports = load_run_cost_reports(limit=limit)
    total_cost = sum(float(r.get("cost") or 0.0) for r in reports)
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in reports)
    return {
        "runs": reports,
        "total_cost": round(total_cost, 4),
        "total_tokens": int(total_tokens),
    }


__all__ = [
    "tail_persona_log",
    "format_persona_entry",
    "tail_novelty_session",
    "load_run_cost_reports",
    "summarize_run_costs",
]
