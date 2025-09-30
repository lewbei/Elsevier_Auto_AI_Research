"""Shared persona helpers for pipeline stages."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lab.config import get_bool

try:
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore

ROOT = Path(__file__).parent.parent.resolve()
PERSONA_LOG = ROOT / "data" / "persona_conversations.jsonl"


def persona_enabled(config_key: Optional[str] = None, env_name: Optional[str] = None) -> bool:
    enabled = False
    if config_key:
        try:
            enabled = get_bool(config_key, False)
        except Exception:
            enabled = False
    if not enabled and env_name:
        try:
            value = str(os.getenv(env_name, "")).lower()
            enabled = value in {"1", "true", "yes", "on"}
        except Exception:
            enabled = False
    return enabled


def gather_notes(
    phase: str,
    context: Dict[str, Any],
    *,
    steps: int = 2,
    config_key: Optional[str] = None,
    env_name: Optional[str] = None,
) -> List[str]:
    if DialogueManager is None:
        return []
    if not persona_enabled(config_key, env_name):
        return []
    try:
        dm = DialogueManager()
        dm.post(
            "User",
            (
                f"Phase: {phase}. Provide structured, actionable guidance tailored to limited compute (<=1 epoch, small steps).\n"
                "Include: top priorities, key risks with mitigations, and minimal concrete actions."
            ),
        )
        dm.post("User", f"Context: {json.dumps(context, ensure_ascii=False)}")
        notes: List[str] = []
        for _ in range(max(1, steps)):
            r = dm.step_auto()
            payload = r.get("json") or {}
            if payload:
                notes.append(json.dumps(payload, ensure_ascii=False))
            else:
                notes.append(str(r))
        _log_notes(phase, notes)
        return notes
    except Exception:
        return []


def _log_notes(phase: str, notes: List[str]) -> None:
    if not notes:
        return
    try:
        PERSONA_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "phase": phase,
            "notes": notes,
        }
        with PERSONA_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


__all__ = ["persona_enabled", "gather_notes"]
