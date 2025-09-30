"""Central registry for stage artifacts/dependencies.

Each stage declares lightweight readiness checks so orchestrators can decide
whether to run, skip, or short-circuit gracefully. This keeps dependency logic
in one place and avoids scattering path calculations across agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

ROOT = Path(__file__).parent.parent.resolve()
DATA = ROOT / "data"
RUNS = ROOT / "runs"
PDF_DIR = ROOT / "pdfs"
SUM_DIR = DATA / "summaries"
NOVELTY_DIR = DATA / "novelty"
NOVELTY_REPORT = NOVELTY_DIR / "novelty_report.json"
PLANS_DIR = DATA / "plans"
PLAN_FILE = DATA / "plan.json"


@dataclass(frozen=True)
class Requirement:
    check: Callable[[], bool]
    message: str


def _any_json(folder: Path) -> bool:
    try:
        return folder.exists() and any(folder.glob("*.json"))
    except Exception:
        return False


def _has_pdfs() -> bool:
    try:
        return PDF_DIR.exists() and any(PDF_DIR.glob("*.pdf"))
    except Exception:
        return False


def _has_summaries() -> bool:
    return _any_json(SUM_DIR)


def _has_novelty_report() -> bool:
    return NOVELTY_REPORT.exists()


def _has_plans() -> bool:
    if PLAN_FILE.exists():
        return True
    return PLANS_DIR.exists() and any(PLANS_DIR.glob("plan_*.json"))


_STAGE_REQUIREMENTS: Dict[str, Tuple[Requirement, ...]] = {
    "summaries": (
        Requirement(_has_pdfs, f"Summaries skipped: no PDFs found under {PDF_DIR}/."),
    ),
    "novelty": (
        Requirement(_has_summaries, f"Skipping novelty: no summaries found in {SUM_DIR}/."),
    ),
    "idea_blueprints": (
        Requirement(_has_novelty_report, f"Skipping idea_blueprints: {NOVELTY_REPORT} not found."),
    ),
    "planner": (
        Requirement(_has_novelty_report, f"Skipping planner: {NOVELTY_REPORT} not found."),
    ),
    "iterate": (
        Requirement(_has_plans, f"Skipping iterate: no plan artifacts found in {PLAN_FILE} or {PLANS_DIR}/."),
    ),
    "plan_eval": (
        Requirement(_has_plans, f"Plan evaluation skipped: no plan artifacts found in {PLAN_FILE} or {PLANS_DIR}/."),
    ),
}


def stage_ready(stage: str) -> Tuple[bool, str]:
    """Return (ready?, message). Message is non-empty only when not ready."""
    for req in _STAGE_REQUIREMENTS.get(stage, ()):  # type: ignore[arg-type]
        if not req.check():
            return False, req.message
    return True, ""


__all__ = [
    "ROOT",
    "DATA",
    "RUNS",
    "PDF_DIR",
    "SUM_DIR",
    "NOVELTY_DIR",
    "NOVELTY_REPORT",
    "PLANS_DIR",
    "PLAN_FILE",
    "stage_ready",
]
