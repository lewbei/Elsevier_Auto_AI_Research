import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from lab.config import get_bool, get

try:
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore


load_dotenv()

HERE = Path(__file__).parent.parent.resolve()
DATA = HERE / "data"
RUNS = HERE / "runs"


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _persona_phase_notes(phase: str, context: Dict[str, Any], steps: int = 2) -> List[str]:
    enable = get_bool("pipeline.orchestrator.personas.enable", False) or (
        str(os.getenv("ORCH_PERSONAS", "")).lower() in {"1", "true", "yes"}
    )
    if not enable or DialogueManager is None:
        return []
    try:
        dm = DialogueManager()
        dm.post(
            "User",
            (
                f"Phase: {phase}. Provide structured, actionable guidance tailored to limited compute (<=1 epoch, small steps).\n"
                "Include: top 3 priorities, key risks with mitigations, and the minimal concrete actions to advance this phase. Be specific."
            ),
        )
        dm.post("User", f"Context: {json.dumps(context, ensure_ascii=False)}")
        notes: List[str] = []
        for i in range(max(1, steps)):
            # Auto step cycles personas in DialogueManager
            r = dm.step_auto()
            notes.append(f"[{r.get('role','')}] {r.get('text','')}")
        return notes
    except Exception:
        return []


def _run_mod(mod: str) -> None:
    cmd = [sys.executable, "-m", mod]
    print(f"[ORCH] RUN {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(HERE), check=True)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    session = DATA / "orchestrator_session.jsonl"
    notes_dir = DATA / "phase_notes"
    steps = int(get("pipeline.orchestrator.phase_steps", 2) or 2)

    def log(role: str, content: Any) -> None:
        line = json.dumps({"ts": _now(), "role": role, "content": content}, ensure_ascii=False)
        with session.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # Phase: literature review / paper finding
    if not (get_bool("pipeline.skip.find_papers", False) or (str(os.getenv("SKIP_FIND_PAPERS", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("find_papers", {}, steps)
        if notes:
            log("find_papers_notes", notes)
            _write_text(notes_dir / "find_papers.txt", "\n\n".join(notes))
        _run_mod("agents.paper_finder")
    else:
        print("[ORCH] Skipping paper_finder per config/env")

    # Phase: summaries (per-paper)
    if not (get_bool("pipeline.skip.summaries", False) or (str(os.getenv("SKIP_SUMMARIES", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("summaries", {}, steps)
        if notes:
            log("summaries_notes", notes)
            _write_text(notes_dir / "summaries.txt", "\n\n".join(notes))
        _run_mod("agents.summarize")
    else:
        print("[ORCH] Skipping summaries per config/env")

    # Phase: novelty
    if not (get_bool("pipeline.skip.novelty", False) or (str(os.getenv("SKIP_NOVELTY", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("novelty", {}, steps)
        if notes:
            log("novelty_notes", notes)
            _write_text(notes_dir / "novelty.txt", "\n\n".join(notes))
        _run_mod("agents.novelty")
    else:
        print("[ORCH] Skipping novelty per config/env")

    # Phase: plan
    if not (get_bool("pipeline.skip.planner", False) or (str(os.getenv("SKIP_PLANNER", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("plan", {}, steps)
        if notes:
            log("plan_notes", notes)
            _write_text(notes_dir / "plan.txt", "\n\n".join(notes))
        _run_mod("agents.planner")
    else:
        print("[ORCH] Skipping planner per config/env")

    # Phase: data prep (advice only; iterate consumes persona notes when enabled)
    notes = _persona_phase_notes("data_prep", {}, steps)
    if notes:
        log("data_prep_notes", notes)
        _write_text(notes_dir / "data_prep.txt", "\n\n".join(notes))

    # Phase: interactive code loop (optional)
    enable_interactive = get_bool("pipeline.interactive.enable", False) or (str(os.getenv("INTERACTIVE_ENABLE", "")).lower() in {"1", "true", "yes"})
    if enable_interactive and not (get_bool("pipeline.skip.interactive", False) or (str(os.getenv("SKIP_INTERACTIVE", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("interactive", {}, steps)
        if notes:
            log("interactive_notes", notes)
            _write_text(notes_dir / "interactive.txt", "\n\n".join(notes))
        _run_mod("agents.interactive")

    # Phase: run (iterate)
    if not (get_bool("pipeline.skip.iterate", False) or (str(os.getenv("SKIP_ITERATE", "")).lower() in {"1", "true", "yes"})):
        notes = _persona_phase_notes("run_iterate", {}, steps)
        if notes:
            log("iterate_notes", notes)
            _write_text(notes_dir / "iterate.txt", "\n\n".join(notes))
        _run_mod("agents.iterate")
    else:
        print("[ORCH] Skipping iterate per config/env")

    # Phase: interpretation
    try:
        summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    notes = _persona_phase_notes("interpretation", {"summary": summary}, steps)
    if notes:
        log("interpret_notes", notes)
        _write_text(RUNS / "interpretation.txt", "\n\n".join(notes))

    # Phase: report (optional)
    write_paper = get_bool("pipeline.write_paper", False) or (str(os.getenv("WRITE_PAPER", "")).lower() in {"1", "true", "yes"})
    if write_paper:
        notes = _persona_phase_notes("report", {}, steps)
        if notes:
            log("report_notes", notes)
            _write_text(notes_dir / "report.txt", "\n\n".join(notes))
        _run_mod("agents.write_paper")

    print("[ORCH] Done")


if __name__ == "__main__":
    main()
