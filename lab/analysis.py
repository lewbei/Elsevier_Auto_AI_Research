from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import os

from .config import get_bool, get
from utils.llm_utils import chat_text_cached, LLMError




def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def summarize_runs_simple(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic summary over runs without LLM.

    Computes top-1, per-group bests, and trends for novelty vs baseline.
    """
    best = {"name": None, "acc": 0.0}
    sums = {"baseline": 0.0, "novelty": 0.0}
    counts = {"baseline": 0, "novelty": 0}
    for r in runs:
        try:
            name = str(r.get("name") or "")
            acc = _safe_float(r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0))
            if acc > best["acc"]:
                best = {"name": name, "acc": acc}
            for k in ("baseline", "novelty"):
                if name.startswith(k):
                    sums[k] += acc
                    counts[k] += 1
        except Exception:
            continue
    means = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in sums}
    delta = means.get("novelty", 0.0) - means.get("baseline", 0.0)
    return {
        "best": best,
        "mean_baseline": means.get("baseline", 0.0),
        "mean_novelty": means.get("novelty", 0.0),
        "delta_mean": delta,
        "n_runs": len(runs),
    }


def summarize_runs_llm(
    runs: List[Dict[str, Any]],
    decision: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
    novelty: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an LLM-written analysis paragraph grounded on provided data only."""
    enable = get_bool("pipeline.analysis.llm.enable", False) or (str(os.getenv("ANALYSIS_LLM", "")).lower() in {"1", "true", "yes"})
    if not enable:
        return ""
    compact: List[Dict[str, Any]] = []
    for r in runs:
        compact.append({
            "name": r.get("name"),
            "metrics": r.get("result", {}).get("metrics", {}),
        })
    payload = {
        "runs": compact,
        "decision": decision,
        "plan": {k: (plan or {}).get(k) for k in ["objective", "hypotheses", "success_criteria"]},
        "novelty": (novelty or {}).get("novelty_component", {}),
    }
    system = (
        "You are a careful analyst. Write a short findings section grounded ONLY in the provided runs/metrics and plan.\n"
        "State what improved or not, call out deltas, anomalies, and suggested next steps under the same budget. Return plain text (no code)."
    )
    try:
        text = chat_text_cached([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], temperature=0.2)
    except LLMError:
        return ""
    return text.strip()


def write_analysis_files(
    out_dir: Path,
    runs: List[Dict[str, Any]],
    decision: Dict[str, Any],
    plan_path: Path,
    novelty_spec: Dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # deterministic summary
    simple = summarize_runs_simple(runs)
    (out_dir / "analysis.json").write_text(json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8")
    # optional LLM summary
    plan = {}
    try:
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        plan = {}
    llm_txt = summarize_runs_llm(runs, decision, plan=plan, novelty=novelty_spec or {})
    if llm_txt:
        (out_dir / "analysis.md").write_text(llm_txt + "\n", encoding="utf-8")
