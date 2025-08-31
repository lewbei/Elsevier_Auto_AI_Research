from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any


def maybe_save_accuracy_bar(path: Path, runs: List[Dict[str, Any]]) -> bool:
    """Try to save a simple accuracy bar chart. Returns True if saved.
    Gracefully returns False if matplotlib is not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    names = []
    accs = []
    for r in runs:
        try:
            names.append(str(r.get("name")))
            accs.append(float(r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0) or 0.0))
        except Exception:
            continue
    if not names:
        return False
    try:
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.4), 3))
        ax.bar(range(len(names)), accs)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Val Acc")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path))
        plt.close(fig)
        return True
    except Exception:
        return False

