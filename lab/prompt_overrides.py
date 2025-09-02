from __future__ import annotations
from pathlib import Path
from typing import Optional
import os


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "on"}


def load_prompt(name: str, base_env: str = "PROMPTS_DIR", default_dir: str = "prompts") -> Optional[str]:
    """Load a prompt override by name.
    Resolution order (first hit wins):
    - If name contains a path separator, treat as relative to PROMPTS_DIR.
    - Hierarchical: <base>/<stem>/<leaf>.md for names like "planner_pi_system" -> planner/pi_system.md
    - Flat: <base>/<name>.md
    Returns stripped string or None if not found.
    """
    base = Path(os.getenv(base_env, default_dir)).resolve()
    # Reject absolute paths outright for safety
    try:
        if Path(name).is_absolute():
            return None
    except Exception:
        pass
    if os.sep in name or "/" in name:
        # Treat as relative to base; ensure final path stays within base
        p = (base / name)
        if p.suffix != ".md":
            p = p.with_suffix(".md")
        try:
            rp = p.resolve()
            if not str(rp).startswith(str(base)):
                return None
            return rp.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    stem = name.replace(" ", "_").lower()
    if "_" in stem:
        parts = stem.split("_", 1)
        cand = (base / parts[0] / f"{parts[1]}.md").resolve()
        try:
            if str(cand).startswith(str(base)) and cand.exists():
                return cand.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    flat = (base / f"{stem}.md").resolve()
    try:
        if str(flat).startswith(str(base)) and flat.exists():
            return flat.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return None

