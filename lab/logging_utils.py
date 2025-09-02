import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict
import datetime


def ensure_dir(p: Path) -> None:
    """Create directory and parent directories if they don't exist."""
    p.mkdir(parents=True, exist_ok=True)


def capture_env() -> Dict[str, Any]:
    """Capture current environment info including Python version, platform, and installed packages."""
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    # Package list (best-effort, avoids deprecated pkg_resources)
    try:
        try:
            from importlib.metadata import distributions  # Python 3.8+
        except Exception:
            distributions = None  # type: ignore

        pkgs = []
        if distributions is not None:
            for dist in distributions():
                name = ""
                version = ""
                try:
                    meta = dist.metadata  # email.message.Message-like
                    name = (meta.get("Name") or "").strip()
                    version = (meta.get("Version") or "").strip()
                except Exception:
                    pass
                if not name:
                    # Fallback best-effort
                    try:
                        name = getattr(dist, "_normalized_name", "") or getattr(dist, "name", "")
                    except Exception:
                        name = ""
                if name:
                    pkgs.append(f"{name}=={version}" if version else name)
        info["packages"] = sorted(pkgs)
    except Exception:
        info["packages"] = []
    return info


def write_json(path: Path, obj: Any) -> None:
    """Write object as JSON to file, creating parent directories as needed."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    """Append object as JSON line to file, creating parent directories as needed."""
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


# ---- Verbose/Debug console helpers ----
def _log_level_env() -> str:
    lvl = os.getenv("LOG_LEVEL", "") or os.getenv("LOGLEVEL", "") or os.getenv("LEVEL", "")
    lvl = (lvl or "").strip().lower()
    if lvl in {"trace", "debug", "info", "warn", "warning", "error"}:
        return lvl
    # Back-compat: VERBOSE=1 implies debug
    if str(os.getenv("VERBOSE", "")).lower() in {"1", "true", "yes", "on"}:
        return "debug"
    return "info"


def get_log_level() -> str:
    """Return effective log level string: debug|info|warn|error."""
    lvl = _log_level_env()
    if lvl in {"trace", "debug"}:
        return "debug"
    if lvl in {"warn", "warning"}:
        return "warn"
    if lvl in {"error"}:
        return "error"
    return "info"


def is_verbose() -> bool:
    """True when debug-level logging is enabled via env/config."""
    return get_log_level() == "debug"


def vprint(msg: str) -> None:
    """Print a line only in verbose (debug) mode with timestamp."""
    if is_verbose():
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[DBG {ts}] {msg}")


def json_dumps_safe(obj: Any, default=str, indent: int = 2) -> str:
    """Best-effort JSON dump that stringifies unsupported objects.
    Intended only for logging and diagnostics.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, default=default)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False, indent=indent)
        except Exception:
            return "{}"


def try_mlflow_log(run_name: str, params: Dict[str, Any], metrics: Dict[str, Any], tags: Dict[str, Any] | None = None) -> None:
    """Best-effort MLflow logging. No-op if mlflow not installed or env disabled.
    Enable by setting MLFLOW_ENABLED=true and optionally MLFLOW_TRACKING_URI.
    """
    if str(os.getenv("MLFLOW_ENABLED", "")).lower() not in {"1", "true", "yes"}:
        return
    try:
        import mlflow  # type: ignore
    except Exception:
        return
    try:
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=run_name):
            # Flatten a few standard params to avoid huge payloads
            flat_params: Dict[str, Any] = {}
            for k in ["model", "input_size", "batch_size", "epochs", "lr", "max_train_steps", "seed"]:
                v = params.get(k)
                if v is not None:
                    flat_params[k] = v
            nc = params.get("novelty_component") or {}
            flat_params["novelty_enabled"] = bool(nc.get("enabled", False))
            mlflow.log_params(flat_params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            if tags:
                mlflow.set_tags(tags)
    except Exception:
        # Swallow all MLflow errors to keep pipeline resilient
        return
