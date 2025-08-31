import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def capture_env() -> Dict[str, Any]:
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
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


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
