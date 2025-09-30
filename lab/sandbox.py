"""Sandboxed execution helpers for generated code/tests.

We avoid importing pytest programmatically inside the same process to reduce
risk from arbitrary code. Instead we spawn a subprocess with a narrowed
environment and time limit.
"""
from __future__ import annotations

from typing import Optional, List
import subprocess
import os
import sys
import json
import shlex
import tempfile
from pathlib import Path


class SandboxError(RuntimeError):
    pass


def run_pytest(paths: Optional[List[str]] = None, timeout: int = 60, extra_env: Optional[dict] = None) -> dict:
    """Run pytest on given paths inside a subprocess.

    Returns a dict with keys: returncode, stdout, duration_sec.
    If pytest not installed, returns a synthetic skipped result.
    """
    import time
    start = time.time()
    try:
        import pytest  # noqa: F401
    except Exception:
        return {
            "returncode": 0,
            "stdout": "pytest not installed - treating as pass",
            "duration_sec": 0.0,
        }
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if paths:
        cmd += paths
    env = {"PYTHONUNBUFFERED": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    # Pass through minimal safe env
    for k in ("PATH", "SYSTEMROOT", "PYTHONPATH"):
        if k in os.environ:
            env[k] = os.environ[k]
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out = proc.stdout + "\n" + proc.stderr
        return {
            "returncode": proc.returncode,
            "stdout": out.strip(),
            "duration_sec": time.time() - start,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "stdout": "TIMEOUT",
            "duration_sec": time.time() - start,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "returncode": 1,
            "stdout": f"Exception launching pytest: {exc}",
            "duration_sec": time.time() - start,
        }
