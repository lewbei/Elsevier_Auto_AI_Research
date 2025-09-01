"""Constrained code editing loop to synthesize tiny training hooks.

This mirrors the example repo's REPLACE/EDIT pattern in a trimmed, safe form:
- Generates a small module at lab/generated_train.py providing optional hooks:
  * build_train_transforms(input_size: int) -> optional torchvision Compose
  * build_model_head(in_features: int, num_classes: int) -> optional nn.Module
  * update_spec(spec: dict) -> dict (may tweak lr, max_train_steps, input_size, novelty_component.enabled)

Safety:
- Forbid file/network/system access; only allow torchvision transforms, torch.nn.
- Validate with py_compile and light import checks before writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import re
import py_compile
import importlib.util
import sys

from utils.llm_utils import chat_text_cached, LLMError


GEN_PATH = Path(__file__).parent / "generated_train.py"


SKELETON = """
try:
    import torchvision.transforms as T
except Exception:
    T = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


def build_train_transforms(input_size: int):
    # Return a torchvision transforms.Compose or None.
    if T is None:
        return None
    # EDIT BELOW: keep short and deterministic, no randomness if possible
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])


def build_model_head(in_features: int, num_classes: int):
    # Return an nn.Module head or None to use default.
    if nn is None:
        return None
    # EDIT BELOW (optional)
    return None


def update_spec(spec: dict) -> dict:
    # Optionally tweak safe keys in spec (lr, max_train_steps, input_size, novelty_component.enabled).
    s = dict(spec)
    try:
        # keep within reasonable bounds
        lr = float(s.get("lr", 1e-3))
        if lr <= 0:
            lr = 1e-3
        s["lr"] = float(min(max(lr, 1e-5), 1e-1))
        mts = int(s.get("max_train_steps", 50))
        s["max_train_steps"] = int(min(max(mts, 10), 1000))
        inp = int(s.get("input_size", 224))
        s["input_size"] = int(min(max(inp, 96), 512))
        nc = s.get("novelty_component") or {"enabled": False, "description": "baseline"}
        if not isinstance(nc, dict):
            nc = {"enabled": False, "description": str(nc)}
        s["novelty_component"] = nc
        return s
    except Exception:
        return spec
""".lstrip()


FORBIDDEN_TOKENS = [
    "import os", "import sys", "open(", "requests.", "urllib.", "subprocess",
    "shutil", "Path(", "torch.hub", "socket", "http", "mlflow",
]


def _extract_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract (kind, content) pairs where kind in {python, REPLACE, EDIT}.

    - ```python ... ``` -> ("python", code)
    - ```REPLACE ... ``` -> ("REPLACE", code)
    - ```EDIT N M ... ``` -> ("EDIT:N:M", lines)
    """
    out: List[Tuple[str, str]] = []
    if not text:
        return out
    # REPLACE and EDIT fenced forms
    for m in re.finditer(r"```(REPLACE|EDIT)([^\n]*)\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        kind = m.group(1).upper()
        header = m.group(2).strip()
        body = m.group(3).rstrip()
        if kind == "REPLACE":
            out.append(("REPLACE", body))
        elif kind == "EDIT":
            # header should contain N M
            nums = re.findall(r"\d+", header)
            if len(nums) >= 2:
                out.append((f"EDIT:{nums[0]}:{nums[1]}", body))
    # Generic python fenced code
    for m in re.finditer(r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        out.append(("python", m.group(1).rstrip()))
    # Fallback: any fenced block
    for m in re.finditer(r"```\s*\n(.*?)```", text, flags=re.DOTALL):
        out.append(("python", m.group(1).rstrip()))
    return out


def _is_safe(code: str) -> bool:
    if any(tok in code for tok in FORBIDDEN_TOKENS):
        return False
    if code.count("import ") > 12:
        return False
    # Must define expected symbols
    needed = ["def build_train_transforms", "def build_model_head", "def update_spec"]
    return all(k in code for k in needed)


def _compile_ok(path: Path) -> bool:
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except Exception:
        return False


def _import_ok(path: Path) -> bool:
    try:
        spec = importlib.util.spec_from_file_location("_generated_train_runtime", str(path))
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_generated_train_runtime"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        # Check symbols exist
        return all(hasattr(mod, fn) for fn in ["build_train_transforms", "build_model_head", "update_spec"])
    except Exception:
        return False


def _apply_edit(current: List[str], n: int, m: int, new_body: str) -> List[str]:
    lines = list(current)
    n = max(0, min(n, len(lines)))
    m = max(0, min(m, len(lines) - 1))
    if n > m:
        n, m = m, n
    new_lines = new_body.splitlines()
    return lines[:n] + new_lines + lines[m + 1 :]


def run_codegen_editor(description: str, extra_context: Optional[str] = None, max_steps: int = 3) -> bool:
    """Iteratively generate lab/generated_train.py via REPLACE/EDIT commands.

    Returns True on success; False on failure.
    """
    # Start from the skeleton
    current = SKELETON.splitlines()

    sys_prompt = (
        "You are a careful ML engineer performing constrained code edits for a tiny training hooks file.\n"
        "You may only return one fenced block per turn: either ```REPLACE ...``` (entire file) or ```EDIT N M ...``` (line range replacement).\n"
        "Allowed building blocks: torchvision transforms (ColorJitter, RandomRotation, RandomErasing, GaussianBlur, RandomHorizontalFlip, Resize, ToTensor) and torch.nn layers.\n"
        "Absolutely no file/network/system access, subprocess, sockets, or unsafe imports. Keep diffs minimal and deterministic."
    )
    user_base = (
        "Target file defines exactly three functions: build_train_transforms(input_size), build_model_head(in_features, num_classes), update_spec(spec).\n"
        f"Novelty description: {description}\n"
        + (f"Context: {extra_context}\n" if extra_context else "") +
        "Instructions: start with a full-file ```REPLACE``` that minimally modifies the skeleton to reflect the novelty (<=4 transforms, small heads).\n"
        "Use only allowed transforms/layers; keep safe ranges (lr 1e-5..1e-1, steps 10..1000, input 96..512); preserve function signatures and comments."
    )

    err_msg = ""
    for step in range(1, max_steps + 1):
        prompt = user_base
        if err_msg:
            prompt += f"\nPrevious error: {err_msg}\nPlease fix using EDIT or REPLACE."
        try:
            text = chat_text_cached([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ], temperature=0.2)
        except LLMError as exc:
            # Stop early; caller may fallback to deterministic behavior
            return False
        blocks = _extract_blocks(text)
        if not blocks:
            err_msg = "No command block detected."
            continue
        # Take the first actionable block
        kind, body = blocks[0]
        if kind == "REPLACE" or kind == "python":
            code = body if kind == "REPLACE" else body
            # safety
            if not _is_safe(code):
                err_msg = "Code failed safety constraints."
                continue
            try:
                GEN_PATH.write_text(code, encoding="utf-8")
            except Exception as exc:
                err_msg = f"Write error: {exc}"
                continue
        elif kind.startswith("EDIT:"):
            parts = kind.split(":")
            try:
                n = int(parts[1])
                m = int(parts[2])
            except Exception:
                err_msg = "Bad EDIT header; need EDIT N M."
                continue
            current = _apply_edit(current, n, m, body)
            code = "\n".join(current)
            if not _is_safe(code):
                err_msg = "Edited code failed safety constraints."
                continue
            try:
                GEN_PATH.write_text(code, encoding="utf-8")
            except Exception as exc:
                err_msg = f"Write error: {exc}"
                continue
        else:
            err_msg = "Unsupported block type."
            continue

        # Validate compile/import
        if not _compile_ok(GEN_PATH):
            err_msg = "py_compile failed."
            continue
        if not _import_ok(GEN_PATH):
            err_msg = "import check failed."
            continue
        return True
    return False
