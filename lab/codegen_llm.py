"""LLM-driven code generation helpers.

This module provides safe, minimal functionality to synthesize tiny Python
modules during iteration, inspired by the example AgentLaboratory behavior.

Design goals:
- Keep scope narrow: only generate a torchvision augmentation shim that the
  pipeline can optionally use via lab.experiment_runner.
- Be robust on machines without keys by failing silently and falling back to
  deterministic code in lab.codegen_utils.
- Avoid risky operations: no I/O, no network calls inside generated code, and
  only a small whitelist of transforms.
"""

from __future__ import annotations

from typing import Optional, List
from pathlib import Path
import re

from utils.llm_utils import chat_text_cached, LLMError
from lab.config import get


_GEN_PATH = Path(__file__).parent / "generated_aug.py"


def _extract_code_blocks(text: str) -> List[str]:
    """Return python code blocks from markdown-style fences.

    Prefers ```python fenced blocks, falls back to generic ``` blocks if needed.
    """
    if not text:
        return []
    blocks: List[str] = []
    # Prefer ```python ... ```
    for m in re.finditer(r"```python\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        code = m.group(1).strip()
        if code:
            blocks.append(code)
    if blocks:
        return blocks
    # Fallback: any fenced block
    for m in re.finditer(r"```\s*\n(.*?)```", text, flags=re.DOTALL):
        code = m.group(1).strip()
        if code:
            blocks.append(code)
    return blocks


def _is_safe_aug_code(code: str) -> bool:
    """Very lightweight safety/shape checks for the augmentation module."""
    if "class GeneratedAug" not in code:
        return False
    if "import torchvision.transforms as T" not in code:
        return False
    # Ensure no obvious I/O or network
    forbidden = [
        "import os", "import sys", "open(", "requests.", "urllib.", "subprocess",
        "torch.hub", "PIL.Image.open", "Path("
    ]
    return not any(tok in code for tok in forbidden)


def write_generated_aug_from_llm(description: str, extra_context: Optional[str] = None) -> Optional[Path]:
    """Ask the LLM to generate a tiny augmentation module and write it to disk.

    Returns the written path on success; None if generation failed. On failure,
    callers should fall back to lab.codegen_utils.write_generated_aug.
    """
    sys_prompt = (
        "You are a careful ML engineer writing a tiny Python augmentation module.\n"
        "Define class GeneratedAug that wraps torchvision transforms.\n"
        "WHITELIST ONLY: T.ColorJitter, T.RandomRotation, T.RandomErasing, T.GaussianBlur, T.RandomHorizontalFlip, T.Compose, T.Resize.\n"
        "Important: DO NOT include T.ToTensor(); the training pipeline appends it.\n"
        "Constraints: minimal imports; absolutely NO file I/O, NO network, NO training or model code, NO side effects.\n"
        "Output: ONLY a single Python fenced code block with the complete module."
    )
    user_prompt = (
        "Create a module that matches exactly this skeleton and behavior. Keep the transform list short and deterministic where possible.\n\n"
        "try:\n    import torchvision.transforms as T\n"
        "except Exception:\n    T = None\n\n"
        "class GeneratedAug:\n"
        "    def __init__(self):\n"
        "        if T is None:\n            self.pipe = None\n        else:\n            self.pipe = T.Compose([\n                # your transforms here (no ToTensor)\n            ])\n\n"
        "    def __call__(self, x):\n        if self.pipe is None:\n            return x\n        return self.pipe(x)\n\n"
        "Choose transforms based on this description: '" + (description or "") + "'.\n"
        + ("Context: " + extra_context if extra_context else "") + "\n"
        "Only use the allowed transforms (no others). Keep it short (<=4 transforms)."
    )

    try:
        model = get("pipeline.codegen.model", None)
        profile = get("pipeline.codegen.llm", None)
        text = chat_text_cached([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.2, model=model, profile=profile)
    except LLMError:
        return None
    blocks = _extract_code_blocks(text)
    if not blocks:
        return None
    # Prefer the first valid block
    for code in blocks:
        if _is_safe_aug_code(code):
            try:
                _GEN_PATH.write_text(code, encoding="utf-8")
                # Compile check (best-effort)
                try:
                    import py_compile
                    py_compile.compile(str(_GEN_PATH), doraise=True)
                except Exception:
                    _GEN_PATH.unlink(missing_ok=True)
                    return None
                return _GEN_PATH
            except Exception:
                return None
    return None


# ---- LLM-generated model (safe subset) ----
_GEN_MODEL_PATH = Path(__file__).parent / "generated_model.py"


def _is_safe_model_code(code: str) -> bool:
    """Very lightweight safety/shape checks for a generated model module."""
    # Must define a build function and a nn.Module class
    if "class GeneratedModel" not in code:
        return False
    if "def build_model(" not in code:
        return False
    # Ensure only safe imports
    bad = ["requests.", "subprocess", "open(", "os.", "sys.", "socket", "urllib", "Path("]
    if any(tok in code for tok in bad):
        return False
    if "import torch" not in code or "import torch.nn as nn" not in code:
        return False
    return True


def write_generated_model_from_llm(description: str, extra_context: Optional[str] = None) -> Optional[Path]:
    """Ask the LLM to generate a tiny classification/regression model module.
    Returns the written path on success; None if generation failed.
    """
    sys_prompt = (
        "You are a careful ML engineer writing a tiny PyTorch model module.\n"
        "Define class GeneratedModel(nn.Module) using only: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Linear.\n"
        "Also define build_model(task:str, num_classes:int, input_size:int, spec:dict)->nn.Module.\n"
        "Constraints: no file/network/system I/O, no side effects; import only torch and torch.nn. Output a single Python fenced code block."
    )
    user_prompt = (
        "Create a minimal CNN with GAP+Linear head; support task='classification' or 'regression' (set output dim accordingly).\n"
        "Description: '" + (description or "") + "'\n"
        + ("Context: " + extra_context if extra_context else "") + "\n"
        "Return only a single Python fenced code block."
    )
    try:
        model = get("pipeline.codegen.model", None)
        profile = get("pipeline.codegen.llm", None)
        text = chat_text_cached([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.2, model=model, profile=profile)
    except LLMError:
        return None
    blocks = _extract_code_blocks(text)
    if not blocks:
        return None
    for code in blocks:
        if _is_safe_model_code(code):
            try:
                _GEN_MODEL_PATH.write_text(code, encoding="utf-8")
                # Compile check
                try:
                    import py_compile
                    py_compile.compile(str(_GEN_MODEL_PATH), doraise=True)
                except Exception:
                    _GEN_MODEL_PATH.unlink(missing_ok=True)
                    return None
                return _GEN_MODEL_PATH
            except Exception:
                return None
    return None
