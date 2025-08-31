"""Utilities for generating small augmentation or model-head modules.

The helpers here write tiny Python files based on text descriptions so that
experiments can dynamically compose augmentations or heads without shipping a
full CLI or config parser.  Generated files are lightweight and intentionally
limited to a safe subset of torchvision transforms.
"""

from pathlib import Path
from typing import Optional
import importlib.util
import sys


TEMPLATE = """
try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None  # type: ignore


class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            self.pipe = T.Compose([
                {{TRANSFORMS}}
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)
"""


def _map_description_to_transforms(description: str) -> str:
    """Convert a free-form description into a torchvision transforms snippet."""
    desc = (description or "").lower()
    parts = []
    # Very small, safe set
    if "jitter" in desc:
        parts.append("T.ColorJitter(0.1,0.1,0.1,0.05)")
    if "rotate" in desc:
        parts.append("T.RandomRotation(10)")
    if "erase" in desc:
        parts.append("T.RandomErasing(p=0.25)")
    if "blur" in desc:
        parts.append("T.GaussianBlur(3)")
    if not parts:
        parts.append("T.RandomHorizontalFlip()")
    return ",\n                ".join(parts)


def write_generated_aug(description: str, out: Optional[Path] = None) -> Path:
    """Write a small augmentation module based on a text description."""
    out = out or Path(__file__).parent / "generated_aug.py"
    code = TEMPLATE.replace("{{TRANSFORMS}}", _map_description_to_transforms(description))
    out.write_text(code, encoding="utf-8")
    return out


# ---- Simple generated head (dropout + linear) ----
HEAD_TEMPLATE = """
import torch.nn as nn


class GeneratedHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)
"""


def write_generated_head(out: Optional[Path] = None) -> Path:
    """Write the `GeneratedHead` module to disk and return the path."""
    out = out or Path(__file__).parent / "generated_head.py"
    out.write_text(HEAD_TEMPLATE, encoding="utf-8")
    return out


# ---- Guardrails / sanity checks for generated modules ----
def _import_from_path(module_path: Path, module_name: str):
    """Import a module given its file path without modifying sys.path permanently."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def sanity_check_generated_aug() -> bool:
    """Quick guard: compile the file and best-effort import/call.
    Returns True if the file compiles; import/call are optional.
    """
    import py_compile
    p = Path(__file__).parent / "generated_aug.py"
    if not p.exists():
        return False
    try:
        py_compile.compile(str(p), doraise=True)
        # Best-effort import and tiny call; ignore failures
        try:
            mod = _import_from_path(p, "_generated_aug_runtime")
            GA = getattr(mod, "GeneratedAug", None)
            if GA is not None:
                _ = GA()(object())
        except Exception:
            pass
        return True
    except Exception:
        return False


def sanity_check_generated_head() -> bool:
    """Try to import GeneratedHead. Returns True if importable and constructible.
    Safe for environments without torch: returns False when torch is missing.
    """
    p = Path(__file__).parent / "generated_head.py"
    if not p.exists():
        return False
    try:
        # Delay torch import to inside the function, catch ImportError
        mod = _import_from_path(p, "_generated_head_runtime")
        GH = getattr(mod, "GeneratedHead", None)
        if GH is None:
            return False
        # Do not instantiate if torch is missing; try minimal construct
        try:
            inst = GH(8, 2, 0.2)  # type: ignore[call-arg]
        except Exception:
            return False
        return inst is not None
    except Exception:
        return False
