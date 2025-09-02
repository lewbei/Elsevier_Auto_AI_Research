from __future__ import annotations

"""Domain detection and lightweight templates.

This module centralizes a minimal notion of research "domains" and exposes
helpers to tailor prompts/spec defaults per domain without changing core
training logic. It stays stub-safe: CV maps cleanly to the existing
experiment runner; other domains provide hints and no-ops that do not break
execution when frameworks are unavailable.

Domains: 'cv' | 'nlp' | 'rl' | 'generic'

Controls (YAML/env):
- config: project.goal (string) is scanned for domain hints
- config: dataset.kind (imagefolder|cifar10|custom) informs CV
- env: DOMAIN can force the domain (cv|nlp|rl|generic)
"""

from typing import Dict, Any, Tuple
import os
import re

from .config import get, dataset_kind


def detect_domain(goal_text: str | None = None) -> str:
    forced = (os.getenv("DOMAIN") or "").strip().lower()
    if forced in {"cv", "nlp", "rl", "generic"}:
        return forced
    txt = " ".join([
        str(goal_text or ""),
        str(get("project.goal", "") or ""),
    ]).lower()
    # dataset.kind implies CV
    kind = (dataset_kind(None) or "").lower()
    if kind in {"imagefolder", "cifar10", "custom"}:
        return "cv"
    # crude keyword heuristics
    if re.search(r"(image|vision|segmentation|detection|isic|cifar|retina|xray)", txt):
        return "cv"
    if re.search(r"(language|nlp|text|token|gpt|llm|bert)", txt):
        return "nlp"
    if re.search(r"(reinforcement|policy|gym|agent|environment)", txt):
        return "rl"
    return "generic"


def template_hints(domain: str) -> Dict[str, Any]:
    """Return conservative defaults/hints for the domain.

    These are used to steer prompts and initial spec suggestions; they do not
    enforce any incompatible behavior with the current CV runner.
    """
    domain = (domain or "generic").lower()
    if domain == "cv":
        return {
            "task": "image_classification",
            "backbones": ["resnet18", "mobilenet_v3_small"],
            "input_size": 224,
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-3,
        }
    if domain == "nlp":
        # Runner does not implement NLP training; keep hints only
        return {
            "task": "language_modeling",
            "backbones": ["gpt-mini", "lstm"],
            "note": "NLP mode is a stub in this version; experiments run via CV stub unless extended.",
        }
    if domain == "rl":
        return {
            "task": "reinforcement_learning",
            "backbones": ["mlp", "cnn"],
            "envs": ["CartPole-v1"],
            "note": "RL mode is a stub in this version; experiments run via CV stub unless extended.",
        }
    return {
        "task": "generic",
        "backbones": ["resnet18"],
    }


def apply_template_defaults(spec: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Fill reasonable defaults consistent with the CV runner and domain hints."""
    out = dict(spec)
    hints = template_hints(domain)
    out.setdefault("input_size", hints.get("input_size", 224))
    out.setdefault("epochs", hints.get("epochs", 1))
    out.setdefault("batch_size", hints.get("batch_size", 16))
    out.setdefault("lr", hints.get("lr", 1e-3))
    # map backbone to model for CV
    backs = hints.get("backbones") or []
    if backs and "model" not in out:
        out["model"] = backs[0]
    return out


def prompt_block(domain: str) -> Tuple[str, Dict[str, Any]]:
    """Return (string, dict) with human-readable domain block and hints dict for LLM prompts."""
    hints = template_hints(domain)
    text = (
        f"Domain: {domain}. Task template: {hints.get('task','generic')}. "
        f"Preferred backbones: {', '.join(hints.get('backbones', [])) or 'resnet18'}. "
        "Keep CPU-friendly budgets (<=1 epoch, small steps)."
    )
    return text, hints

