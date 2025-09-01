from __future__ import annotations
import json
import os
import re
import pathlib
from typing import Dict, Any, List

DATA_DIR = pathlib.Path("data")
SCORES_PATH = DATA_DIR / "idea_scores.jsonl"

def _has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))

def specificity_score(it: Dict[str, Any]) -> int:
    s = 0
    s += 1 if _has_numbers(it.get("spec_hint", "")) else 0
    s += 1 if _has_numbers(it.get("method", "")) else 0
    s += min(2, len(it.get("eval_plan") or []))
    return s  # 0..4+

def grounding_score(it: Dict[str, Any], facets: Dict[str, List[str]]) -> int:
    titles = it.get("derived_from_titles") or []
    s = 0
    s += min(2, len([t for t in titles if str(t).strip()]))
    s += 1 if len((it.get("delta_vs_prior") or "").strip()) >= 40 else 0
    # bonus if method text mentions any allowed backbone explicitly
    txt = " ".join([
        str(it.get("title", "")),
        str(it.get("method", "")),
        str(it.get("spec_hint", "")),
    ]).lower()
    if any(bb in txt for bb in facets.get("backbones", [])):
        s += 1
    return s  # 0..4

def penalty(it: Dict[str, Any], facets: Dict[str, List[str]]) -> int:
    txt = " ".join([str(it.get(k, "")) for k in ["title", "why_novel", "spec_hint", "method"]]).lower()
    p = 0
    ban_env = os.getenv("NOVELTY_BAN", "")
    ban_terms = [b.strip().lower() for b in ban_env.split(",") if b.strip()] or [
        "resnet18",
        "flip",
        "jitter",
        "generic backbone",
    ]
    for bad in ban_terms:
        if bad and bad in txt:
            p += 2
    # penalize if referencing a backbone not in facets
    allowed = set(facets.get("backbones", []))
    if "backbone" in txt and allowed:
        if not any(bb in txt for bb in allowed):
            p += 3
    return p

def score_and_filter_ideas(
    panel: Dict[str, Any],
    facets: Dict[str, List[str]],
    keep_top: int = 8,
    min_score: int = 2,
) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ideas = list(panel.get("new_ideas_detailed") or [])
    ranked = []
    # Initialize scores file if missing
    if not SCORES_PATH.exists():
        SCORES_PATH.write_text("", encoding="utf-8")
    for it in ideas:
        spec = specificity_score(it)
        grnd = grounding_score(it, facets)
        pen = penalty(it, facets)
        total = 2 * spec + 3 * grnd - pen
        ranked.append((total, spec, grnd, pen, it))
        with open(SCORES_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "title": it.get("title", ""),
                        "spec": spec,
                        "grounding": grnd,
                        "penalty": pen,
                        "total": total,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    ranked.sort(key=lambda x: x[0], reverse=True)
    kept = [r[-1] for r in ranked[:keep_top] if r[0] >= min_score]
    panel["new_ideas_detailed"] = kept
    panel["new_ideas"] = [it.get("title", "") for it in kept]
    return panel

