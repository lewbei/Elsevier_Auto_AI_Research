"""Synthesize a literature review from existing per-paper summaries.

Reads JSON records under data/summaries/*.json (produced by agents.summarize)
and composes a concise literature review in Markdown. If the LLM is not
available, falls back to a deterministic, frequency-based outline.

Outputs:
- data/lit_review.md (always)
- Prints the review to stdout
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List
import os

from dotenv import load_dotenv
from lab.config import get
from utils.llm_utils import chat_text_cached, LLMError


load_dotenv()

DATA_DIR = pathlib.Path("data")
SUM_DIR = DATA_DIR / "summaries"
OUT_PATH = DATA_DIR / "lit_review.md"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_summaries() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not SUM_DIR.exists():
        return records
    files = sorted([p for p in SUM_DIR.glob("*.json") if p.is_file()])
    for p in files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and isinstance(rec.get("summary"), dict):
                records.append(rec)
        except Exception:
            continue
    return records


def _compress(rec: Dict[str, Any]) -> Dict[str, Any]:
    s = rec.get("summary", {}) if isinstance(rec.get("summary"), dict) else {}
    def _list(k: str, n: int) -> List[str]:
        v = s.get(k)
        if not isinstance(v, list):
            return []
        return [str(x)[:200] for x in v[:n]]
    def _str(k: str, n: int) -> str:
        v = str(s.get(k) or "")
        return v[:n]
    return {
        "title": _str("title", 160),
        "problem": _str("problem", 240),
        "methods": _list("methods", 6),
        "datasets": _list("datasets", 6),
        "results": _list("results", 6),
        "novelty_claims": _list("novelty_claims", 8),
        "limitations": _list("limitations", 6),
        "keywords": _list("keywords", 8),
        "year": (s.get("year") or ""),
        "venue": (s.get("venue") or ""),
        "doi": (s.get("doi") or ""),
    }


def _build_llm_review(items: List[Dict[str, Any]]) -> str:
    goal = str(get("project.goal", "your goal") or "your goal")
    system = (
        "You are an expert researcher synthesizing a literature review from compressed per-paper summaries.\n"
        f"Focus on the project goal: {goal}.\n"
        "Write a well-structured Markdown review (no code fences) with the following sections: Overview, Thematic Synthesis, Methods & Datasets (common/rare), Results Trends (with qualitative comparisons), Novelty Landscape (what is genuinely new vs incremental), Limitations, and Gaps & Opportunities (actionable).\n"
        "Guidelines: remain factual and grounded in the summaries; avoid speculation; prefer short paragraphs and bullet points; highlight consensus and disagreements; call out reproducibility or evaluation concerns where relevant."
    )
    # Keep payload bounded
    payload: List[Dict[str, Any]] = items[:60]
    user = json.dumps({"papers": payload}, ensure_ascii=False)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = chat_text_cached(msgs, temperature=0.2)
    return text.strip()


def _build_offline_review(items: List[Dict[str, Any]]) -> str:
    # Simple frequency-based outline
    from collections import Counter
    titles = [it.get("title") or "Untitled" for it in items]
    methods = [m for it in items for m in (it.get("methods") or [])]
    datasets = [d for it in items for d in (it.get("datasets") or [])]
    nov = [n for it in items for n in (it.get("novelty_claims") or [])]
    lim = [l for it in items for l in (it.get("limitations") or [])]
    top = lambda xs, k=8: [f"- {w} ({c})" for w, c in Counter(xs).most_common(k) if w]

    lines: List[str] = []
    lines.append("# Literature Review (Auto)")
    lines.append("")
    lines.append(f"Corpus size: {len(titles)} papers")
    lines.append("")
    lines.append("## Common Methods")
    lines.extend(top(methods))
    lines.append("")
    lines.append("## Common Datasets")
    lines.extend(top(datasets))
    lines.append("")
    lines.append("## Claimed Novelties")
    lines.extend(top(nov))
    lines.append("")
    lines.append("## Reported Limitations")
    lines.extend(top(lim))
    lines.append("")
    lines.append("## Gaps & Opportunities")
    lines.append("- Explore underrepresented combinations of methods and datasets")
    lines.append("- Address frequent limitations via targeted ablations")
    lines.append("- Standardize evaluation to reduce metric variance")
    return "\n".join(lines)


def main() -> None:
    _ensure_dirs()
    records = _read_summaries()
    if not records:
        print(f"[ERR] No summaries found in {SUM_DIR}. Run agents.summarize first.")
        return
    items = [_compress(r) for r in records]
    try:
        review = _build_llm_review(items)
    except LLMError as exc:
        # No offline fallback per project policy
        print(f"[ERR] LLM failed for lit_review ({exc}); skipping lit_review (no offline fallback).")
        return

    OUT_PATH.write_text(review, encoding="utf-8")
    print(review)
    print(f"\n[DONE] Literature review written to {OUT_PATH}")


if __name__ == "__main__":
    main()
