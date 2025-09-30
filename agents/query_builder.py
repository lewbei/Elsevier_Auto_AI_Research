"""Helpers for generating search keywords and queries."""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from utils.llm_utils import chat_json, LLMError

_STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "with",
    "using",
    "use",
    "via",
    "from",
    "by",
    "into",
    "at",
    "as",
    "is",
    "are",
    "be",
    "being",
    "been",
    "this",
    "that",
    "these",
    "those",
    "we",
    "i",
}


def _generate_llm_keywords(
    goal: str,
    *,
    count: int,
    max_len: int,
    model: Optional[str],
    profile: Optional[str],
) -> List[str]:
    if not goal or count <= 0:
        return []
    payload = {
        "goal": goal,
        "count": max(1, count),
        "max_len": max(8, max_len),
        "format": "json",
    }
    system = (
        "You generate concise search keywords for academic literature retrieval.\n"
        "Return strictly JSON of the form {\"keywords\": [\"term\", ...]} with unique items.\n"
        "Each keyword must stay under max_len characters, avoid stopwords alone, and remain specific to the goal."
    )
    try:
        js = chat_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.0, model=model, profile=profile)
    except LLMError:
        return []
    out: List[str] = []
    for item in (js.get("keywords") or []):
        term = str(item).strip()
        if not term:
            continue
        out.append(term[:max_len])
        if len(out) >= count:
            break
    return out


def _domain_keywords_from_goal(goal: str) -> List[str]:
    g = (goal or "").lower()
    hints: List[str] = []
    if any(x in g for x in ["human pose", "pose classification", "skeleton"]):
        hints += [
            "human pose classification",
            "skeleton-based",
            "pose estimation",
            "keypoint detection",
            "few-shot learning",
            "meta-learning",
            "graph neural network",
            "transformer",
            "action recognition",
        ]
    return hints


def _fallback_keywords(goal: str) -> List[str]:
    tokens = [t for t in re.split(r"[^a-z0-9]+", (goal or "").lower()) if t and t not in _STOPWORDS]
    if not tokens:
        return []
    phrases: List[str] = []
    # whole phrase
    phrases.append(" ".join(tokens[:4]))
    # bigrams
    for i in range(len(tokens) - 1):
        phrases.append(" ".join(tokens[i : i + 2]))
    # trigrams
    for i in range(len(tokens) - 2):
        phrases.append(" ".join(tokens[i : i + 3]))
    # unique preserve order
    seen = set()
    uniq: List[str] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        uniq.append(phrase)
    return uniq[:8]


def _normalize_keywords(
    goal: str,
    include_terms: List[str],
    synonyms: List[str],
    cfg_keywords: List[str],
    llm_keywords: List[str],
) -> List[str]:
    seen = set()
    out: List[str] = []

    def add(term: str) -> None:
        term = str(term).strip()
        if not term or term.lower() in _STOPWORDS:
            return
        if term in seen:
            return
        seen.add(term)
        out.append(term)

    for bucket in (cfg_keywords, llm_keywords, _domain_keywords_from_goal(goal), include_terms, synonyms):
        for entry in bucket or []:
            add(entry)
    return out


def build_keywords(
    goal: str,
    include_terms: List[str],
    synonyms: List[str],
    cfg_keywords: List[str],
    *,
    llm_count: int,
    llm_max_len: int,
    llm_model: Optional[str],
    llm_profile: Optional[str],
    allow_fallback: bool = True,
    require_llm: bool = False,
) -> Tuple[List[str], List[str]]:
    llm_keywords: List[str] = []
    if llm_count > 0:
        llm_keywords = _generate_llm_keywords(
            goal,
            count=llm_count,
            max_len=llm_max_len,
            model=llm_model,
            profile=llm_profile,
        )
    if require_llm and not llm_keywords:
        return [], []
    fallback_keywords: List[str] = _fallback_keywords(goal) if allow_fallback else []
    primary = llm_keywords or fallback_keywords
    keywords = _normalize_keywords(goal, include_terms, synonyms, cfg_keywords, primary)
    if not keywords:
        seed = primary or cfg_keywords or include_terms or synonyms
        if seed:
            keywords = [str(seed[0]).strip() or "deep learning"]
        elif allow_fallback:
            keywords = fallback_keywords or [goal.strip() or "deep learning"]
        else:
            keywords = [goal.strip() or "deep learning"]
    return keywords, llm_keywords


__all__ = ["build_keywords"]
