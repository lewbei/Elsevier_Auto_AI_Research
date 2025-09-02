"""Tier‑1 Novelty Validator: rubric → grading → blueprint (LLM‑driven).

Artifacts written under data/:
- tier1_rubric.json
- tier1_grade.json
- paper_blueprint_llm.md

This module is additive and safe: on LLM error, it writes skinny fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

from lab.config import get
from utils.llm_utils import chat_json, chat_text_cached, LLMError


DATA_DIR = pathlib.Path("data")
RUBRIC_PATH = DATA_DIR / "tier1_rubric.json"
GRADE_PATH = DATA_DIR / "tier1_grade.json"
BLUEPRINT_PATH = DATA_DIR / "paper_blueprint_llm.md"
GATE_PATH = DATA_DIR / "tier1_gate.json"
UPGRADE_PLAN_PATH = DATA_DIR / "tier1_upgrade_plan.md"
GATE_PATH = DATA_DIR / "tier1_gate.json"
UPGRADE_PLAN_PATH = DATA_DIR / "tier1_upgrade_plan.md"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _hash_final(final: Dict[str, Any], venue: str, num_dims: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(final, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(venue.encode("utf-8"))
    h.update(str(num_dims).encode("utf-8"))
    return h.hexdigest()[:16]


def _signals_from_final(final: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "claim_like": (final.get("problems") or [])[:2],
        "objectives": (final.get("objectives") or [])[:4],
        "contributions": (final.get("contributions") or [])[:4],
        "research_questions": (final.get("research_questions") or [])[:2],
    }


def _normalize_weights(dims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tots = sum(float(d.get("weight") or 0.0) for d in dims) or 1.0
    out: List[Dict[str, Any]] = []
    for d in dims:
        dd = dict(d)
        try:
            w = float(dd.get("weight") or 0.0) / tots
        except Exception:
            w = 0.0
        dd["weight"] = max(0.0, min(1.0, w))
        out.append(dd)
    return out


def generate_rubric_llm(final: Dict[str, Any], citations: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    _ensure_dirs()
    venue = str(get("research.venue", "MICCAI|CVPR medical imaging tracks") or "MICCAI|CVPR medical imaging tracks")
    num_dims = int(get("tier1.rubric.num_dims", 7) or 7)
    signals = _signals_from_final(final)

    system = (
        "You are a senior program committee member creating a tier‑1 review rubric.\n"
        "Produce 6–9 dimensions with short prompts and weights that sum to 1.0.\n"
        "Cover originality, baselines/ablations, generalization/OOD, fairness/bias, compute/reproducibility, clarity/clinical/ethics as applicable.\n"
        "Return strictly JSON."
    )
    user = {
        "venue": venue,
        "num_dimensions": max(6, min(9, num_dims)),
        "constraints": {"compute": "<= 1 epoch, small steps", "repro": "minimal deps, deterministic"},
        "signals": signals,
        "citations_available": bool(citations),
        "output_schema": {
            "venue": "string",
            "dimensions": [
                {
                    "name": "string",
                    "prompt": "string",
                    "weight": 0.1,
                    "pass_criteria": ["string"],
                    "failure_modes": ["string"],
                }
            ],
        },
        "self_check": "Ensure 6–9 dimensions and weights sum to 1.0; return JSON only.",
    }

    try:
        js = chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.1)
        dims = js.get("dimensions") or []
        if not isinstance(dims, list) or len(dims) < 6:
            raise LLMError("Invalid rubric shape")
        dims = _normalize_weights(dims)
    except Exception:
        # Skinny fallback rubric with equal weights
        base = [
            {"name": n, "prompt": n, "weight": 1.0, "pass_criteria": [], "failure_modes": []}
            for n in [
                "originality", "baselines_ablation", "generalization_OOD",
                "fairness_bias", "compute_reproducibility", "clarity_ethics",
            ]
        ]
        dims = _normalize_weights(base)
        js = {"venue": venue, "dimensions": dims}

    meta = {
        "_meta": {
            "hash": _hash_final(final, venue, num_dims),
            "venue": venue,
            "dims": len(dims),
            "created_at": _now(),
        }
    }
    rubric = {"venue": venue, "dimensions": dims, **meta}
    RUBRIC_PATH.write_text(json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    return rubric


def grade_with_rubric_llm(final: Dict[str, Any], rubric: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_dirs()
    signals = _signals_from_final(final)

    dims = rubric.get("dimensions") or []
    out_dims: List[Dict[str, Any]] = []
    for d in dims:
        prompt = str(d.get("prompt") or d.get("name") or "Evaluate")
        system = (
            "You are a meta‑reviewer grading a paper strictly per the provided dimension.\n"
            "Return JSON only with score in [0,1] (granularity 0.05), reason, missing (concrete items), and quotes (exact sentences from signals)."
        )
        user = {
            "dimension": {"name": d.get("name"), "prompt": prompt, "weight": d.get("weight")},
            "paper_signals": signals,
            "final_json": final,
            "scoring_instructions": {"scale": "0..1 step 0.05", "quotes": True, "missing_items": True},
            "output_schema": {"score": 0.0, "reason": "string", "missing": ["string"], "quotes": ["string"]},
        }
        try:
            js = chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)
            score = float(js.get("score") or 0.0)
            reason = str(js.get("reason") or "")
            missing = js.get("missing") or []
            quotes = js.get("quotes") or []
        except Exception:
            score, reason, missing, quotes = 0.0, "LLM error", [], []
        out_dims.append({
            "name": d.get("name"),
            "weight": d.get("weight"),
            "score": max(0.0, min(1.0, score)),
            "reason": reason,
            "missing": missing if isinstance(missing, list) else [],
            "quotes": quotes if isinstance(quotes, list) else [],
        })

    # Weighted overall
    dims_n = _normalize_weights(out_dims)
    overall = 0.0
    for d in dims_n:
        try:
            overall += float(d.get("weight") or 0.0) * float(d.get("score") or 0.0)
        except Exception:
            pass
    grade = {"overall": round(overall, 4), "dimensions": dims_n, "_meta": rubric.get("_meta", {})}
    GRADE_PATH.write_text(json.dumps(grade, ensure_ascii=False, indent=2), encoding="utf-8")
    return grade


def build_blueprint_md_llm(final: Dict[str, Any], rubric: Dict[str, Any], grade: Dict[str, Any]) -> str:
    _ensure_dirs()
    # Try an LLM-composed blueprint first
    system = (
        "You are a senior researcher drafting a blueprint to upgrade a paper to tier‑1 quality.\n"
        "Use the user's sentences verbatim where quoting claim/objectives/contributions/RQs. Return Markdown only."
    )
    user = {
        "signals": _signals_from_final(final),
        "rubric": rubric,
        "grade": grade,
        "instructions": {
            "sections": [
                "Claim / Objectives / Contributions / RQs (verbatim quotes)",
                "Per‑dimension: weight, score, reason, missing → actions",
                "Overall score",
            ]
        }
    }
    try:
        md = chat_text_cached([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ], temperature=0.2)
        md = md.strip()
        if md:
            BLUEPRINT_PATH.write_text(md, encoding="utf-8")
            return md
    except LLMError:
        pass

    # Fallback deterministic builder
    lines: List[str] = []
    lines.append("# Paper Blueprint (LLM Fallback)")
    sig = _signals_from_final(final)
    lines.append("\n## Signals (verbatim)")
    for k in ["claim_like", "objectives", "contributions", "research_questions"]:
        lines.append(f"### {k}")
        for s in (sig.get(k) or []):
            lines.append(f"- {s}")
    lines.append("\n## Rubric & Grades")
    for d in (grade.get("dimensions") or []):
        lines.append(f"### {d.get('name')} — w={float(d.get('weight') or 0):.2f} score={float(d.get('score') or 0):.2f}")
        if d.get("reason"):
            lines.append(f"- Reason: {d.get('reason')}")
        for m in (d.get("missing") or []):
            lines.append(f"- Missing→Action: {m}")
        if d.get("quotes"):
            lines.append("- Quotes:")
            for q in d.get("quotes"):
                lines.append(f"  > {q}")
    lines.append(f"\n## Overall\n- Score: {float(grade.get('overall') or 0):.2f}")
    md = "\n".join(lines)
    BLUEPRINT_PATH.write_text(md, encoding="utf-8")
    return md


def _assert_tier1_ready(grade: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic pass/fail gate using environment floors."""
    import os as _os, json as _json
    overall_min = float(_os.getenv("TIER1_OVERALL_MIN", "0.70"))
    dim_floor = float(_os.getenv("TIER1_DIM_FLOOR", "0.60"))
    dims = grade.get("dimensions") or []
    failures: List[str] = []
    try:
        overall = float(grade.get("overall") or 0.0)
    except Exception:
        overall = 0.0
    if overall < overall_min:
        failures.append(f"overall<{overall_min:.2f}")
    for d in dims:
        nm = str(d.get("name") or "")
        try:
            sc = float(d.get("score") or 0.0)
        except Exception:
            sc = 0.0
        if sc < dim_floor:
            failures.append(f"{nm}<{dim_floor:.2f}")
    gate = {"pass": not failures, "failures": failures, "overall_min": overall_min, "dim_floor": dim_floor}
    try:
        GATE_PATH.write_text(_json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return gate


def _load_user_spec_for_plan() -> Dict[str, Any]:
    import os as _os, json as _json, pathlib as _pl
    p = _pl.Path(_os.getenv("USER_SPEC_PATH", "data/user_spec.json"))
    if p.exists():
        try:
            return _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_upgrade_plan(final: Dict[str, Any], rubric: Dict[str, Any], grade: Dict[str, Any], gate: Dict[str, Any]) -> str:
    """Deterministic Tier‑1 upgrade plan builder, writes Markdown."""
    _ensure_dirs()
    spec = _load_user_spec_for_plan()
    obj = spec.get("objective", "") or (final.get("problems") or [""])[0]
    hyps = [h.strip() for h in (spec.get("hypotheses") or []) if isinstance(h, str) and h.strip()]
    if not hyps:
        hyps = [h for h in (final.get("research_questions") or [])[:3] if isinstance(h, str) and h.strip()]

    lines: List[str] = []
    lines.append("# Tier-1 Journal Upgrade Plan")
    if obj:
        lines.append("\n> **Objective (verbatim):** " + obj)
    if hyps:
        lines.append("\n> **Hypotheses (verbatim):**")
        for h in hyps:
            lines.append(f"> - {h}")

    lines.append("""
---
## Methodological Upgrades
### Constrained FiLM (cFiLM)
- Identity-anchored init (γ≈1, β≈0), residual reparam with E[Δγ]=E[Δβ]=0.
- Sparsity on (γ−1,β) via group-Lasso by metadata factor; orthogonality gating with low-rank W to restrict modulation subspace.
- Stability penalty on (γ,β) trajectories; per-factor masks (age, sex, site) logged and visualized.

### Uncertainty-Mass Selection (UMS)
- Score S(box)=α·CAM_conf − β·MC_var + γ·CAM_mass_in_box with per-image z-score normalization; ∥[α,β,γ]∥₁=1.
- Select α,β,γ by small CV to maximize AP@0.5_small subject to FP/image≤κ.
- Curriculum: strict→relaxed gating driven by detector calibration (ECE) and FP/image caps.

---
## Experimental Program
### Stage-A (≤100 steps, sanity and pilot)
- Keep your T1–T10 quick-smoke protocol; 2 seeds; bootstrap CIs; early-stop.
### Stage-B (full budget for journal evidence)
- ≥2 datasets with metadata + localization; one external/OOD. Site-stratified splits.
- Strong baselines: image-only, concat, shuffled-metadata FiLM, attention-gated fusion, MIL-WSOD, entropy/ensemble uncertainty.
- Metrics: AUROC/AUPRC, sensitivity@95%, ECE/Brier, decision curves; AP@0.5_small, FROC, objectness calibration; subgroup/OOD deltas.
- Stats: DeLong, paired bootstrap, McNemar; FDR control.
---
## Ablations & Sensitivity (must report)
- cFiLM removals; conv4 vs conv5; metadata shuffles; capacity-matched adapters.
- UMS term ablations; CAM variants; MC-T sweeps; curriculum schedules.
- Budget sensitivity: ≤100-step vs full epoch trends; seed variance.
---
## Acceptance Gates (hard)
- ΔAUROC ≥ 0.02 & Δsens@95% ≥ 0.03 vs strongest baseline on 2 datasets (non-overlapping 95% CIs; FDR q≤0.05).
- ΔAP@0.5_small ≥ 0.03 with FP/image ≤ +20%; non-overlapping 95% CIs on 2 datasets.
- Fairness: no subgroup drop >10%; OOD retains ≥70% of in-domain gains.
- Calibration not degraded.
---
## Reproducibility & Artifacts
- Deterministic seeds, pinned env, exact param counts (FiLM ±5%).
- Release configs, logs, pseudo-labels, threshold curves, and code.
""")
    lines.append("\n---\n\n## Tier-1 Gate Summary")
    lines.append(f"- PASS: {bool(gate.get('pass'))}")
    if gate.get("failures"):
        lines.append("- Fail reasons:")
        for f in gate.get("failures"):
            lines.append(f"  - {f}")
    lines.append(f"- Floors: overall≥{gate.get('overall_min')} and per-dimension≥{gate.get('dim_floor')}")
    md = "\n".join(lines)
    try:
        UPGRADE_PLAN_PATH.write_text(md, encoding="utf-8")
    except Exception:
        pass
    return md


def run_tier1_validator(final: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_dirs()
    try:
        citations = final.get("citations") or []
        rubric = generate_rubric_llm(final, citations=citations)
        grade = grade_with_rubric_llm(final, rubric)
        gate = _assert_tier1_ready(grade)
        build_blueprint_md_llm(final, rubric, grade)
        build_upgrade_plan(final, rubric, grade, gate)
        return {"rubric": rubric, "grade": grade, "gate": gate}
    except Exception as exc:
        # Write minimal artifacts on failure for auditability
        try:
            if not RUBRIC_PATH.exists():
                RUBRIC_PATH.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
            if not GRADE_PATH.exists():
                GRADE_PATH.write_text(json.dumps({"overall": 0.0, "dimensions": []}, ensure_ascii=False, indent=2), encoding="utf-8")
            if not BLUEPRINT_PATH.exists():
                BLUEPRINT_PATH.write_text("# Paper Blueprint\n(validator failure)", encoding="utf-8")
        except Exception:
            pass
        raise exc
