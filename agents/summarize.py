import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from utils.pdf_utils import extract_text_from_pdf
from utils.llm_utils import chat_json, LLMError
from lab.config import get as _cfg_get


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


# ---- Deterministic metadata extraction helpers ----
def _as_str(x: Any) -> str:
    return "" if x is None else str(x)

def _as_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    if x in (None, "", []):
        return []
    return [str(x)]

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

# ---- All‑LLM 3‑pass extraction helpers ----

def _llm_pass1_extraction(
    text: str,
    *,
    timeout: Optional[int] = None,
    max_tries: Optional[int] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    system = (
        "You are an expert scientific information extractor. From the provided paper text only, produce a broad extraction pack JSON for downstream summarization. "
        "No outside knowledge. If something isn’t in the text, leave it blank. Keep each string under ~240 chars. Use normalized short names for models/metrics/datasets. JSON only."
    )
    user = {
        "paper_text": text,
        "want": {
            "meta": {"title": "string", "authors": ["string"], "venue": "string", "year": "string", "doi": "string", "url": "string"},
            "sections": {
                "abstract": ["string"],
                "introduction": ["string"],
                "methods": ["string"],
                "results": ["string"],
                "discussion": ["string"],
                "conclusion": ["string"],
            },
            "paper_type": "review|research",
            "datasets": ["string"],
            "metrics": ["string"],
            "results_numbers": [{"metric": "string", "value": "string", "dataset": "string", "split": "string", "baseline": "string", "improvement": "string"}],
            "results": ["string"],
            "ablations": ["string"],
            "baselines": ["string"],
            "novelty_claims": ["string"],
            "limitations": ["string"],
            "resources": {"code_url": "string", "data_url": "string", "model_url": "string"},
            "reproducibility": {"code_available": "string", "data_available": "string", "artifacts": ["string"]},
            "captions": {"tables": ["string"], "figures": ["string"]},
            "coverage_report": {"missing": ["string"], "notes": ["string"]},
            "related_work": [
                {
                    "title": "string",
                    "citation": "string",
                    "venue": "string",
                    "year": "string",
                    "doi": "string",
                    "url": "string",
                    "relation": "string",
                    "method_summary": "string",
                    "methods": ["string"],
                    "datasets": ["string"],
                    "metrics": ["string"],
                    "results_notes": ["string"]
                }
            ],
        },
        "rules": [
            "Facts only from text; if absent -> empty string/list.",
            "Normalize names: e.g., resnet18, cifar10, accuracy, auc, f1.",
            "Numbers → put in results_numbers; include split/dataset if stated.",
            "Short bullet phrases; dedupe; no speculation; JSON only.",
            "For related_work, extract at most 12 entries grounded in the paper text (e.g., References and in-text comparisons). Use concise citation strings like 'Author et al., YEAR — note'.",
        ],
    }
    kwargs: Dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = int(timeout)
    if max_tries is not None:
        kwargs["max_tries"] = int(max_tries)
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, **kwargs)


def _llm_pass2_gap_fill(
    text: str,
    pack: Dict[str, Any],
    *,
    timeout: Optional[int] = None,
    max_tries: Optional[int] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    system = (
        "You are a meticulous auditor. Given an extraction pack and the same paper text, fill only missing/uncertain fields. Do not change already-filled correct fields. "
        "If still not present in the text, keep blank and record in coverage_report.missing_after_second_pass. JSON only."
    )
    user = {"paper_text": text, "extraction_pack": pack}
    kwargs: Dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = int(timeout)
    if max_tries is not None:
        kwargs["max_tries"] = int(max_tries)
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, **kwargs)


def _llm_merge_packs(
    packs: List[Dict[str, Any]],
    *,
    timeout: Optional[int] = None,
    max_tries: Optional[int] = None,
    model: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    if len(packs) == 1:
        return packs[0]
    system = (
        "You are a careful merger. Given multiple extraction packs of the same paper chunks, merge them into a single pack with the same shape. "
        "Keep filled fields, dedupe lists, and preserve coverage_report notes. JSON only."
    )
    user = {"packs": packs}
    kwargs: Dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = int(timeout)
    if max_tries is not None:
        kwargs["max_tries"] = int(max_tries)
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, **kwargs)


def _llm_pass3_finalize_schema(merged_pack: Dict[str, Any], *, model: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    system = (
        "You are a strict JSON formatter. Convert the merged extraction pack into the EXACT target schema. No extra keys. Scalars must be strings; lists must be arrays of strings. "
        "Keep each list item concise (<240 chars). If a field is unknown, leave it empty ('' or []). JSON only."
    )
    target_schema = {
        "title": "string",
        "authors": ["string"],
        "venue": "string",
        "year": "string",
        "doi": "string",
        "url": "string",
        "paper_type": "review|research",
        "problem": "string",
        "background": ["string"],
        "research_questions": ["string"],
        "hypotheses": ["string"],
        "tasks": ["string"],
        "methods": ["string"],
        "architecture": ["string"],
        "training_procedure": ["string"],
        "hyperparameters": ["string"],
        "datasets": ["string"],
        "dataset_details": [{"name": "string", "size": "string", "splits": "string", "preprocessing": "string", "notes": "string"}],
        "metrics": ["string"],
        "results": ["string"],
        "results_numbers": [{"metric": "string", "value": "string", "dataset": "string", "split": "string", "baseline": "string", "improvement": "string"}],
        "ablations": ["string"],
        "baselines": ["string"],
        "novelty_claims": ["string"],
        "novelty_type": ["string"],
        "prior_work_overlap": ["string"],
        "limitations": ["string"],
        "failure_modes": ["string"],
        "ethical": ["string"],
        "threats_to_validity": ["string"],
        "reproducibility": {"code_available": "string", "data_available": "string", "artifacts": ["string"]},
        "resources": {"code_url": "string", "data_url": "string", "model_url": "string"},
        "conclusions": ["string"],
        "future_work": ["string"],
        "keywords": ["string"],
        "confidence": "0.0-1.0 float as string",
        "related_work": [
            {
                "title": "string",
                "citation": "string",
                "venue": "string",
                "year": "string",
                "doi": "string",
                "url": "string",
                "relation": "string",
                "method_summary": "string",
                "methods": ["string"],
                "datasets": ["string"],
                "metrics": ["string"],
                "results_notes": ["string"]
            }
        ],
    }
    user = {
        "extraction_pack": merged_pack,
        "target_schema": target_schema,
        "normalization_rules": [
            "Normalize model/dataset/metric names to short common forms.",
            "Classify paper_type as 'review' only if the text clearly indicates survey/review/systematic review/meta-analysis; otherwise 'research'.",
            "Compute improvement if both baseline and new value exist; else ''.",
            "Do not invent DOIs/URLs/venues; leave empty if not explicit in the text.",
            "Do not invent anything",
            # Must-capture additions
            "Training knobs (normalized 'key: value' strings) MUST appear in training_procedure[]. Include empty values if unknown:"
            "epochs: , batch_size: , optimizer: , learning_rate: , lr_scheduler: , weight_decay: , loss: , class_weights: , imbalance_strategy: , early_stopping: , seed: , mixed_precision: , hardware: ",
            "Hyperparameters[] MUST mirror key knobs as normalized pairs (include empty if unknown): batch_size: , optimizer: , lr: , epochs: , weight_decay: , dropout: , label_smoothing: , betas: ",
            "Augmentations: include 'aug: ...' entries with magnitudes when present (rotate, flip, scale, color_jitter, noise).",
            "Metrics MUST include 'accuracy' and an AUC entry with explicit mode: 'auc (macro|micro|ovr|unspecified)'; include precision/recall/f1 with averaging tags when stated or '(unspecified)' otherwise; include 'confusion_matrix'.",
            "results_numbers rows should use normalized fields: metric, value, dataset, split (train|val|test), baseline, improvement; prefer per-model rows when present.",
            "If dataset_details[] exists, ensure each object has keys name, size, splits, preprocessing, notes (empty string when unknown). If none exist but datasets[] is present, add one combined entry with empty keys except name.",
            "If the paper uses YOLO/detection, include YOLO-specific strings within architecture[]/training_procedure[] when present: yolo_version:, backbone:, img_size:, anchors:, conf_thresh:, iou_thresh:, nms:, task_adapt.",
            "For related_work[], keep ≤12 entries; dedupe by DOI/title; include a short 'citation' string and optional DOI/URL when explicitly present; 'relation' should briefly explain the connection (baseline, similar method, prior SOTA). For each item include method_summary (<=200 chars) and, when stated in the text, methods[], datasets[], metrics[], and results_notes[]."
        ],
    }
    kwargs: Dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, **kwargs)


def _llm_meta_fill(text: str, pack: Dict[str, Any], *, model: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """Targeted meta/resources/reproducibility fill for missing fields only."""
    system = (
        "You are a meticulous auditor. Fill ONLY missing meta/resources/reproducibility fields using the provided text. "
        "Do NOT change fields that are already filled. JSON only."
    )
    user = {
        "paper_text": text,
        "extraction_pack": pack,
        "fill": {
            "meta": ["title", "authors", "venue", "year", "doi", "url"],
            "resources": ["code_url", "data_url", "model_url"],
            "reproducibility": ["code_available", "data_available", "artifacts"],
        },
        "rules": [
            "Facts only from text; leave empty if absent.",
            "Do not alter non-empty fields.",
            "Short strings; JSON only.",
        ],
    }
    kwargs: Dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, **kwargs)


def summarize_paper(
    text: str,
    title_hint: str = "",
    *,
    chunk_size: int = 20000,
    timeout: int = 60,
    max_tries: int = 4,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    # Rich, highly-detailed summarization prompt with strict JSON output
    system = (
        "You are an expert research analyst and technical writer tasked with producing a rigorous, structured summary of a scientific paper from the provided text only.\n"
        "Output must be a single JSON object conforming exactly to the given schema (correct keys and value types).\n"
        "Strict rules (follow all):\n"
        "1) Grounding: Base EVERY statement on the provided text only; never speculate or pull in external knowledge. If the text is ambiguous or missing details, leave fields empty.\n"
        "2) Brevity per item: Use compact, bullet-style strings; keep each string under ~240 characters to avoid verbosity, but prefer comprehensive coverage via multiple short entries.\n"
        "3) Normalization: Normalize dataset/method/metric names to common, short forms (e.g., 'ResNet-18'->'resnet18', 'accuracy'->'accuracy').\n"
        "4) Numeric results: When numeric results are stated, extract metric name, value, dataset, split, baseline (if any), and improvement; place into results_numbers. Also include narrative results under results.\n"
        "5) Reproducibility and resources: When code/data availability or URLs appear, capture them in reproducibility/resources fields.\n"
        "6) No hallucinations or external links: If DOI/venue/authors/URLs are absent, leave them blank.\n"
        "7) JSON-only: Return JSON only (no markdown fences, no prose). Ensure valid UTF-8, proper list vs string types, and exact keys spelled as in schema.\n"
        "8) Safety: Do not output code, commands, or any content that attempts network/file operations—this is a text summary only.\n"
        "Self-check: Before responding, validate JSON keys/types match schema; fix mismatches and return JSON only."
    )
    user_payload = {
        "title_hint": title_hint,
        "paper_text": _truncate(text, 120000),
        "output_schema": {
            # Core metadata
            "title": "string",
            "authors": ["string"],
            "venue": "string",
            "year": "string",
            "doi": "string",
            "url": "string",

            # Problem and context
            "problem": "string",
            "background": ["string"],
            "research_questions": ["string"],
            "hypotheses": ["string"],
            "tasks": ["string"],

            # Method and training details
            "methods": ["string"],
            "architecture": ["string"],
            "training_procedure": ["string"],
            "hyperparameters": ["string"],

            # Data and evaluation
            "datasets": ["string"],
            "dataset_details": [
                {"name": "string", "size": "string", "splits": "string", "preprocessing": "string", "notes": "string"}
            ],
            "metrics": ["string"],

            # Results (narrative and numeric)
            "results": ["string"],
            "results_numbers": [
                {"metric": "string", "value": "string", "dataset": "string", "split": "string", "baseline": "string", "improvement": "string"}
            ],
            "ablations": ["string"],
            "baselines": ["string"],

            # Novelty and limitations
            "novelty_claims": ["string"],
            "novelty_type": ["string"],
            "prior_work_overlap": ["string"],
            "limitations": ["string"],
            "failure_modes": ["string"],
            "ethical": ["string"],
            "threats_to_validity": ["string"],

            # Reproducibility and resources
            "reproducibility": {"code_available": "string", "data_available": "string", "artifacts": ["string"]},
            "resources": {"code_url": "string", "data_url": "string", "model_url": "string"},

            # Conclusions and keywords
            "conclusions": ["string"],
            "future_work": ["string"],
            "keywords": ["string"],
            "confidence": "0.0-1.0 float as string",
            "related_work": [
                {
                    "title": "string",
                    "citation": "string",
                    "venue": "string",
                    "year": "string",
                    "doi": "string",
                    "url": "string",
                    "relation": "string",
                    "method_summary": "string",
                    "methods": ["string"],
                    "datasets": ["string"],
                    "metrics": ["string"],
                    "results_notes": ["string"]
                }
            ]
        },
        "instructions": (
            "Extract exact facts from the text. Use short phrases, not paragraphs. Where numeric comparisons are given, include them in results_numbers. "
            "If dates/venues/DOIs are missing, leave blank. "
            "MUST include normalized training knobs as 'key: value' in training_procedure and mirrored in hyperparameters (use empty values if unknown): epochs, batch_size, optimizer, learning_rate (or lr), lr_scheduler, weight_decay, loss, class_weights, imbalance_strategy, early_stopping, seed, mixed_precision, hardware. "
            "Include augmentation lines as 'aug: ...' with magnitudes when present. "
            "Metrics MUST include 'accuracy' and 'auc (macro|micro|ovr|unspecified)' plus precision/recall/f1 with averaging tags (or '(unspecified)' if not stated), and 'confusion_matrix'. "
            "Ensure dataset_details[] objects include keys: name, size, splits, preprocessing, notes (use '' when unknown)."
            " Extract up to 12 related_work items from the paper text (References and in-text comparisons). Each should include a concise 'citation' string and optional DOI/URL if explicitly present. Also include method_summary and, when stated, methods[], datasets[], metrics[], results_notes[] that characterize how that prior work approaches the problem."
        ),
    }
    # All‑LLM 3‑pass flow with deterministic chunking (no offline fallbacks)
    full_text = text
    if chunk_size <= 0:
        chunk_size = len(full_text) or 1

    packs: List[Dict[str, Any]] = []
    if len(full_text) <= chunk_size:
        if verbose:
            print(f"[SUM] Pass1: 1 chunk (size={len(full_text)})")
        pck = _llm_pass1_extraction(full_text, timeout=timeout, max_tries=max_tries, model=model, profile=profile)
        packs.append(pck)
    else:
        n_chunks = (len(full_text) + chunk_size - 1) // chunk_size
        if verbose:
            print(f"[SUM] Pass1: {n_chunks} chunks (chunk_size={chunk_size}, total={len(full_text)})")
        for i in range(0, len(full_text), chunk_size):
            idx = (i // chunk_size) + 1
            if verbose:
                print(f"[SUM]  • chunk {idx}/{n_chunks} …")
            ch = _llm_pass1_extraction(full_text[i : i + chunk_size], timeout=timeout, max_tries=max_tries, model=model, profile=profile)
            packs.append(ch)

    # Merge packs (LLM)
    if verbose:
        print("[SUM] Merge extraction packs …")
    merged = _llm_merge_packs(packs, timeout=timeout, max_tries=max_tries, model=model, profile=profile)

    # Pass 2: gap fill
    if verbose:
        print("[SUM] Gap‑fill missing fields …")
    merged2 = _llm_pass2_gap_fill(full_text, merged, timeout=timeout, max_tries=max_tries, model=model, profile=profile)

    # Targeted meta/resources fill if still missing
    missing_meta = []
    mp = merged2.get("meta", {}) if isinstance(merged2.get("meta"), dict) else {}
    for k in ("title","venue","year","doi","url"):
        if not _as_str(mp.get(k)):
            missing_meta.append(k)
    res = merged2.get("resources", {}) if isinstance(merged2.get("resources"), dict) else {}
    if missing_meta or not _as_str(res.get("code_url")) or not _as_str(res.get("data_url")):
        if verbose:
            print("[SUM] Targeted meta/resources fill …")
        try:
            merged2 = _llm_meta_fill(full_text, merged2, model=model, profile=profile)
        except LLMError:
            if verbose:
                print("[SUM]  • meta/resources fill skipped due to LLM error.")

    # Pass 3: finalize to target schema
    if verbose:
        print("[SUM] Finalize to target schema …")
    js = _llm_pass3_finalize_schema(merged2, model=model, profile=profile)

    # Optional: JSON schema validation/repair (if jsonschema installed)
    def _validate_or_repair(j: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import jsonschema  # type: ignore
        except Exception:
            return j
        SUMMARY_SCHEMA = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "paper_type": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "venue": {"type": "string"},
                "year": {"type": "string"},
                "doi": {"type": "string"},
                "url": {"type": "string"},
                "problem": {"type": "string"},
                "background": {"type": "array", "items": {"type": "string"}},
                "research_questions": {"type": "array", "items": {"type": "string"}},
                "hypotheses": {"type": "array", "items": {"type": "string"}},
                "tasks": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
                "architecture": {"type": "array", "items": {"type": "string"}},
                "training_procedure": {"type": "array", "items": {"type": "string"}},
                "hyperparameters": {"type": "array", "items": {"type": "string"}},
                "datasets": {"type": "array", "items": {"type": "string"}},
                "dataset_details": {"type": "array"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "results": {"type": "array", "items": {"type": "string"}},
                "results_numbers": {"type": "array"},
                "ablations": {"type": "array", "items": {"type": "string"}},
                "baselines": {"type": "array", "items": {"type": "string"}},
                "novelty_claims": {"type": "array", "items": {"type": "string"}},
                "novelty_type": {"type": "array", "items": {"type": "string"}},
                "prior_work_overlap": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "failure_modes": {"type": "array", "items": {"type": "string"}},
                "ethical": {"type": "array", "items": {"type": "string"}},
                "threats_to_validity": {"type": "array", "items": {"type": "string"}},
                "reproducibility": {"type": "object"},
                "resources": {"type": "object"},
                "conclusions": {"type": "array", "items": {"type": "string"}},
                "future_work": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string"},
                "related_work": {"type": "array"},
            },
            "required": ["title", "problem", "methods", "datasets", "results", "keywords", "confidence"],
        }
        try:
            jsonschema.validate(j, SUMMARY_SCHEMA)
            return j
        except Exception:
            # Surgical repairs
            for k in (
                "authors", "background", "research_questions", "hypotheses", "tasks", "methods", "architecture",
                "training_procedure", "hyperparameters", "datasets", "metrics", "results", "ablations", "baselines",
                "novelty_claims", "novelty_type", "prior_work_overlap", "limitations", "failure_modes", "ethical",
                "threats_to_validity", "conclusions", "future_work", "keywords",
            ):
                if k in j and not isinstance(j[k], list):
                    j[k] = [j[k]] if j[k] not in (None, "") else []
            for k in ("venue", "year", "doi", "url", "title", "problem", "confidence"):
                if k in j and isinstance(j[k], list):
                    j[k] = " ".join([str(v) for v in j[k]])
            j.setdefault("reproducibility", {"code_available": "", "data_available": "", "artifacts": []})
            j.setdefault("resources", {"code_url": "", "data_url": "", "model_url": ""})
            return j

    js = _validate_or_repair(js)

    # Enforce must-capture minimal presence and normalization
    def _ensure_list(val):
        return val if isinstance(val, list) else ([] if val in (None, "") else [str(val)])

    def _ensure_kv_presence(lines: List[str], keys: List[str]) -> List[str]:
        have = {str(x).strip().lower().split(":", 1)[0] + ":" for x in lines if ":" in str(x)}
        out = list(lines)
        for k in keys:
            kk = k.lower()
            if kk not in have:
                out.append(f"{k} ")
        return out

    def _metrics_with_required(metrics: List[str], rn: List[Dict[str, Any]]) -> List[str]:
        """Preserve existing metrics and repair AUC mode tags without dropping valid entries.

        - Do not remove pre-existing metrics with explicit modes (e.g., 'auc (macro)').
        - If an AUC entry lacks a mode, replace that exact entry with 'auc (unspecified)'.
        - Optionally add 'accuracy' if missing (kept for backward-compat).
        - Optionally add a single 'auc (unspecified)' only when there is evidence of AUC in results_numbers and no AUC in metrics at all.
        """
        out = list(metrics)
        mset = {str(m).strip().lower() for m in out}

        # Ensure accuracy present (backward-compat; can be removed if strict non-invention is desired)
        if "accuracy" not in mset:
            out.append("accuracy")
            mset.add("accuracy")

        # Repair AUC entries missing a mode; do not drop correctly formed ones
        replaced_any = False
        for idx, m in enumerate(list(out)):
            s = str(m).strip()
            if s.lower().startswith("auc") and "(" not in s:
                out[idx] = "auc (unspecified)"
                replaced_any = True

        # If there is no AUC in metrics at all, but results_numbers contain AUC rows, add one generic entry
        has_auc_metric = any(str(m).strip().lower().startswith("auc") for m in out)
        has_auc_rn = any(str((r.get("metric") or "")).lower().startswith("auc") for r in (rn or []) if isinstance(r, dict))
        if not has_auc_metric and has_auc_rn:
            out.append("auc (unspecified)")

        return out

    enforce_required = True

    if enforce_required:
        # training_procedure
        tp_lines = _ensure_list(js.get("training_procedure"))
        tp_required = [
            "epochs:", "batch_size:", "optimizer:", "learning_rate:", "lr_scheduler:", "weight_decay:", "loss:",
            "class_weights:", "imbalance_strategy:", "early_stopping:", "seed:", "mixed_precision:", "hardware:",
        ]
        js["training_procedure"] = _ensure_kv_presence(tp_lines, tp_required)

        # hyperparameters mirror
        hp_lines = _ensure_list(js.get("hyperparameters"))
        hp_required = [
            "batch_size:", "optimizer:", "lr:", "epochs:", "weight_decay:", "dropout:", "label_smoothing:", "betas:",
        ]
        js["hyperparameters"] = _ensure_kv_presence(hp_lines, hp_required)

        # Do not inject placeholder augmentations; keep only what the model returned

        # metrics enforcement
        js["metrics"] = _metrics_with_required(_ensure_list(js.get("metrics")), js.get("results_numbers") or [])

        # dataset_details presence and keys
        dds = js.get("dataset_details")
        if not isinstance(dds, list) or len(dds) == 0:
            datasets = _ensure_list(js.get("datasets"))
            name = "combined" if len(datasets) > 1 else (datasets[0] if datasets else "combined")
            dds = [{"name": str(name), "size": "", "splits": "", "preprocessing": "", "notes": ""}]
        fixed_dds = []
        for dd in dds:
            if not isinstance(dd, dict):
                continue
            fixed_dds.append({
                "name": str(dd.get("name") or ""),
                "size": str(dd.get("size") or ""),
                "splits": str(dd.get("splits") or ""),
                "preprocessing": str(dd.get("preprocessing") or ""),
                "notes": str(dd.get("notes") or ""),
            })
        js["dataset_details"] = fixed_dds

    # Build a detailed, backward-compatible summary object with correct types
    out: Dict[str, Any] = {
        "title": _as_str(js.get("title") or title_hint),
        "paper_type": _as_str(js.get("paper_type")),
        "problem": _as_str(js.get("problem")),
        "methods": _as_list(js.get("methods")),
        "datasets": _as_list(js.get("datasets")),
        "results": _as_list(js.get("results")),
        "novelty_claims": _as_list(js.get("novelty_claims")),
        "limitations": _as_list(js.get("limitations")),
        "keywords": _as_list(js.get("keywords")),
        "confidence": _as_str(js.get("confidence")),
    }

    # Scalars
    out["venue"] = _as_str(js.get("venue"))
    out["year"] = _as_str(js.get("year"))
    out["doi"] = _as_str(js.get("doi"))
    out["url"] = _as_str(js.get("url"))

    # Lists
    for k in (
        "authors", "background", "research_questions", "hypotheses", "tasks", "architecture", "training_procedure",
        "hyperparameters", "metrics", "ablations", "baselines", "novelty_type", "prior_work_overlap", "failure_modes",
        "ethical", "threats_to_validity", "conclusions", "future_work",
    ):
        out[k] = _as_list(js.get(k))

    # Structured lists/dicts
    dd = js.get("dataset_details")
    out["dataset_details"] = dd if isinstance(dd, list) else []
    rn = js.get("results_numbers")
    out["results_numbers"] = rn if isinstance(rn, list) else []
    out["reproducibility"] = _as_dict(js.get("reproducibility"))
    out["resources"] = _as_dict(js.get("resources"))

    # Related work normalization (list of objects with citation fields)
    def _fix_related_work(x: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(x, list):
            for it in x:
                if isinstance(it, dict):
                    items.append({
                        "title": _as_str(it.get("title")),
                        "citation": _as_str(it.get("citation")),
                        "venue": _as_str(it.get("venue")),
                        "year": _as_str(it.get("year")),
                        "doi": _as_str(it.get("doi")),
                        "url": _as_str(it.get("url")),
                        "relation": _as_str(it.get("relation")),
                        "method_summary": _as_str(it.get("method_summary")),
                        "methods": _as_list(it.get("methods")),
                        "datasets": _as_list(it.get("datasets")),
                        "metrics": _as_list(it.get("metrics")),
                        "results_notes": _as_list(it.get("results_notes")),
                    })
                else:
                    s = _as_str(it)
                    if s:
                        items.append({
                            "title": "",
                            "citation": s,
                            "venue": "",
                            "year": "",
                            "doi": "",
                            "url": "",
                            "relation": "",
                            "method_summary": "",
                            "methods": [],
                            "datasets": [],
                            "metrics": [],
                            "results_notes": [],
                        })
        return items
    out["related_work"] = _fix_related_work(js.get("related_work"))

    # Heuristic fallback for paper_type when missing
    def _infer_paper_type(title: str, background: List[str]) -> str:
        t = (title or "").lower()
        if any(k in t for k in ["review", "survey", "systematic", "meta-analysis", "bibliometric"]):
            return "review"
        bg = " ".join(background or []).lower()
        if any(k in bg for k in ["this review", "we review", "survey of", "systematic review", "meta-analysis"]):
            return "review"
        return "research"
    if not out.get("paper_type"):
        out["paper_type"] = _infer_paper_type(out.get("title", ""), out.get("background", []))

    # Optional: fill missing values with a clear note instead of blanks
    def _fill_not_mentioned(obj: Any) -> Any:
        note = "not mentioned in the paper"
        if isinstance(obj, dict):
            return {k: _fill_not_mentioned(v) for k, v in obj.items()}
        if isinstance(obj, list):
            if not obj:
                return [note]
            return [_fill_not_mentioned(v) for v in obj]
        if isinstance(obj, str):
            return obj if obj.strip() else note
        return obj

    try:
        from lab.config import get_bool as _get_bool
        fill_missing = bool(_get_bool("pipeline.summarize.fill_not_mentioned", True))
    except Exception:
        fill_missing = True
    if fill_missing:
        # Avoid overriding title with note; handle others
        title_val = out.get("title", "")
        out = _fill_not_mentioned(out)
        out["title"] = title_val if str(title_val).strip() else out["title"]

    # Completeness score (light)
    checks = [bool(out.get("title")), bool(out.get("problem")), bool(out.get("methods")), bool(out.get("datasets")), bool(out.get("results"))]
    try:
        out["_completeness"] = round(sum(1 for c in checks if c) / len(checks), 3)
    except Exception:
        out["_completeness"] = 0.0

    if verbose:
        print(
            "[SUM] Final stats: "
            f"title='{out.get('title','')[:60]}', venue='{out.get('venue','')}', year='{out.get('year','')}', doi_present={bool(out.get('doi'))}, "
            f"results_numbers={len(out.get('results_numbers',[]))}, dataset_details={len(out.get('dataset_details',[]))}, completeness={out.get('_completeness',0.0)}"
        )
    return out


def criticize_paper(summary: Dict[str, Any], *, timeout: int = 60, max_tries: int = 4, model: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    system = (
        "You are a skeptical reviewer. Given a structured summary, evaluate the strength of novelty, "
        "identify overlaps with common prior work (without external search), highlight potential weaknesses, "
        "and propose concrete follow-up experiments. Return JSON. Be concise and practical."
    )
    user_payload = {
        "summary": summary,
        "output_schema": {
            "novelty_strength": "low|medium|high",
            "possible_prior_overlap": ["string"],
            "weaknesses": ["string"],
            "followup_experiments": ["string"],
        }
    }
    kwargs: Dict[str, Any] = {}
    if model is not None:
        kwargs["model"] = model
    if profile is not None:
        kwargs["profile"] = profile
    kwargs["timeout"] = int(timeout)
    kwargs["max_tries"] = int(max_tries)
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0, **kwargs)
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]
    return {
        "novelty_strength": str(js.get("novelty_strength") or ""),
        "possible_prior_overlap": _as_list(js.get("possible_prior_overlap")),
        "weaknesses": _as_list(js.get("weaknesses")),
        "followup_experiments": _as_list(js.get("followup_experiments")),
    }

def summarize_pdf(
    pdf_path: str,
    *,
    title_hint: Optional[str] = None,
    max_pages: int = 0,
    max_chars: int = 0,
    chunk_size: int = 20000,
    timeout: int = 60,
    max_tries: int = 4,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper: extract text from a PDF and summarize it.

    No files are written; returns the structured summary dict.
    """
    text = extract_text_from_pdf(pdf_path, max_pages=max_pages, max_chars=max_chars)
    hint = title_hint if title_hint is not None else os.path.splitext(os.path.basename(str(pdf_path)))[0].replace("_", " ")
    return summarize_paper(
        text,
        title_hint=hint,
        chunk_size=chunk_size,
        timeout=timeout,
        max_tries=max_tries,
        model=model,
        profile=profile,
        verbose=verbose,
    )


def process_pdfs(
    pdf_dir: str | Path,
    out_dir: str | Path,
    *,
    max_pages: int = 0,
    max_chars: int = 0,
    chunk_size: int = 20000,
    timeout: int = 60,
    max_tries: int = 4,
    model: Optional[str] = None,
    profile: Optional[str] = None,
    verbose: bool = True,
    skip_existing: bool = True,
) -> int:
    """Process all PDFs in pdf_dir and write summary JSONs into out_dir.

    Returns the number of new summaries written.
    """
    pdf_dir_p = Path(pdf_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    if not pdf_dir_p.exists():
        if verbose:
            print(f"[SUM] PDF dir not found: {pdf_dir_p}")
        return 0
    pdfs = sorted([p for p in pdf_dir_p.glob("*.pdf") if p.is_file()])
    if not pdfs:
        if verbose:
            print("[SUM] No PDFs found to process.")
        return 0
    written = 0
    for i, pdf in enumerate(pdfs, start=1):
        out_path = out_dir_p / f"{pdf.stem}.json"
        if skip_existing and out_path.exists():
            if verbose:
                print(f"[SUM] Skip existing: {out_path.name}")
            continue
        if verbose:
            print(f"[SUM] ({i}/{len(pdfs)}) Extracting: {pdf.name}")
        try:
            text = extract_text_from_pdf(str(pdf), max_pages=max_pages, max_chars=max_chars)
        except Exception as exc:
            if verbose:
                print(f"[SUM]  • extract failed: {pdf.name} :: {exc}")
            continue
        try:
            summ = summarize_paper(
                text,
                title_hint=pdf.stem.replace("_", " "),
                chunk_size=chunk_size,
                timeout=timeout,
                max_tries=max_tries,
                model=model,
                profile=profile,
                verbose=verbose,
            )
        except LLMError as exc:
            if verbose:
                print(f"[SUM]  • summarize failed: {pdf.name} :: {exc}")
            continue
        try:
            crit = criticize_paper(summ, timeout=timeout, max_tries=max_tries, model=model, profile=profile)
        except LLMError as exc:
            if verbose:
                print(f"[SUM]  • critic failed: {pdf.name} :: {exc}")
            crit = {"novelty_strength": "", "possible_prior_overlap": [], "weaknesses": [], "followup_experiments": []}
        record = {"summary": summ, "critic": crit}
        # Optional embeddings (GPU-only, hard-fail when enabled)
        try:
            emb_enable = bool(_cfg_get("embeddings.enable", False))
        except Exception:
            emb_enable = False
        if emb_enable:
            from utils.embeddings import embed_and_store, compute_summary_text
            provider = str(_cfg_get("embeddings.provider", "huggingface") or "huggingface").strip().lower()
            model_name = str(_cfg_get("embeddings.model", "google/embeddinggemma-300m") or "google/embeddinggemma-300m")
            dtype = str(_cfg_get("embeddings.dtype", "float16") or "float16")
            bs = int(_cfg_get("embeddings.batch_size", 8) or 8)
            max_len = int(_cfg_get("embeddings.max_length", 1024) or 1024)
            content = str(_cfg_get("embeddings.content", "raw_pdf") or "raw_pdf").strip().lower()
            # Prefer raw PDF text as requested; else summary_text
            if content in {"raw_pdf", "both"}:
                _ = embed_and_store(pdf.stem, text, provider=provider, model=model_name, batch_size=bs, max_length=max_len, dtype=dtype)
            if content in {"summary_text", "both"}:
                st = compute_summary_text(summ)
                _ = embed_and_store(pdf.stem + "_summary", st, provider=provider, model=model_name, batch_size=bs, max_length=max_len, dtype=dtype)
        try:
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if verbose:
                print(f"[SUM]  • wrote {out_path}")
            written += 1
        except Exception as exc:
            if verbose:
                print(f"[SUM]  • write failed: {out_path.name} :: {exc}")
            continue
    if verbose:
        print(f"[SUM] Done. New summaries: {written}")
    return written
