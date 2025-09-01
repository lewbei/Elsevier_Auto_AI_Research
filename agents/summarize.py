import os
import json
import pathlib
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

from utils.pdf_utils import extract_text_from_pdf
from utils.llm_utils import chat_json, LLMError
from lab.config import get


load_dotenv()

DATA_DIR = pathlib.Path("data")
SUM_DIR = DATA_DIR / "summaries"
PDF_DIR = pathlib.Path("pdfs")


def _ensure_dirs() -> None:
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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

def _llm_pass1_extraction(text: str) -> Dict[str, Any]:
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
        },
        "rules": [
            "Facts only from text; if absent -> empty string/list.",
            "Normalize names: e.g., resnet18, cifar10, accuracy, auc, f1.",
            "Numbers → put in results_numbers; include split/dataset if stated.",
            "Short bullet phrases; dedupe; no speculation; JSON only.",
        ],
    }
    model = os.getenv("SUM_MODEL") or (get("pipeline.summarize.model", None) if isinstance(get("pipeline.summarize.model", None), str) else None)
    profile = get("pipeline.summarize.llm", None)
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)


def _llm_pass2_gap_fill(text: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a meticulous auditor. Given an extraction pack and the same paper text, fill only missing/uncertain fields. Do not change already-filled correct fields. "
        "If still not present in the text, keep blank and record in coverage_report.missing_after_second_pass. JSON only."
    )
    user = {"paper_text": text, "extraction_pack": pack}
    model = os.getenv("SUM_MODEL") or (get("pipeline.summarize.model", None) if isinstance(get("pipeline.summarize.model", None), str) else None)
    profile = get("pipeline.summarize.llm", None)
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)


def _llm_merge_packs(packs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(packs) == 1:
        return packs[0]
    system = (
        "You are a careful merger. Given multiple extraction packs of the same paper chunks, merge them into a single pack with the same shape. "
        "Keep filled fields, dedupe lists, and preserve coverage_report notes. JSON only."
    )
    user = {"packs": packs}
    model = os.getenv("SUM_MODEL") or (get("pipeline.summarize.model", None) if isinstance(get("pipeline.summarize.model", None), str) else None)
    profile = get("pipeline.summarize.llm", None)
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)


def _llm_pass3_finalize_schema(merged_pack: Dict[str, Any]) -> Dict[str, Any]:
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
    }
    user = {
        "extraction_pack": merged_pack,
        "target_schema": target_schema,
        "normalization_rules": [
            "Normalize model/dataset/metric names to short common forms.",
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
            "If the paper uses YOLO/detection, include YOLO-specific strings within architecture[]/training_procedure[] when present: yolo_version:, backbone:, img_size:, anchors:, conf_thresh:, iou_thresh:, nms:, task_adapt:."
        ],
    }
    model = os.getenv("SUM_MODEL") or (get("pipeline.summarize.model", None) if isinstance(get("pipeline.summarize.model", None), str) else None)
    profile = get("pipeline.summarize.llm", None)
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)


def _llm_meta_fill(text: str, pack: Dict[str, Any]) -> Dict[str, Any]:
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
    model = os.getenv("SUM_MODEL") or (get("pipeline.summarize.model", None) if isinstance(get("pipeline.summarize.model", None), str) else None)
    profile = get("pipeline.summarize.llm", None)
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0, model=model, profile=profile)


def summarize_paper(text: str, title_hint: str = "", file_stem: str | None = None) -> Dict[str, Any]:
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
        "paper_text": _truncate(text, 40000),
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
            "confidence": "0.0-1.0 float as string"
        },
        "instructions": (
            "Extract exact facts from the text. Use short phrases, not paragraphs. Where numeric comparisons are given, include them in results_numbers. "
            "If dates/venues/DOIs are missing, leave blank. "
            "MUST include normalized training knobs as 'key: value' in training_procedure and mirrored in hyperparameters (use empty values if unknown): epochs, batch_size, optimizer, learning_rate (or lr), lr_scheduler, weight_decay, loss, class_weights, imbalance_strategy, early_stopping, seed, mixed_precision, hardware. "
            "Include augmentation lines as 'aug: ...' with magnitudes when present. "
            "Metrics MUST include 'accuracy' and 'auc (macro|micro|ovr|unspecified)' plus precision/recall/f1 with averaging tags (or '(unspecified)' if not stated), and 'confusion_matrix'. "
            "Ensure dataset_details[] objects include keys: name, size, splits, preprocessing, notes (use '' when unknown)."
        ),
    }
    # All‑LLM 3‑pass flow with chunking support
    # Pass 1: extraction per chunk
    full_text = text
    packs: List[Dict[str, Any]] = []
    # If pass1_chunk <= 0, treat as "no chunking" when chunking is used.
    try:
        cfg_chunk = get("pipeline.summarize.pass1_chunk", None)
        chunk_size = int(os.getenv("SUM_PASS1_CHUNK", str(cfg_chunk if isinstance(cfg_chunk, int) else "20000")) or 20000)
    except Exception:
        chunk_size = 20000
    if chunk_size <= 0:
        chunk_size = len(full_text) or 1
    # Progress/detail toggles prefer YAML but allow env override
    try:
        yaml_progress = get("pipeline.summarize.progress", True)
        show_progress = (str(os.getenv("SUM_PROGRESS", "")).lower() in {"1", "true", "yes", "on"}) or bool(yaml_progress)
    except Exception:
        show_progress = True
    try:
        yaml_detail = get("pipeline.summarize.detail", False)
        show_detail = (str(os.getenv("SUM_DETAIL", "")).lower() in {"1", "true", "yes", "on"}) or bool(yaml_detail)
    except Exception:
        show_detail = False
    # Prefer a single full-text pass first, and only fall back to chunking on LLM error
    try:
        yaml_force = get("pipeline.summarize.force_chunk", False)
        force_chunk = (str(os.getenv("SUM_FORCE_CHUNK", "")).lower() in {"1", "true", "yes", "on"}) or bool(yaml_force)
    except Exception:
        force_chunk = False
    attempted_full = False
    if not force_chunk:
        if show_progress:
            print(f"[SUM] Pass1: full-text (size={len(full_text)}) …")
        try:
            t0 = time.time()
            p1 = _llm_pass1_extraction(full_text)
            packs.append(p1)
            # Optional raw saving
            if str(os.getenv("SUM_SAVE_RAW", "")).lower() in {"1", "true", "yes", "on"} and file_stem:
                (SUM_DIR / f"{file_stem}.pass1.json").write_text(json.dumps(p1, ensure_ascii=False, indent=2), encoding="utf-8")
            attempted_full = True
            if show_detail:
                print(f"[SUM]  • full-text took {time.time()-t0:.1f}s")
        except LLMError as exc:
            if show_progress:
                msg = str(exc)
                print(f"[SUM]  • full-text failed ({msg[:80]}…). Falling back to chunking.")
            attempted_full = False

    if not attempted_full:
        if len(full_text) <= chunk_size:
            if show_progress:
                print(f"[SUM] Pass1: 1 chunk (size={len(full_text)})")
            t0 = time.time()
            pck = _llm_pass1_extraction(full_text)
            packs.append(pck)
            if str(os.getenv("SUM_SAVE_RAW", "")).lower() in {"1", "true", "yes", "on"} and file_stem and not attempted_full:
                (SUM_DIR / f"{file_stem}.pass1.json").write_text(json.dumps(pck, ensure_ascii=False, indent=2), encoding="utf-8")
            if show_detail:
                print(f"[SUM]  • chunk 1 took {time.time()-t0:.1f}s")
        else:
            n_chunks = (len(full_text) + chunk_size - 1) // chunk_size
            if show_progress:
                print(f"[SUM] Pass1: {n_chunks} chunks (chunk_size={chunk_size}, total={len(full_text)})")
            idx = 0
            for i in range(0, len(full_text), chunk_size):
                idx += 1
                if show_progress:
                    start = i
                    end = min(i + chunk_size, len(full_text))
                    print(f"[SUM]  • chunk {idx}/{n_chunks} ({start}:{end}) …")
                t0 = time.time()
                ch = _llm_pass1_extraction(full_text[i : i + chunk_size])
                packs.append(ch)
                if str(os.getenv("SUM_SAVE_RAW", "")).lower() in {"1", "true", "yes", "on"} and file_stem:
                    (SUM_DIR / f"{file_stem}.pass1.{idx}.json").write_text(json.dumps(ch, ensure_ascii=False, indent=2), encoding="utf-8")
                if show_detail:
                    print(f"[SUM]    ⤷ chunk {idx} took {time.time()-t0:.1f}s")
    # Merge packs (LLM)
    if show_progress:
        print("[SUM] Merge extraction packs …")
    t0 = time.time()
    merged = _llm_merge_packs(packs)
    save_raw = False
    try:
        save_raw = bool(get("pipeline.summarize.save_raw", False)) or (str(os.getenv("SUM_SAVE_RAW", "")).lower() in {"1", "true", "yes", "on"})
    except Exception:
        save_raw = (str(os.getenv("SUM_SAVE_RAW", "")).lower() in {"1", "true", "yes", "on"})
    if save_raw and file_stem:
        (SUM_DIR / f"{file_stem}.merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    if show_detail:
        meta = merged.get("meta", {}) if isinstance(merged.get("meta"), dict) else {}
        meta_present = [k for k in ("title","venue","year","doi","url") if meta.get(k)]
        print(f"[SUM]  • merge took {time.time()-t0:.1f}s; meta={meta_present} datasets={len(merged.get('datasets',[]) or [])} "
              f"metrics={len(merged.get('metrics',[]) or [])} results_numbers={len(merged.get('results_numbers',[]) or [])}")
    # Pass 2: gap fill
    if show_progress:
        print("[SUM] Gap‑fill missing fields …")
    t0 = time.time()
    merged2 = _llm_pass2_gap_fill(full_text, merged)
    if save_raw and file_stem:
        (SUM_DIR / f"{file_stem}.gapfill.json").write_text(json.dumps(merged2, ensure_ascii=False, indent=2), encoding="utf-8")
    if show_detail:
        # crude diff: count fields newly filled in meta and list counts that increased
        filled = []
        for k in ("title","venue","year","doi","url"):
            if (merged.get("meta",{}).get(k) in (None, "")) and (merged2.get("meta",{}).get(k)):
                filled.append(k)
        def _len(x):
            return len(x or []) if isinstance(x, list) else 0
        diffs = []
        for k in ("datasets","metrics","results_numbers","results","ablations","baselines","novelty_claims"):
            a, b = _len(merged.get(k)), _len(merged2.get(k))
            if b > a:
                diffs.append(f"{k}+{b-a}")
        print(f"[SUM]  • gap‑fill took {time.time()-t0:.1f}s; meta_filled={filled or 'none'}; list_increases={', '.join(diffs) or 'none'}")

    # Optional targeted meta/resources fill if still missing
    missing_meta = []
    mp = merged2.get("meta", {}) if isinstance(merged2.get("meta"), dict) else {}
    for k in ("title","venue","year","doi","url"):
        if not _as_str(mp.get(k)):
            missing_meta.append(k)
    res = merged2.get("resources", {}) if isinstance(merged2.get("resources"), dict) else {}
    rep = merged2.get("reproducibility", {}) if isinstance(merged2.get("reproducibility"), dict) else {}
    if missing_meta or not _as_str(res.get("code_url")) or not _as_str(res.get("data_url")):
        if show_progress:
            print("[SUM] Targeted meta/resources fill …")
        try:
            m3 = _llm_meta_fill(full_text, merged2)
            merged2 = m3
            if save_raw and file_stem:
                (SUM_DIR / f"{file_stem}.meta_fill.json").write_text(json.dumps(m3, ensure_ascii=False, indent=2), encoding="utf-8")
        except LLMError:
            if show_detail:
                print("[SUM]  • meta/resources fill skipped due to LLM error.")
    # Pass 3: finalize to target schema
    if show_progress:
        print("[SUM] Finalize to target schema …")
    t0 = time.time()
    js = _llm_pass3_finalize_schema(merged2)
    if save_raw and file_stem:
        (SUM_DIR / f"{file_stem}.final_raw.json").write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    if show_detail:
        print(f"[SUM]  • finalize took {time.time()-t0:.1f}s")

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
        mset = {str(m).strip().lower() for m in metrics}
        out = list(metrics)
        if "accuracy" not in mset:
            out.append("accuracy")
        # AUC with explicit mode
        auc_modes = [m for m in metrics if str(m).strip().lower().startswith("auc")]
        if not auc_modes:
            # infer from results_numbers if any auc row exists
            has_auc_rn = any(str(r.get("metric") or "").lower().startswith("auc") for r in (rn or []) if isinstance(r, dict))
            out.append("auc (unspecified)" if has_auc_rn else "auc (unspecified)")
        else:
            # ensure mode tag present
            fixed = []
            for m in auc_modes:
                s = str(m)
                if "(" not in s:
                    fixed.append("auc (unspecified)")
            out = [mm for mm in out if str(mm).strip().lower() not in {str(x).strip().lower() for x in auc_modes}] + fixed
        # precision/recall/f1 with averaging tag (best-effort)
        if not any(str(m).strip().lower().startswith("precision") for m in out):
            out.append("precision (unspecified)")
        if not any(str(m).strip().lower().startswith("recall") for m in out):
            out.append("recall (unspecified)")
        if not any(str(m).strip().lower().startswith("f1") for m in out):
            out.append("f1 (unspecified)")
        if not any(str(m).strip().lower() == "confusion_matrix" for m in out):
            out.append("confusion_matrix")
        return out

    # Toggle via YAML/env if needed
    try:
        enforce_required = bool(get("pipeline.summarize.enforce_required", True))
        if str(os.getenv("SUM_ENFORCE", "")).lower() in {"0", "false", "no", "off"}:
            enforce_required = False
        if str(os.getenv("SUM_ENFORCE", "")).lower() in {"1", "true", "yes", "on"}:
            enforce_required = True
    except Exception:
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

        # augmentations presence: if none present in either, add a generic placeholder
        has_aug = any(str(x).strip().lower().startswith("aug:") for x in js["training_procedure"] + _ensure_list(js.get("methods")))
        if not has_aug:
            js["training_procedure"].append("aug: ")

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

    # Completeness score (light)
    checks = [bool(out.get("title")), bool(out.get("problem")), bool(out.get("methods")), bool(out.get("datasets")), bool(out.get("results"))]
    try:
        out["_completeness"] = round(sum(1 for c in checks if c) / len(checks), 3)
    except Exception:
        out["_completeness"] = 0.0

    if show_detail:
        print(
            "[SUM] Final stats: "
            f"title='{out.get('title','')[:60]}', venue='{out.get('venue','')}', year='{out.get('year','')}', doi_present={bool(out.get('doi'))}, "
            f"results_numbers={len(out.get('results_numbers',[]))}, dataset_details={len(out.get('dataset_details',[]))}, completeness={out.get('_completeness',0.0)}"
        )
    return out


def criticize_paper(summary: Dict[str, Any]) -> Dict[str, Any]:
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
    profile = get("pipeline.summarize.llm", None)
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0, profile=profile)
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


def main() -> None:
    _ensure_dirs()
    if not PDF_DIR.exists():
        print(f"[ERR] PDF dir not found: {PDF_DIR}")
        return

    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf") if p.is_file()])
    if not pdfs:
        print("[INFO] No PDFs found to process.")
        return

    num_summaries = 0

    for i, pdf in enumerate(pdfs, start=1):
        title_hint = pdf.stem.replace("_", " ")
        out_path = SUM_DIR / f"{pdf.stem}.json"
        if out_path.exists():
            print(f"[SKIP] Summary exists: {out_path.name}")
            continue

        print(f"[PDF {i}/{len(pdfs)}] Extracting: {pdf.name}")
        try:
            # YAML overrides take precedence; env can override YAML
            cfg_max_pages = get("pipeline.summarize.max_pages", None)
            cfg_max_chars = get("pipeline.summarize.max_chars", None)
            # Default to 0 (no limit) unless explicitly configured
            default_pages = "0" if cfg_max_pages is None else str(cfg_max_pages)
            default_chars = "0" if cfg_max_chars is None else str(cfg_max_chars)
            max_pages = int(os.getenv("SUM_MAX_PAGES", default_pages) or default_pages)
            max_chars = int(os.getenv("SUM_MAX_CHARS", default_chars) or default_chars)
            text = extract_text_from_pdf(pdf, max_pages=max_pages, max_chars=max_chars)
        except Exception as exc:
            print(f"[SKIP] Failed to extract text: {pdf.name} :: {exc}")
            continue

        try:
            summ = summarize_paper(text, title_hint=title_hint, file_stem=pdf.stem)
        except LLMError as exc:
            print(f"[SKIP] LLM summarize failed: {pdf.name} :: {exc}")
            continue

        try:
            crit = criticize_paper(summ)
        except LLMError as exc:
            print(f"[WARN] LLM critic failed: {pdf.name} :: {exc}")
            crit = {
                "novelty_strength": "",
                "possible_prior_overlap": [],
                "weaknesses": [],
                "followup_experiments": [],
            }

        # PDF hash for caching/regression
        record = {"summary": summ, "critic": crit}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {out_path}")
        num_summaries += 1

    if num_summaries == 0:
        print("[INFO] No new summaries produced.")
    else:
        print(f"[DONE] Wrote {num_summaries} summaries to {SUM_DIR}")


if __name__ == "__main__":
    main()
