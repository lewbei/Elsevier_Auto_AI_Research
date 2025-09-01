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
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)


def _llm_pass2_gap_fill(text: str, pack: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a meticulous auditor. Given an extraction pack and the same paper text, fill only missing/uncertain fields. Do not change already-filled correct fields. "
        "If still not present in the text, keep blank and record in coverage_report.missing_after_second_pass. JSON only."
    )
    user = {"paper_text": text, "extraction_pack": pack}
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)


def _llm_merge_packs(packs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(packs) == 1:
        return packs[0]
    system = (
        "You are a careful merger. Given multiple extraction packs of the same paper chunks, merge them into a single pack with the same shape. "
        "Keep filled fields, dedupe lists, and preserve coverage_report notes. JSON only."
    )
    user = {"packs": packs}
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)


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
        ],
    }
    return chat_json(system, json.dumps(user, ensure_ascii=False), temperature=0.0)


def summarize_paper(text: str, title_hint: str = "") -> Dict[str, Any]:
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
            "If dates/venues/DOIs are missing, leave blank."
        ),
    }
    # All‑LLM 3‑pass flow with chunking support
    # Pass 1: extraction per chunk
    full_text = text
    packs: List[Dict[str, Any]] = []
    # If pass1_chunk <= 0, treat as "no chunking" (use full text in one call)
    try:
        cfg_chunk = get("pipeline.summarize.pass1_chunk", None)
        chunk_size = int(os.getenv("SUM_PASS1_CHUNK", str(cfg_chunk if isinstance(cfg_chunk, int) else "20000")) or 20000)
    except Exception:
        chunk_size = 20000
    if chunk_size <= 0:
        chunk_size = len(full_text) or 1
    show_progress = str(os.getenv("SUM_PROGRESS", "1")).lower() in {"1", "true", "yes", "on"}
    show_detail = str(os.getenv("SUM_DETAIL", "")).lower() in {"1", "true", "yes", "on"}
    if len(full_text) <= chunk_size:
        if show_progress:
            print(f"[SUM] Pass1: 1 chunk (size={len(full_text)})")
        t0 = time.time()
        packs.append(_llm_pass1_extraction(full_text))
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
            packs.append(_llm_pass1_extraction(full_text[i : i + chunk_size]))
            if show_detail:
                print(f"[SUM]    ⤷ chunk {idx} took {time.time()-t0:.1f}s")
    # Merge packs (LLM)
    if show_progress:
        print("[SUM] Merge extraction packs …")
    t0 = time.time()
    merged = _llm_merge_packs(packs)
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
    # Pass 3: finalize to target schema
    if show_progress:
        print("[SUM] Finalize to target schema …")
    t0 = time.time()
    js = _llm_pass3_finalize_schema(merged2)
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
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0)
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
            max_pages = int(os.getenv("SUM_MAX_PAGES", str(cfg_max_pages if isinstance(cfg_max_pages, int) else "12")) or 12)
            max_chars = int(os.getenv("SUM_MAX_CHARS", str(cfg_max_chars if isinstance(cfg_max_chars, int) else "40000")) or 40000)
            text = extract_text_from_pdf(pdf, max_pages=max_pages, max_chars=max_chars)
        except Exception as exc:
            print(f"[SKIP] Failed to extract text: {pdf.name} :: {exc}")
            continue

        try:
            summ = summarize_paper(text, title_hint=title_hint)
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
