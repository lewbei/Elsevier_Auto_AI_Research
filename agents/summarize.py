import os
import json
import pathlib
from typing import Dict, Any, List
from dotenv import load_dotenv

from utils.pdf_utils import extract_text_from_pdf
from utils.llm_utils import chat_json, LLMError


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
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0)
    # Normalize shapes
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]

    # Build a detailed, backward-compatible summary object
    out: Dict[str, Any] = {
        # Core fields used elsewhere in the pipeline
        "title": str(js.get("title") or title_hint or ""),
        "problem": str(js.get("problem") or ""),
        "methods": _as_list(js.get("methods")),
        "datasets": _as_list(js.get("datasets")),
        "results": _as_list(js.get("results")),
        "novelty_claims": _as_list(js.get("novelty_claims")),
        "limitations": _as_list(js.get("limitations")),
        "keywords": _as_list(js.get("keywords")),
        "confidence": str(js.get("confidence") or ""),
    }

    # Additional rich fields (best-effort normalization)
    def _as_dict(x):
        return x if isinstance(x, dict) else {}

    for k in (
        "authors", "venue", "year", "doi", "url", "background", "research_questions", "hypotheses",
        "tasks", "architecture", "training_procedure", "hyperparameters", "metrics", "ablations",
        "baselines", "novelty_type", "prior_work_overlap", "failure_modes", "ethical",
        "threats_to_validity", "conclusions", "future_work",
    ):
        v = js.get(k)
        out[k] = _as_list(v)

    # Structured lists/dicts
    dd = js.get("dataset_details")
    out["dataset_details"] = dd if isinstance(dd, list) else []
    rn = js.get("results_numbers")
    out["results_numbers"] = rn if isinstance(rn, list) else []
    out["reproducibility"] = _as_dict(js.get("reproducibility"))
    out["resources"] = _as_dict(js.get("resources"))

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
            text = extract_text_from_pdf(pdf, max_pages=12, max_chars=40000)
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
