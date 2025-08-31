import os
import json
import pathlib
from typing import Dict, Any, List
from dotenv import load_dotenv

from pdf_utils import extract_text_from_pdf
from llm_utils import chat_json, LLMError


load_dotenv()

DATA_DIR = pathlib.Path("data")
SUM_DIR = DATA_DIR / "summaries"
REPORT_PATH = DATA_DIR / "novelty_report.json"
PDF_DIR = pathlib.Path("pdfs")


def _ensure_dirs() -> None:
    SUM_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def summarize_paper(text: str, title_hint: str = "") -> Dict[str, Any]:
    system = (
        "You are an expert research analyst. Read the provided paper text and return a JSON object "
        "summarizing key details. Keep answers concise and extract only what is present."
    )
    user_payload = {
        "title_hint": title_hint,
        "paper_text": _truncate(text, 40000),
        "output_schema": {
            "title": "string",
            "problem": "string",
            "methods": ["string"],
            "datasets": ["string"],
            "results": ["string"],
            "novelty_claims": ["string"],
            "limitations": ["string"],
            "keywords": ["string"],
            "confidence": "0.0-1.0 float as string",
        },
        "instructions": "Be faithful to the text. If a field is unknown, return an empty list or empty string."
    }
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.0)
    # Normalize shapes
    def _as_list(x):
        if isinstance(x, list):
            return [str(i) for i in x]
        if not x:
            return []
        return [str(x)]

    return {
        "title": str(js.get("title") or title_hint or ""),
        "problem": str(js.get("problem") or ""),
        "methods": _as_list(js.get("methods")),
        "datasets": _as_list(js.get("datasets")),
        "results": _as_list(js.get("results")),
        "novelty_claims": _as_list(js.get("novelty_claims")),
        "limitations": _as_list(js.get("limitations")),
        "keywords": _as_list(js.get("keywords")),
        "confidence": str(js.get("confidence") or "")
    }


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


def group_novelty_and_ideas(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Build compact input
    items = []
    for s in summaries:
        items.append({
            "title": _truncate(s.get("title") or "", 120),
            "novelty_claims": s.get("novelty_claims") or [],
            "methods": s.get("methods") or [],
            "limitations": s.get("limitations") or [],
        })
    system = (
        "You are a panel moderator coordinating multiple experts. Given a list of papers with their novelty claims, "
        "cluster the claims into 3-6 themes, summarize each theme, name representative papers, and then propose 5-8 "
        "new research ideas that combine or extend these themes. Return JSON."
    )
    user_payload = {
        "papers": items,
        "output_schema": {
            "themes": [
                {
                    "name": "string",
                    "summary": "string",
                    "representative_papers": ["string"],
                }
            ],
            "new_ideas": ["string"],
        }
    }
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.2)
    # light shape enforcement
    themes = js.get("themes") or []
    if not isinstance(themes, list):
        themes = []
    new_ideas = js.get("new_ideas") or []
    if not isinstance(new_ideas, list):
        new_ideas = []
    return {"themes": themes, "new_ideas": new_ideas}


def main() -> None:
    _ensure_dirs()
    if not PDF_DIR.exists():
        print(f"[ERR] PDF dir not found: {PDF_DIR}")
        return

    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf") if p.is_file()])
    if not pdfs:
        print("[INFO] No PDFs found to process.")
        return

    summaries: List[Dict[str, Any]] = []
    num_pdfs = len(pdfs)
    num_summaries = 0

    for i, pdf in enumerate(pdfs, start=1):
        title_hint = pdf.stem.replace("_", " ")
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
        summaries.append(record)
        num_summaries += 1

        out_path = SUM_DIR / f"{pdf.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {out_path}")

    if not summaries:
        print("[INFO] No summaries produced. Exiting.")
        return

    try:
        panel = group_novelty_and_ideas([s["summary"] for s in summaries])
    except LLMError as exc:
        print(f"[ERR] LLM group discussion failed: {exc}")
        panel = {"themes": [], "new_ideas": []}

    final = {
        "num_papers": len(summaries),
        "themes": panel.get("themes", []),
        "new_ideas": panel.get("new_ideas", []),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote novelty report: {REPORT_PATH}")

    # Write light literature stats
    try:
        stats = {
            "num_pdfs": num_pdfs,
            "num_summaries": num_summaries,
            "num_themes": len(final.get("themes") or []),
            "num_new_ideas": len(final.get("new_ideas") or []),
        }
        (DATA_DIR / "lit_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
