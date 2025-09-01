import json
import sys
from pathlib import Path


def _is_empty_scalar(v):
    return v is None or (isinstance(v, str) and v.strip() == "")


def _scan_summary(summary: dict) -> list[str]:
    missing: list[str] = []

    # Top-level scalar fields
    for key in (
        "title",
        "problem",
        "venue",
        "year",
        "doi",
        "url",
        "confidence",
    ):
        if _is_empty_scalar(summary.get(key)):
            missing.append(f"summary.{key}")

    # Top-level list fields (missing if empty list)
    for key in (
        "methods",
        "datasets",
        "results",
        "novelty_claims",
        "limitations",
        "keywords",
        "authors",
        "background",
        "research_questions",
        "hypotheses",
        "tasks",
        "architecture",
        "training_procedure",
        "hyperparameters",
        "metrics",
        "ablations",
        "baselines",
        "novelty_type",
        "prior_work_overlap",
        "failure_modes",
        "ethical",
        "threats_to_validity",
        "conclusions",
        "future_work",
    ):
        v = summary.get(key)
        if not isinstance(v, list) or len(v) == 0:
            missing.append(f"summary.{key}")

    # dataset_details entries
    for i, dd in enumerate(summary.get("dataset_details", []) or []):
        if not isinstance(dd, dict):
            continue
        for fld in ("name", "size", "splits", "preprocessing", "notes"):
            if _is_empty_scalar(dd.get(fld)):
                missing.append(f"summary.dataset_details[{i}].{fld}")

    # results_numbers entries
    for i, rn in enumerate(summary.get("results_numbers", []) or []):
        if not isinstance(rn, dict):
            continue
        for fld in ("metric", "value", "dataset", "split", "baseline", "improvement"):
            if _is_empty_scalar(rn.get(fld)):
                missing.append(f"summary.results_numbers[{i}].{fld}")

    # reproducibility/resources
    rep = summary.get("reproducibility", {}) or {}
    if _is_empty_scalar(rep.get("code_available")):
        missing.append("summary.reproducibility.code_available")
    # data_available can be blank if unspecified; still report if empty
    if _is_empty_scalar(rep.get("data_available")):
        missing.append("summary.reproducibility.data_available")

    res = summary.get("resources", {}) or {}
    for fld in ("code_url", "data_url", "model_url"):
        if _is_empty_scalar(res.get(fld)):
            missing.append(f"summary.resources.{fld}")

    return missing


def main():
    if len(sys.argv) < 2:
        print("Usage: python dev/gap_report.py path/to/summary.json")
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or "summary" not in doc:
        print("Invalid file format: expected top-level 'summary' field")
        sys.exit(1)

    summary = doc.get("summary", {}) or {}
    missing = _scan_summary(summary)

    print(f"GAP REPORT for {path.name}")
    print("- Total missing fields:", len(missing))
    for m in missing:
        print("  •", m)

    # Quick guidance for common gaps
    hints = []
    if any(m.startswith("summary.resources.code_url") for m in missing):
        hints.append("No code_url detected: if the PDF doesn’t include a repo URL, this remains blank by design.")
    if any("dataset_details" in m for m in missing):
        hints.append("Dataset details missing: ensure the paper text specifies sizes/splits; external lookup is intentionally not used.")
    if any("results_numbers" in m for m in missing):
        hints.append("Numeric results incomplete: the paper may not state baselines/splits; improvement requires both baseline and new values.")

    if hints:
        print("\nNotes:")
        for h in hints:
            print("-", h)


if __name__ == "__main__":
    main()

