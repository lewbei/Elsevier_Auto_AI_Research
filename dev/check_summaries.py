import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SUM_DIR = ROOT / "data" / "summaries"


def _lower_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    return []


def _get_summary(js: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(js.get("summary"), dict):
        return js["summary"]
    return js


def _has_key_prefix(items: List[str], key_prefixes: List[str]) -> bool:
    prefixes = tuple(k.lower() for k in key_prefixes)
    for s in items:
        ss = str(s).strip().lower()
        if any(ss.startswith(k) for k in prefixes):
            return True
    return False


def _find_key_value(items: List[str], key: str) -> Tuple[bool, str]:
    key_l = key.lower()
    for s in items:
        ss = str(s).strip()
        if ss.lower().startswith(key_l):
            return True, ss
    return False, ""


def _check_meta(s: Dict[str, Any], issues: List[str]) -> None:
    title = str(s.get("title") or "").strip()
    year = str(s.get("year") or "").strip()
    doi = str(s.get("doi") or "").strip()
    url = str(s.get("url") or "").strip()
    authors = _lower_list(s.get("authors"))
    if not title:
        issues.append("meta: missing title")
    if not year:
        issues.append("meta: missing year")
    if not (doi or url):
        issues.append("meta: missing doi/url (at least one required)")
    if not authors:
        issues.append("meta: missing authors[]")


def _check_dataset(s: Dict[str, Any], issues: List[str]) -> None:
    datasets = _lower_list(s.get("datasets"))
    dds = s.get("dataset_details") or []
    if not datasets:
        issues.append("dataset: datasets[] is empty")
    if not isinstance(dds, list) or not dds:
        issues.append("dataset: dataset_details[] missing")
        return
    # Basic checks per entry
    num_pat = re.compile(r"\d+")
    split_pat = re.compile(r"train\s*\d+|test\s*\d+|val(?!ue)\s*\d+", re.IGNORECASE)
    for i, dd in enumerate(dds, 1):
        if not isinstance(dd, dict):
            issues.append(f"dataset_details[{i}]: not an object")
            continue
        name = str(dd.get("name") or "").strip()
        size = str(dd.get("size") or "").strip()
        splits = str(dd.get("splits") or "").strip()
        if not name:
            issues.append(f"dataset_details[{i}]: missing name")
        if not size or not num_pat.search(size):
            issues.append(f"dataset_details[{i}]: size should include numeric total (e.g., '331')")
        if not splits or (not split_pat.search(splits)):
            issues.append(f"dataset_details[{i}]: splits should include Train/Test counts (e.g., 'Train 264, Test 67')")


def _check_training_and_hparams(s: Dict[str, Any], issues: List[str]) -> None:
    tp = _lower_list(s.get("training_procedure"))
    hp = _lower_list(s.get("hyperparameters"))
    # Required knobs in training_procedure
    required_tp = ["epochs:", "batch_size:", "optimizer:", "learning_rate:", "lr:"]
    # Satisfy lr by either learning_rate or lr
    if not _has_key_prefix(tp, ["epochs:"]):
        issues.append("training_procedure: missing 'epochs:'")
    if not _has_key_prefix(tp, ["batch_size:"]):
        issues.append("training_procedure: missing 'batch_size:' (include as empty if unknown)")
    if not _has_key_prefix(tp, ["optimizer:"]):
        issues.append("training_procedure: missing 'optimizer:'")
    if not (_has_key_prefix(tp, ["learning_rate:"]) or _has_key_prefix(tp, ["lr:"])):
        issues.append("training_procedure: missing 'learning_rate:' or 'lr:'")
    # Augmentation presence (names-only acceptable)
    if not any(str(x).strip().lower().startswith("aug:") for x in tp + _lower_list(s.get("methods"))):
        issues.append("augmentation: expected at least one 'aug: ...' entry in training_procedure[] or methods[]")
    # Hyperparameters mirror (check presence of a few key mirrors)
    for key in ["batch_size:", "optimizer:", "lr:", "epochs:", "weight_decay:"]:
        if not _has_key_prefix(hp, [key]):
            issues.append(f"hyperparameters: missing '{key}' entry (include empty if unknown)")


def _check_metrics_and_results(s: Dict[str, Any], issues: List[str]) -> None:
    metrics = [str(m).lower() for m in _lower_list(s.get("metrics"))]
    if "accuracy" not in metrics:
        issues.append("metrics: expected 'accuracy'")
    # AUC presence with averaging hint
    auc_entries = [m for m in metrics if m.startswith("auc")]
    if not auc_entries:
        issues.append("metrics: expected 'auc (macro|micro|ovr|unspecified)'")
    else:
        if not any("(" in m and ")" in m for m in auc_entries):
            issues.append("metrics: 'auc' lacks averaging mode (add 'auc (unspecified)' if not given)")
    # Recall/F1 with averaging where possible
    if not any(m.startswith("recall") for m in metrics):
        if "sensitivity" in metrics:
            issues.append("metrics: 'recall' missing (has 'sensitivity'); prefer 'recall (macro|weighted)' entry")
        else:
            issues.append("metrics: expected 'recall (macro|weighted)'")
    if not any(m.startswith("f1") for m in metrics):
        issues.append("metrics: expected 'f1 (macro|weighted)'")
    if "confusion_matrix" not in metrics:
        issues.append("metrics: expected 'confusion_matrix'")

    # results_numbers coverage: at least one test accuracy and one test AUC
    rn = s.get("results_numbers") or []
    if not isinstance(rn, list) or not rn:
        issues.append("results_numbers: missing entries")
        return
    test_acc = False
    test_auc = False
    for row in rn:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "").lower()
        split = str(row.get("split") or "").lower()
        if (metric == "test_accuracy") or (metric == "accuracy" and split.startswith("test")):
            test_acc = True
        if (metric.startswith("auc") and split.startswith("test")):
            test_auc = True
    if not test_acc:
        issues.append("results_numbers: missing test accuracy row")
    if not test_auc:
        issues.append("results_numbers: missing test AUC row")


def _check_repro(s: Dict[str, Any], issues: List[str]) -> None:
    rep = s.get("reproducibility") or {}
    if not isinstance(rep, dict):
        issues.append("reproducibility: not an object")
        return
    code_av = str(rep.get("code_available") or "").strip().lower()
    data_av = str(rep.get("data_available") or "").strip().lower()
    if code_av not in {"yes", "no", ""}:
        issues.append("reproducibility: code_available should be 'yes' or 'no' (or empty if unknown)")
    if data_av and data_av not in {"public", "restricted", "no"}:
        issues.append("reproducibility: data_available should be one of public|restricted|no (or empty)")


def _gate_complete(s: Dict[str, Any]) -> Tuple[bool, List[str]]:
    gate_issues: List[str] = []
    # Gate checks
    if not str(s.get("title") or "").strip():
        gate_issues.append("gate: title empty")
    if not str(s.get("year") or "").strip():
        gate_issues.append("gate: year empty")
    if not (str(s.get("doi") or "").strip() or str(s.get("url") or "").strip()):
        gate_issues.append("gate: doi/url both empty")
    if not _lower_list(s.get("datasets")):
        gate_issues.append("gate: datasets[] empty")
    dds = s.get("dataset_details") or []
    if not isinstance(dds, list) or not dds:
        gate_issues.append("gate: dataset_details[] missing")
    else:
        if not str(dds[0].get("size") or "").strip():
            gate_issues.append("gate: dataset_details.size missing")
        if not str(dds[0].get("splits") or "").strip():
            gate_issues.append("gate: dataset_details.splits missing")
    # training knobs
    tp = _lower_list(s.get("training_procedure"))
    for key in ["epochs:", "batch_size:", "optimizer:"]:
        if not _has_key_prefix(tp, [key]):
            gate_issues.append(f"gate: training_procedure missing '{key}'")
    if not (_has_key_prefix(tp, ["learning_rate:"]) or _has_key_prefix(tp, ["lr:"])):
        gate_issues.append("gate: training_procedure missing 'lr:'")
    # metrics
    metrics = [str(m).lower() for m in _lower_list(s.get("metrics"))]
    if "accuracy" not in metrics:
        gate_issues.append("gate: metrics missing accuracy")
    if not any(m.startswith("auc") for m in metrics):
        gate_issues.append("gate: metrics missing auc")
    # results_numbers
    rn = s.get("results_numbers") or []
    if not isinstance(rn, list) or not rn:
        gate_issues.append("gate: results_numbers empty")
    else:
        ok_acc = any((str(r.get("metric") or "").lower() in {"test_accuracy"} or (str(r.get("metric") or "").lower() == "accuracy" and str(r.get("split") or "").lower().startswith("test"))) for r in rn if isinstance(r, dict))
        ok_auc = any((str(r.get("metric") or "").lower().startswith("auc") and str(r.get("split") or "").lower().startswith("test")) for r in rn if isinstance(r, dict))
        if not ok_acc:
            gate_issues.append("gate: results_numbers missing test accuracy")
        if not ok_auc:
            gate_issues.append("gate: results_numbers missing test AUC")
    # limitations non-empty
    if not _lower_list(s.get("limitations")):
        gate_issues.append("gate: limitations[] empty")
    return (len(gate_issues) == 0), gate_issues


def audit_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"file": str(path), "error": f"invalid json: {e}", "issues": [], "gate_ok": False, "gate_issues": ["invalid json"]}
    s = _get_summary(data)
    issues: List[str] = []
    _check_meta(s, issues)
    _check_dataset(s, issues)
    _check_training_and_hparams(s, issues)
    _check_metrics_and_results(s, issues)
    _check_repro(s, issues)
    gate_ok, gate_issues = _gate_complete(s)
    return {
        "file": str(path),
        "issues": issues,
        "gate_ok": gate_ok,
        "gate_issues": gate_issues,
    }


def main() -> int:
    if not SUM_DIR.exists():
        print(f"[ERR] summaries dir not found: {SUM_DIR}")
        return 2
    files = sorted([p for p in SUM_DIR.glob("*.json") if p.is_file()])
    if not files:
        print("[INFO] No summary JSON files found.")
        return 0
    exit_code = 0
    for p in files:
        rep = audit_file(p)
        print(f"\n=== {p.name} ===")
        if rep.get("error"):
            print(f"  ERROR: {rep['error']}")
            exit_code = 1
            continue
        for issue in rep["issues"]:
            print(f"  - {issue}")
        if not rep["issues"]:
            print("  (no detailed issues)")
        if rep["gate_ok"]:
            print("  Gate: PASS")
        else:
            print("  Gate: FAIL")
            for gi in rep["gate_issues"]:
                print(f"    * {gi}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

