import json
import os
import pathlib
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from utils.llm_utils import chat_json_cached, LLMError
from lab.codegen_utils import write_generated_aug, write_generated_head
from lab.experiment_runner import run_experiment
from lab.logging_utils import capture_env, try_mlflow_log
from lab.report_html import write_dashboard
from lab.mutations import propose_mutations  # type: ignore
from lab.plot_utils import maybe_save_accuracy_bar  # type: ignore
from lab.config import get
from lab.config import dataset_path_for


load_dotenv()

RUNS_DIR = pathlib.Path("runs")
EXPERIMENTS_DIR = pathlib.Path("experiments")
NOVELTY_REPORT = pathlib.Path("data/novelty_report.json")


def _ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def propose_baseline_and_novelty(novelty: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    topic = str(get("project.goal", "your task") or "your task")
    system = (
        "You are an AI scientist. Propose two minimal experiment specs for " + topic + ": "
        "(1) a baseline using standard components, (2) a novelty variant incorporating a single, concrete novelty; "
        "and (3) an ablation that removes the novelty component from (2) only. Keep runnable on CPU in <= 1 epoch and small steps. Return JSON."
    )
    user_payload = {
        "novelty_report": novelty,
        "constraints": {
            "dataset": "ImageFolder under data/isic with train/ and val/",
            "compute": "CPU or single GPU, <= 1 epoch, limit steps",
            "framework": "PyTorch if available, else allow stub",
        },
        "output_schema": {
            "baseline": {
                "title": "string",
                "dataset_path": "data/isic",
                "input_size": 224,
                "model": "resnet18",
                "epochs": 1,
                "batch_size": 16,
                "lr": 0.001,
                "max_train_steps": 50,
                "seed": 42,
                "novelty_component": {"description": "baseline", "enabled": False}
            },
            "novelty": {
                "title": "string",
                "dataset_path": "data/isic",
                "input_size": 224,
                "model": "resnet18",
                "epochs": 1,
                "batch_size": 16,
                "lr": 0.001,
                "max_train_steps": 50,
                "seed": 42,
                "novelty_component": {"description": "string", "enabled": True}
            },
            "ablation": {
                "title": "string",
                "dataset_path": "data/isic",
                "input_size": 224,
                "model": "resnet18",
                "epochs": 1,
                "batch_size": 16,
                "lr": 0.001,
                "max_train_steps": 50,
                "seed": 42,
                "novelty_component": {"description": "same as novelty but disabled", "enabled": False}
            }
        }
    }
    js = chat_json_cached(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.2)
    baseline = js.get("baseline") or {}
    novelty_spec = js.get("novelty") or {}
    ablation = js.get("ablation") or {}

    # Fill defaults
    def fill(spec: Dict[str, Any], title: str) -> Dict[str, Any]:
        return {
            "title": spec.get("title") or title,
            "dataset_path": spec.get("dataset_path") or "data/isic",
            "input_size": int(spec.get("input_size") or 224),
            "model": spec.get("model") or "resnet18",
            "epochs": int(spec.get("epochs") or 1),
            "batch_size": int(spec.get("batch_size") or 16),
            "lr": float(spec.get("lr") or 1e-3),
            "max_train_steps": int(spec.get("max_train_steps") or 50),
            "seed": int(spec.get("seed") or 42),
            "novelty_component": spec.get("novelty_component") or {"description": "baseline", "enabled": False},
        }

    b, n, a = fill(baseline, "Baseline"), fill(novelty_spec, "Novelty"), fill(ablation, "Ablation")
    n = _apply_novelty_desc_to_spec(n)
    return b, n, a


def _apply_novelty_desc_to_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight mapper: adjust spec fields based on novelty description keywords."""
    nc = spec.get("novelty_component") or {}
    desc = str(nc.get("description") or "").lower()
    if not desc:
        return spec
    # Enable generated head if description mentions head/classifier/dropout
    if any(k in desc for k in ["head", "classifier", "dropout"]):
        spec["use_generated_head"] = True
        # crude extraction of dropout like "dropout 0.3"
        import re
        m = re.search(r"dropout\s*([0-9]*\.?[0-9]+)", desc)
        if m:
            try:
                p = float(m.group(1))
                if 0.0 < p < 0.9:
                    spec["dropout_p"] = p
            except Exception:
                pass
    # Choose optimizer if specified
    if "sgd" in desc:
        spec["optimizer"] = "sgd"
    elif "adam" in desc:
        spec["optimizer"] = "adam"
    # Input size hints
    if "high-res" in desc or "larger input" in desc:
        spec["input_size"] = max(int(spec.get("input_size", 224)), 256)
    if "low-res" in desc or "smaller input" in desc:
        spec["input_size"] = min(int(spec.get("input_size", 224)), 192)
    return spec


def analyze_and_decide(results: List[Dict[str, Any]], target_delta: float = 0.005) -> Dict[str, Any]:
    acc = {r["name"]: r["result"]["metrics"].get("val_accuracy", 0.0) for r in results}
    baseline_acc = acc.get("baseline", 0.0)
    novelty_acc = acc.get("novelty", 0.0)
    ablation_acc = acc.get("ablation", 0.0)
    success = (novelty_acc - baseline_acc) >= target_delta and (novelty_acc - ablation_acc) >= 0.0
    return {
        "baseline_acc": baseline_acc,
        "novelty_acc": novelty_acc,
        "ablation_acc": ablation_acc,
        "target_delta": target_delta,
        "success": success,
    }


def verify_and_fix_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Simple verifier/normalizer that enforces ranges and required fields."""
    fixed = dict(spec)
    fixed.setdefault("dataset_path", dataset_path_for())
    # ranges
    def clamp(v, lo, hi, cast) -> Any:
        try:
            return max(lo, min(hi, cast(v)))
        except (ValueError, TypeError):
            return lo
    fixed["input_size"] = clamp(fixed.get("input_size", 224), 96, 512, int)
    fixed["epochs"] = clamp(fixed.get("epochs", 1), 1, 5, int)
    fixed["batch_size"] = clamp(fixed.get("batch_size", 16), 1, 128, int)
    fixed["lr"] = clamp(fixed.get("lr", 1e-3), 1e-5, 1e-1, float)
    fixed["max_train_steps"] = clamp(fixed.get("max_train_steps", 50), 10, 1000, int)
    fixed["seed"] = clamp(fixed.get("seed", 42), 1, 10_000, int)
    # novelty component presence
    nc = fixed.get("novelty_component") or {"description": "baseline", "enabled": False}
    if not isinstance(nc, dict):
        nc = {"description": str(nc), "enabled": False}
    if "enabled" not in nc:
        nc["enabled"] = False
    if "description" not in nc:
        nc["description"] = "baseline"
    fixed["novelty_component"] = nc
    # model string
    m = str(fixed.get("model") or "resnet18").lower()
    if m not in {"resnet18", "efficientnet_b0", "mobilenet_v3_small"}:
        m = "resnet18"
    fixed["model"] = m
    # title
    fixed["title"] = str(fixed.get("title") or "Experiment")
    return fixed


def engineer_refine_specs(
    prior_results: List[Dict[str, Any]],
    baseline: Dict[str, Any],
    novelty_spec: Dict[str, Any],
    ablation: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Use LLM to suggest targeted changes; then verify/normalize specs."""
    system = (
        "You are an Engineer agent. Given baseline/novelty/ablation specs and their metrics, propose small, safe refinements "
        "to improve novelty performance while keeping constraints (<=1 epoch, small steps). Only tweak lr (0.5x-2x), "
        "max_train_steps (+/- up to 50), input_size (keep 96-512), or enable/disable novelty. Return JSON with updated specs."
    )
    compact = []
    for r in prior_results:
        compact.append({
            "name": r["name"],
            "val_accuracy": r["result"].get("metrics", {}).get("val_accuracy", 0.0),
            "spec": {k: r["spec"].get(k) for k in ["lr", "max_train_steps", "input_size", "seed", "model", "novelty_component"]},
        })
    user_payload = {
        "prior": compact,
        "baseline": baseline,
        "novelty": novelty_spec,
        "ablation": ablation,
        "constraints": {
            "epochs": "<=1",
            "max_train_steps": "<=1000",
            "safe_changes": ["lr", "max_train_steps", "input_size", "novelty_component.enabled"],
        },
        "output_schema": {
            "baseline": {},
            "novelty": {},
            "ablation": {}
        }
    }
    js = chat_json_cached(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.2)
    b = verify_and_fix_spec(js.get("baseline") or baseline)
    n = verify_and_fix_spec(js.get("novelty") or novelty_spec)
    a = verify_and_fix_spec(js.get("ablation") or ablation)
    # Apply novelty description mapping to guide codegen/toggles
    n = _apply_novelty_desc_to_spec(n)
    # Ensure ablation describes same novelty but disabled
    a["novelty_component"] = {
        "description": n.get("novelty_component", {}).get("description", "baseline"),
        "enabled": False,
    }
    # Ensure novelty has novelty enabled
    n_nc = n.get("novelty_component", {})
    n["novelty_component"] = {"description": n_nc.get("description", "novelty"), "enabled": True}
    # Minimal codegen: write a generated augmentation module based on novelty description
    try:
        write_generated_aug(n["novelty_component"]["description"])  # writes lab/generated_aug.py
    except Exception:
        pass
    return b, n, a


def iterate(novelty: Dict[str, Any], max_iters: int = 2) -> None:
    _ensure_dirs()
    all_runs: List[Dict[str, Any]] = []
    goal_reached = False
    # Capture environment once for reproducibility
    try:
        env = capture_env()
        _write_json(RUNS_DIR / "env.json", env)
    except Exception:
        pass
    baseline = novelty_spec = ablation = None
    for it in range(1, max_iters + 1):
        if it == 1 or baseline is None:
            print(f"[ITER {it}] Proposing baseline/novelty/ablation specs…")
            try:
                baseline, novelty_spec, ablation = propose_baseline_and_novelty(novelty)
            except LLMError as exc:
                print(f"[ERR] LLM failed to propose specs: {exc}")
                break
        else:
            print(f"[ITER {it}] Refining specs based on prior results…")
            try:
                baseline, novelty_spec, ablation = engineer_refine_specs(iter_results, baseline, novelty_spec, ablation)
            except LLMError as exc:
                print(f"[WARN] Engineer failed to refine specs: {exc}. Reusing previous specs.")

        run_id = f"iter{it}_{_timestamp()}"
        _write_json(EXPERIMENTS_DIR / f"{run_id}_baseline.json", baseline)
        _write_json(EXPERIMENTS_DIR / f"{run_id}_novelty.json", novelty_spec)
        _write_json(EXPERIMENTS_DIR / f"{run_id}_ablation.json", ablation)

        # HITL confirmation gate for spec approval
        hitl = str(os.getenv("HITL_CONFIRM", "")).lower() in {"1", "true", "yes"}
        auto = str(os.getenv("HITL_AUTO_APPROVE", "")).lower() in {"1", "true", "yes"}
        if hitl and not auto:
            pending = {
                "iter": it,
                "run_id": run_id,
                "baseline": baseline,
                "novelty": novelty_spec,
                "ablation": ablation,
            }
            _write_json(EXPERIMENTS_DIR / f"pending_{run_id}.json", pending)
            _write_json(pathlib.Path("data/spec_pending.json"), pending)
            print(f"[PENDING] Specs written for approval: experiments/pending_{run_id}.json (and data/spec_pending.json). Set HITL_AUTO_APPROVE=1 to proceed.")
            return

        # Ensure generated modules exist for this iteration (safe defaults)
        try:
            desc = (novelty_spec.get("novelty_component", {}) or {}).get("description", "")
            write_generated_aug(desc or "jitter rotate")
        except Exception:
            pass
        try:
            write_generated_head()
        except Exception:
            pass

        # Breadth: base triplet plus a second model variant for each
        base_triplet = [("baseline", baseline), ("novelty", novelty_spec), ("ablation", ablation)]
        matrix: List[Tuple[str, Dict[str, Any]]] = []
        for name, spec in base_triplet:
            matrix.append((name, spec))
            spec2 = dict(spec)
            spec2["model"] = "mobilenet_v3_small"
            matrix.append((f"{name}_mbv3", spec2))

        # Modest extra variants for novelty (resnet18 only): +10% lr and +32 input size
        nov = dict(novelty_spec)
        lr = float(nov.get("lr", 1e-3))
        inp = int(nov.get("input_size", 224))
        nov_lr_up = dict(nov)
        nov_lr_up["lr"] = lr * 1.1
        nov_inp_up = dict(nov)
        nov_inp_up["input_size"] = inp + 32
        matrix.append(("novelty_lr_up", nov_lr_up))
        matrix.append(("novelty_inp_up", nov_inp_up))
        # Novelty with SGD optimizer
        nov_sgd = dict(nov)
        nov_sgd["optimizer"] = "sgd"
        matrix.append(("novelty_sgd", nov_sgd))
        # Novelty with generated head (higher dropout)
        nov_head = dict(nov)
        nov_head["use_generated_head"] = True
        nov_head["dropout_p"] = 0.3
        matrix.append(("novelty_dropout_high", nov_head))

        # Optional: beam mutations derived from novelty spec
        mutate_k = int(os.getenv("MUTATE_K", "0") or 0)
        if mutate_k > 0:
            muts = propose_mutations(nov, max_k=mutate_k)
            for idx, m in enumerate(muts, start=1):
                matrix.append((f"novelty_mut{idx}", m))

        # Repeats (baseline and novelty only)
        repeat_n = max(1, int(os.getenv("REPEAT_N", "1") or 1))
        if repeat_n > 1:
            extra: List[Tuple[str, Dict[str, Any]]] = []
            for base_name, base_spec in [("baseline", baseline), ("novelty", novelty_spec)]:
                for i in range(2, repeat_n + 1):
                    extra.append((f"{base_name}_rep{i}", dict(base_spec)))
            matrix.extend(extra)

        iter_results: List[Dict[str, Any]] = []
        start_ts = time.time()
        budget_sec = float(os.getenv("TIME_BUDGET_SEC", "0") or 0)
        parallel = str(os.getenv("PARALLEL_RUNS", "")).lower() in {"1", "true", "yes"}

        def _run_one(name_spec: Tuple[str, Dict[str, Any]]):
            name, spec = name_spec
            print(f"[RUN] {name} …")
            res = run_experiment(spec)
            return name, spec, res

        if parallel:
            max_workers = int(os.getenv("PARALLEL_WORKERS", "2") or 2)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_one, ns): ns[0] for ns in matrix}
                for fut in as_completed(futs):
                    name, spec, result = fut.result()
                    out = {"iter": it, "name": name, "spec": spec, "result": result, "run_id": run_id}
                    iter_results.append(out)
                    _write_json(RUNS_DIR / run_id / f"{name}_metrics.json", result)
                    _write_json(RUNS_DIR / run_id / f"{name}_spec.json", spec)
                    try_mlflow_log(run_name=f"{run_id}-{name}", params=spec, metrics=result.get("metrics", {}), tags={"iter": it})
                    if budget_sec and (time.time() - start_ts) > budget_sec:
                        print("[BUDGET] Time budget exceeded; stopping further runs in this iteration.")
                        break
        else:
            for name, spec in matrix:
                name2, spec2, result = _run_one((name, spec))
                out = {"iter": it, "name": name2, "spec": spec2, "result": result, "run_id": run_id}
                iter_results.append(out)
                _write_json(RUNS_DIR / run_id / f"{name2}_metrics.json", result)
                _write_json(RUNS_DIR / run_id / f"{name2}_spec.json", spec2)
                try_mlflow_log(run_name=f"{run_id}-{name2}", params=spec2, metrics=result.get("metrics", {}), tags={"iter": it})
                if budget_sec and (time.time() - start_ts) > budget_sec:
                    print("[BUDGET] Time budget exceeded; stopping further runs in this iteration.")
                    break
        all_runs.extend(iter_results)

        decision = analyze_and_decide(iter_results, target_delta=0.005)
        _write_json(RUNS_DIR / run_id / "decision.json", decision)
        print(f"[DECISION] {decision}")
        if decision.get("success"):
            goal_reached = True
            break

    # Summary
    summary = {
        "goal_reached": goal_reached,
        "total_runs": len(all_runs),
        "iterations": max_iters,
        "runs": all_runs,
    }
    _write_json(RUNS_DIR / "summary.json", summary)

    # Persist best run for reporting
    try:
        best = select_best_run(all_runs)
        if best:
            _write_json(RUNS_DIR / "best.json", best)
            # Also export a compact best_config.json
            _write_json(RUNS_DIR / "best_config.json", {"spec": best.get("spec", {}), "metrics": best.get("result", {}).get("metrics", {})})
    except Exception:
        pass

    # Light markdown report
    lines = ["# Iterative Experiment Report", "", f"Goal reached: {goal_reached}", ""]
    for r in all_runs:
        acc = r["result"]["metrics"].get("val_accuracy", 0.0)
        lines.append(f"- Iter {r['iter']} {r['name']}: acc={acc:.4f} run_id={r['run_id']}")
    (RUNS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Static HTML dashboard for quick scan
    try:
        write_dashboard(RUNS_DIR / "dashboard.html", all_runs)
    except Exception:
        pass
    # Optional plot of accuracies
    try:
        maybe_save_accuracy_bar(RUNS_DIR / "accuracy.png", all_runs)
    except Exception:
        pass

    # Aggregates for repeats
    try:
        aggs = aggregate_repeats(all_runs)
        if aggs:
            _write_json(RUNS_DIR / "aggregates.json", aggs)
    except Exception:
        pass
    print(f"[DONE] Wrote iteration report to {RUNS_DIR / 'report.md'}")


def main() -> None:
    if not NOVELTY_REPORT.exists():
        print(f"[ERR] Missing novelty report at {NOVELTY_REPORT}. Run agents_novelty.py first.")
        return
    try:
        novelty = json.loads(NOVELTY_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] Failed to read novelty report: {exc}")
        return
    # Allow overriding the number of iterations via env var MAX_ITERS (default=2)
    try:
        max_iters = int(os.getenv("MAX_ITERS", "2") or 2)
        if max_iters < 1:
            max_iters = 1
    except Exception:
        max_iters = 2
    iterate(novelty, max_iters=max_iters)


def select_best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the run dict with highest val_accuracy; empty dict if none."""
    best: Dict[str, Any] = {}
    best_acc = -1.0
    for r in runs:
        try:
            acc = float(r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0) or 0.0)
        except Exception:
            acc = 0.0
        if acc > best_acc:
            best_acc = acc
            best = r
    return best


def aggregate_repeats(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std val_accuracy for groups (baseline/novelty) ignoring _rep suffixes.
    Returns mapping name -> {mean, std, n}.
    """
    import math
    groups = {"baseline": [], "novelty": []}
    for r in runs:
        name = str(r.get("name") or "")
        base = name.split("_rep", 1)[0]
        if base in groups:
            try:
                acc = float(r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0) or 0.0)
                groups[base].append(acc)
            except Exception:
                pass
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in groups.items():
        if not vals:
            continue
        n = float(len(vals))
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        out[k] = {"mean": mean, "std": std, "n": n}
    return out


if __name__ == "__main__":
    main()
