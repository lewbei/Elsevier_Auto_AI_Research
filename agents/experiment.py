import json
import pathlib
from typing import Any, Dict

from dotenv import load_dotenv
from llm_utils import chat_json, LLMError
from lab.experiment_runner import run_experiment
from lab.config import dataset_path_for, get


load_dotenv()

EXPERIMENTS_DIR = pathlib.Path("experiments")
RUNS_DIR = pathlib.Path("runs")
NOVELTY_REPORT = pathlib.Path("data/novelty_report.json")


def _ensure_dirs() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def propose_experiment(novelty: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(get("project.goal", "your task") or "your task")
    system = (
        "You are an AI scientist planning a minimal, runnable deep-learning experiment for " + topic + ". "
        "Given themes and new ideas, propose a concrete spec using accessible components. Keep it simple and runnable "
        "on limited compute (<= 1 epoch, small steps)."
    )
    user_payload = {
        "novelty_report": novelty,
        "constraints": {
            "dataset": "ImageFolder under data/isic with train/ and val/",
            "compute": "CPU or single GPU, <= 1 epoch, limit steps",
            "framework": "PyTorch+Torchvision if available, else stub",
        },
        "output_schema": {
            "title": "string",
            "dataset_name": "string",
            "dataset_path": "string (default data/isic)",
            "input_size": 224,
            "model": "resnet18|efficientnet_b0|mobilenet_v3_small",
            "epochs": 1,
            "batch_size": 16,
            "lr": 0.001,
            "max_train_steps": 50,
            "augmentations": ["string"],
            "novelty_component": {
                "description": "string",
                "enabled": True
            }
        }
    }
    js = chat_json(system, json.dumps(user_payload, ensure_ascii=False), temperature=0.2)
    # Fill defaults if missing
    spec = {
        "title": js.get("title") or "Skin Cancer Experiment",
        "dataset_name": js.get("dataset_name") or "ISIC",
        "dataset_path": js.get("dataset_path") or dataset_path_for(),
        "input_size": int(js.get("input_size") or 224),
        "model": js.get("model") or "resnet18",
        "epochs": int(js.get("epochs") or 1),
        "batch_size": int(js.get("batch_size") or 16),
        "lr": float(js.get("lr") or 1e-3),
        "max_train_steps": int(js.get("max_train_steps") or 50),
        "augmentations": js.get("augmentations") or ["resize", "flip"],
        "novelty_component": js.get("novelty_component") or {"description": "baseline", "enabled": False},
    }
    return spec


def main() -> None:
    _ensure_dirs()
    if not NOVELTY_REPORT.exists():
        print(f"[ERR] Missing novelty report at {NOVELTY_REPORT}. Run agents_novelty.py first.")
        return
    try:
        novelty = json.loads(NOVELTY_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERR] Failed to read novelty report: {exc}")
        return

    try:
        spec = propose_experiment(novelty)
    except LLMError as exc:
        print(f"[ERR] LLM failed to propose experiment: {exc}")
        return

    exp_id = "exp_001"
    (EXPERIMENTS_DIR / f"{exp_id}.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote spec: {EXPERIMENTS_DIR / f'{exp_id}.json'}")

    result = run_experiment(spec)
    run_dir = RUNS_DIR / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote metrics: {run_dir / 'metrics.json'}")

    report = {
        "experiment_id": exp_id,
        "title": spec.get("title"),
        "spec": spec,
        "result": result,
        "conclusion": (
            "Stubbed result due to missing dependencies/dataset."
            if result.get("mode") != "real" else
            "Completed minimal real run; use more steps/epochs for stronger signal."
        )
    }
    (run_dir / "experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote experiment report: {run_dir / 'experiment_report.json'}")


if __name__ == "__main__":
    main()

