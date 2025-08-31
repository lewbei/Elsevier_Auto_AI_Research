from pathlib import Path
from lab.logging_utils import capture_env, write_json
from lab.report_html import write_dashboard


def test_capture_env_returns_dict(tmp_path: Path):
    env = capture_env()
    assert isinstance(env, dict)
    p = tmp_path / "env.json"
    write_json(p, env)
    assert p.exists()


def test_write_dashboard_creates_html(tmp_path: Path):
    runs = [
        {"iter": 1, "name": "baseline", "result": {"metrics": {"val_accuracy": 0.1}, "mode": "stub"}, "run_id": "r1"},
        {"iter": 1, "name": "novelty", "result": {"metrics": {"val_accuracy": 0.2}, "mode": "stub"}, "run_id": "r1"},
    ]
    out = tmp_path / "dashboard.html"
    write_dashboard(out, runs)
    assert out.exists()
    assert "Experiment Dashboard" in out.read_text(encoding="utf-8")

