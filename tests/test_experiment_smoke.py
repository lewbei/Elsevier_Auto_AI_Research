import json
import pathlib


def test_agents_experiment_present():
    assert pathlib.Path("agents_experiment.py").exists()
    assert pathlib.Path("lab/experiment_runner.py").exists()


def test_requirements_lists_core_deps():
    req = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
    for pkg in ["requests", "python-dotenv", "pypdf"]:
        assert pkg in req

