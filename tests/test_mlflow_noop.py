from lab.logging_utils import try_mlflow_log


def test_try_mlflow_log_noop_without_mlflow(monkeypatch):
    # Ensure disabled by default
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    try_mlflow_log("run", {"lr": 0.001}, {"val_accuracy": 0.1})
    # Enable but without mlflow installed should still no-op
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    try_mlflow_log("run2", {"lr": 0.001}, {"val_accuracy": 0.1})

