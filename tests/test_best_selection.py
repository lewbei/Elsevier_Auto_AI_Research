from agents_iterate import select_best_run


def test_select_best_run_picks_highest():
    runs = [
        {"name": "a", "result": {"metrics": {"val_accuracy": 0.2}}},
        {"name": "b", "result": {"metrics": {"val_accuracy": 0.5}}},
        {"name": "c", "result": {"metrics": {"val_accuracy": 0.3}}},
    ]
    best = select_best_run(runs)
    assert best.get("name") == "b"

