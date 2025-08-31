from agents_iterate import aggregate_repeats


def test_aggregate_repeats_mean_std():
    runs = [
        {"name": "baseline", "result": {"metrics": {"val_accuracy": 0.2}}},
        {"name": "baseline_rep2", "result": {"metrics": {"val_accuracy": 0.4}}},
        {"name": "novelty", "result": {"metrics": {"val_accuracy": 0.5}}},
        {"name": "novelty_rep2", "result": {"metrics": {"val_accuracy": 0.7}}},
    ]
    agg = aggregate_repeats(runs)
    assert "baseline" in agg and "novelty" in agg
    assert abs(agg["baseline"]["mean"] - 0.3) < 1e-6
    assert abs(agg["novelty"]["mean"] - 0.6) < 1e-6
    assert agg["baseline"]["n"] == 2
    assert agg["novelty"]["n"] == 2

