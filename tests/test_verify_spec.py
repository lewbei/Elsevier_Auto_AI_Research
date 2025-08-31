from agents_iterate import verify_and_fix_spec


def test_verify_and_fix_spec_ranges():
    spec = {
        "input_size": 32,  # too small, will be clamped to >=96
        "epochs": 10,  # clamped to <=5
        "batch_size": 0,  # clamped to >=1
        "lr": 10.0,  # clamped to <=0.1
        "max_train_steps": 5,  # clamped to >=10
        "seed": -1,  # clamped to >=1
        "model": "weirdnet",
        "novelty_component": "something",
    }
    fixed = verify_and_fix_spec(spec)
    assert fixed["input_size"] >= 96
    assert fixed["epochs"] <= 5
    assert fixed["batch_size"] >= 1
    assert fixed["lr"] <= 0.1
    assert fixed["max_train_steps"] >= 10
    assert fixed["seed"] >= 1
    assert fixed["model"] == "resnet18"
    assert isinstance(fixed["novelty_component"], dict)
    assert "enabled" in fixed["novelty_component"]

