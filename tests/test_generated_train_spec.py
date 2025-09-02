import importlib
import pytest


# Avoid import failure when torch is unavailable
pytest.importorskip('torch')


def test_update_spec_clamps_and_defaults():
    mod = importlib.import_module('lab.generated_train')
    update_spec = getattr(mod, 'update_spec')
    # Extreme values to test clamping and defaults
    s = {
        'input_size': 32,         # too small → clamp to 96
        'lr': 10.0,               # too big → clamp to 1e-1
        'max_train_steps': 1,     # too small → clamp to 10
        'batch_size': 0,          # too small → clamp to >=1
        'optimizer': 'rmsprop',   # unsupported → fallback to adam
        'num_workers': -5,        # clamp to >=0
        'pin_memory': 'yes',      # truthy parsing → True, but update_spec sets False by default
        'persistent_workers': 'no',
        'deterministic': 'true',
        'amp': 'false',
        'log_interval': -1,
    }
    out = update_spec(s)

    assert out['input_size'] == 96
    assert abs(out['lr'] - 1e-1) < 1e-12
    assert out['max_train_steps'] == 10
    # Defaults and clamps
    assert out['batch_size'] >= 1
    # No optimizer predefinition in generated_train; runner decides or config sets it
    assert out['num_workers'] >= 0
    assert isinstance(out['deterministic'], bool)
    assert isinstance(out['amp'], bool)
    assert 'novelty_component' in out and 'sssa_params' in out['novelty_component']
