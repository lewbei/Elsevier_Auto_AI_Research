# Tiny training hooks for ResNet18 + Scale-Selective Spectral Attention (SSSA)
# Allowed building blocks: torchvision transforms and torch.nn layers only.
import torch
import torch.nn as nn
from typing import Any
try:
    import torchvision.transforms as T  # type: ignore
except Exception:
    T = None  # type: ignore

def build_train_transforms(input_size: int):
    """
    Training augmentation is not predefined here.
    Return None to let the runner/config decide, or optionally return a
    torchvision transforms.Compose constructed by the caller.
    """
    _ = int(max(96, min(512, int(input_size))))  # normalize but unused
    return None


def build_model_head(in_features: int, num_classes: int) -> nn.Module:
    """
    Small classification head. Kept intentionally compact so that any
    additional novelty parameters (SSSA) can remain <= ~10% of baseline params.
    Uses only torch.nn layers.
    """
    # Small hidden projection: scale with input but floor to keep it tiny.
    hidden = int(max(32, min(128, max(1, int(in_features * 0.05)))))
    head = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes),
    )
    return head


def update_spec(spec: dict) -> dict:
    """
    Update the run spec to include succinct SSSA (Scale-Selective Spectral Attention)
    configuration and enforce safe numeric ranges.

    SSSA summary (stored under spec['novelty_component']['sssa_params']):
      - block_size: 8 (non-overlapping 8x8 DCT blocks on normalized input)
      - bands: partition zig-zag flattened block into low=6, mid=12, high=(remaining)
      - target_scales: ['layer2','layer3','layer4'] (ResNet intermediate outputs)
      - per-band 1x1 conv projections are bilinearly upsampled to each scale spatial size
      - per-(band x channel) scalar gates are learned (init=1.0) with L1 reg lambda=1e-4
      - parameter increase target: <= 0.10 (10%) of baseline; projection widths must be reduced as needed
      - enabled: True
    Notes: The core model modifications are implemented as a lightweight spectral modulation
    on multi-scale features and are intended to be applied before any fusion/pooling.
    """
    # Ensure basic numeric ranges are safe and deterministic
    spec = dict(spec)  # shallow copy to avoid mutating caller data in-place
    spec.setdefault('title', 'ResNet18 + Scale-Selective Spectral Attention (SSSA)')
    # Clamp input size to allowed range
    if 'input_size' in spec:
        spec['input_size'] = int(max(96, min(512, int(spec['input_size']))))
    # Clamp learning rate to safe range (1e-5 .. 1e-1)
    if 'lr' in spec:
        spec['lr'] = float(max(1e-5, min(1e-1, float(spec['lr']))))
    # Clamp max_train_steps to safe range (10 .. 1000)
    if 'max_train_steps' in spec:
        spec['max_train_steps'] = int(max(10, min(1000, int(spec['max_train_steps']))))
    # Ensure seed is present
    spec.setdefault('seed', 42)

    # Production defaults for common training toggles (safe, conservative)
    def _truthy(x: Any) -> bool:
        s = str(x).strip().lower()
        return s in {"1", "true", "yes", "on"}

    # batch_size and optimizer defaults
    try:
        if 'batch_size' in spec:
            spec['batch_size'] = int(max(1, min(1024, int(spec['batch_size']))))
        else:
            spec['batch_size'] = 16
    except Exception:
        spec['batch_size'] = 16
    # Do not predefine optimizer here; leave to runner/config

    # DataLoader hints
    try:
        if 'num_workers' in spec:
            spec['num_workers'] = int(max(0, min(16, int(spec['num_workers']))))
    except Exception:
        spec['num_workers'] = 0
    try:
        spec['pin_memory'] = bool(spec['pin_memory']) if 'pin_memory' in spec else False
    except Exception:
        spec['pin_memory'] = False
    try:
        spec['persistent_workers'] = bool(spec['persistent_workers']) if 'persistent_workers' in spec else False
    except Exception:
        spec['persistent_workers'] = False

    # Reproducibility and AMP
    try:
        spec['deterministic'] = _truthy(spec['deterministic']) if 'deterministic' in spec else False
    except Exception:
        spec['deterministic'] = False
    try:
        spec['amp'] = _truthy(spec['amp']) if 'amp' in spec else False
    except Exception:
        spec['amp'] = False
    try:
        if 'log_interval' in spec:
            spec['log_interval'] = int(max(0, int(spec['log_interval'])))
    except Exception:
        pass

    # Attach succinct SSSA config under novelty_component for downstream hooks
    novelty = spec.get('novelty_component', {})
    sssa_params = {
        'enabled': True,
        'description': 'Scale-Selective Spectral Attention (SSSA): 8x8 DCT blocks, '
                       'zig-zag partitioning into low(6), mid(12), high(rest); '
                       'per-band 1x1 projections + per-(band×channel) gates (init=1.0); '
                       'L1 regularization lambda=1e-4; target param overhead <= 0.10.',
        'block_size': 8,
        'bands': {'low': 6, 'mid': 12, 'high': 'rest'},
        'target_scales': ['layer2', 'layer3', 'layer4'],
        'gate_init': 1.0,
        'l1_lambda': 1e-4,
        'param_overhead_fraction': 0.10,
    }
    novelty['sssa_params'] = sssa_params
    novelty['enabled'] = True
    spec['novelty_component'] = novelty

    return spec
