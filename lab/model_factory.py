"""Model Factory

Provides pluggable backbone construction so architectural novelty can be explored.

Supported identifiers (spec.model or spec.arch.type):
  - resnet18          (torchvision, no pretrained weights by default)
  - simple_cnn        (configurable width/depth)

Spec Extensions:
  spec.arch = {
      'type': 'simple_cnn',
      'width': 64,
      'depth': 4,
      'activation': 'relu',  # relu|gelu|silu
      'norm': 'batch',       # batch|layer|none
      'dropout': 0.0
  }

Novelty mutations may supply spec.novelty_component.arch_mutation with keys:
  - width_factor (float)
  - depth_delta (int)
  - add_dropout (float)
  - activation_override

build_model(spec, num_classes, task='classification') -> (nn.Module, meta_dict)
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

def build_model(spec: Dict[str, Any], num_classes: int, task: str = 'classification') -> Tuple["nn.Module", Dict[str, Any]]:  # type: ignore
    try:
        import torch
        import torch.nn as nn
        import math
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PyTorch required for build_model: {exc}")

    model_name = str(spec.get('model') or spec.get('arch', {}).get('type') or 'resnet18').lower()
    arch_cfg = dict(spec.get('arch') or {})

    # Apply novelty arch mutation if present
    mut = ((spec.get('novelty_component') or {}).get('arch_mutation')) if isinstance(spec.get('novelty_component'), dict) else None
    if isinstance(mut, dict):
        # Only mutate simple_cnn architectures
        if model_name in {'simple_cnn'} or mut.get('force_simple', False):
            model_name = 'simple_cnn'
            arch_cfg.setdefault('type', 'simple_cnn')
            w = int(arch_cfg.get('width', 64))
            d = int(arch_cfg.get('depth', 4))
            if 'width_factor' in mut:
                try:
                    w = max(8, min(1024, int(w * float(mut['width_factor']))))
                except Exception:
                    pass
            if 'depth_delta' in mut:
                try:
                    d = max(1, min(32, d + int(mut['depth_delta'])))
                except Exception:
                    pass
            arch_cfg['width'] = w
            arch_cfg['depth'] = d
            if 'add_dropout' in mut:
                try:
                    arch_cfg['dropout'] = float(arch_cfg.get('dropout', 0.0)) + float(mut['add_dropout'])
                except Exception:
                    pass
            if 'activation_override' in mut:
                arch_cfg['activation'] = mut['activation_override']

    if model_name == 'simple_cnn':
        width = int(arch_cfg.get('width', 64))
        depth = int(arch_cfg.get('depth', 4))
        activation_name = str(arch_cfg.get('activation', 'relu')).lower()
        norm_name = str(arch_cfg.get('norm', 'batch')).lower()
        dropout_p = float(arch_cfg.get('dropout', 0.0))

        Act = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
        }.get(activation_name, nn.ReLU)

        def norm_layer(ch: int):
            if norm_name == 'batch':
                return nn.BatchNorm2d(ch)
            if norm_name == 'layer':
                # Use GroupNorm with 1 group as a LayerNorm 2D surrogate
                return nn.GroupNorm(1, ch)
            return nn.Identity()

        layers = []
        in_ch = 3
        for i in range(depth):
            layers.append(nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=False))
            layers.append(norm_layer(width))
            layers.append(Act())
            if dropout_p > 0:
                layers.append(nn.Dropout2d(dropout_p))
            in_ch = width
        layers.append(nn.AdaptiveAvgPool2d(1))
        body = nn.Sequential(*layers)
        head_out = 1 if task == 'regression' else num_classes
        head = nn.Linear(width, head_out)
        model = nn.Sequential(body, nn.Flatten(), head)
        meta = {
            'backbone': 'simple_cnn',
            'width': width,
            'depth': depth,
            'params_million': round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        }
        return model, meta

    # Fallback / ResNet18 path
    try:
        from torchvision.models import resnet18  # type: ignore
        import torch.nn as nn  # re-import for isolated scope clarity
        m = resnet18(weights=None)
        # Caller will replace final fc later for regression or custom head
        meta = {'backbone': 'resnet18', 'params_million': round(sum(p.numel() for p in m.parameters()) / 1e6, 3)}
        return m, meta
    except Exception as exc:  # pragma: no cover
        # Final fallback: extremely small CNN
        import torch.nn as nn  # type: ignore
        small = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, num_classes if task!='regression' else 1)
        )
        meta = {'backbone': 'fallback_small', 'error': str(exc)}
        return small, meta


__all__ = ['build_model']