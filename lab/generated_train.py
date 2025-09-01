# Tiny training hooks for ResNet18 + fixed multi-scale edge modulator (CFEM-O)
# Exposes exactly three functions required by the training harness:
#   build_train_transforms(input_size)
#   build_model_head(in_features, num_classes)
#   update_spec(spec)
#
# This file is intentionally minimal and deterministic. It does not construct the
# modulator layers themselves (those are attached in model construction code),
# but it injects the required configuration for the harness to build/freeze
# modules and to compute the orthogonality regularizer schedule.

from typing import Dict, Any
from torchvision import transforms
import torch.nn as nn
import copy

# Build a compact but useful augmentation pipeline (<=4 transforms).
# Allowed transforms: Resize, RandomHorizontalFlip, ColorJitter, ToTensor.
def build_train_transforms(input_size: int):
    # Keep input_size in safe range [96, 512]
    if input_size < 96:
        input_size = 96
    elif input_size > 512:
        input_size = 512

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ])


# Lightweight classifier head used on top of a ResNet backbone.
# in_features: number of channels fed to the head (flattened if needed).
# num_classes: final output classes.
def build_model_head(in_features: int, num_classes: int):
    # keep a tiny head compatible with short-probe regime training
    mid = max(16, in_features // 4)
    return nn.Sequential(
        nn.Linear(in_features, mid),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(mid, num_classes),
    )


# Update the experiment spec to include the modulator / orthogonality schedule and
# short-probe training defaults. This function performs minimal, deterministic edits.
def update_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    s = copy.deepcopy(spec)

    # Ensure sensible, safe ranges for a reproducible run
    s.setdefault("seed", 42)
    s["batch_size"] = 8  # short-probe baseline
    s["max_train_steps"] = 100
    s["lr"] = float(max(1e-5, min(1e-1, s.get("lr", 1e-4))))
    # set default optimizer if missing
    s.setdefault("optimizer", "adam")

    # Novelty component / modulator config describing how to construct the module.
    # The actual implementation in model-building code should:
    #  - compute a fixed multi-scale edge prior (3x3 Sobel + 5x5 Laplacian),
    #    aggregate to a single 1-channel map, resize to match ResNet layer2 spatial
    #    dims, pass through a 1x1 conv + sigmoid => 'att' (channel attention).
    #  - include a tiny learnable 3x3 edge branch (a small conv producing a
    #    single-channel map). Pool both att and learned edge and compute orthogonality
    #    loss L_orth = mean((cosine(pool(att), pool(learned_edge)))^2).
    #  - fuse multiplicatively at ResNet layer2: feat := feat * (1 + alpha * att), alpha=1.0.
    s.setdefault("novelty_component", {})
    nc = s["novelty_component"]
    nc["name"] = "fixed_multiscale_edge_modulator_cfem_o"
    nc["enabled"] = True

    # Deterministic hyperparameters for the modulator
    nc["fixed_edge_priors"] = {
        "scales": [3, 5],                       # 3x3 Sobel + 5x5 Laplacian
        "aggregation": "sum_to_single_channel", # aggregate to 1-channel map
        "normalize": True,
    }
    nc["modulation"] = {
        "apply_at": "resnet_layer2",  # hook location
        "fusion": "multiplicative",   # feat := feat * (1 + alpha * att)
        "alpha": 1.0,
        "att_transform": {"conv1x1_out_channels": None, "activation": "sigmoid"},
    }
    nc["learnable_edge_branch"] = {
        "kernel_size": 3,
        "out_channels": 1,
        "init": "kaiming_uniform",
        # only this branch + modulator + classifier head are unfrozen in short-probe
    }
    # Orthogonality regularizer schedule: lambda ramps linearly 0 -> 0.1 over 50 steps
    nc["orthogonality"] = {
        "loss_name": "L_orth",
        "formulation": "mean((cosine(pool(att), pool(learned_edge)))^2)",
        "lambda_max": 0.1,
        "ramp_steps": 50,
        "pooling": "global_avg",  # pooling used before cosine similarity
    }

    # Short-probe regime: freeze backbone except modulator, learnable-edge branch, classifier head.
    # If the downstream harness wants "full" training, set short_probe to False externally.
    s.setdefault("short_probe", True)
    if s["short_probe"]:
        s["freeze_backbone_except"] = ["modulator", "learnable_edge_branch", "head"]
        # Ensure optimizer/steps consistent with short-probe baseline
        s["batch_size"] = 8
        s["max_train_steps"] = 100
        s["lr"] = 1e-4

    # Provide a concise textual description (useful for logging)
    s["title"] = "ResNet18 + Fixed Multi-scale Sobel Multiplicative Modulator (CFEM-O)"
    s["notes"] = (
        "Inject fixed multi-scale edge prior (3x3 Sobel + 5x5 Laplacian), "
        "1x1 conv + sigmoid => att, fuse at resnet layer2 multiplicatively with alpha=1.0. "
        "Include tiny learnable 3x3 edge branch. Orthogonality loss ramps 0->0.1 over 50 steps. "
        "Short-probe: freeze backbone except modulator/learnable-edge/head."
    )

    return s