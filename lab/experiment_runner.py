"""Experiment execution helpers.

This refactor splits environment checks, dataset preparation, and execution
into smaller helpers so agents can reuse them flexibly.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lab.config import (
    dataset_name,
    dataset_path_for,
    dataset_kind,
    dataset_splits,
    get_bool,
    get,
)
from lab.logging_utils import is_verbose, vprint


@dataclass
class EnvironmentStatus:
    torch_available: bool
    torchvision_available: bool
    cuda_available: bool
    reason: Optional[str] = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _seed_everything(seed: int) -> None:
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(seed)
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)


def detect_environment() -> EnvironmentStatus:
    try:
        import torch  # type: ignore
    except Exception as exc:
        return EnvironmentStatus(False, False, False, reason=f"torch import failed: {exc}")
    try:
        import torchvision  # type: ignore
    except Exception as exc:
        return EnvironmentStatus(True, False, torch.cuda.is_available(), reason=f"torchvision import failed: {exc}")
    return EnvironmentStatus(True, True, torch.cuda.is_available())


def dataset_choice() -> str:
    env_ds = str(os.getenv("DATASET", "") or "").strip().lower()
    if env_ds in {"isic", "cifar10"}:
        return env_ds
    ds = dataset_name("isic")
    return ds if ds in {"isic", "cifar10"} else "isic"


def _imagefolder_datasets(spec: Dict[str, Any], *, fallback: bool) -> Tuple[Any, Any, Any, int]:
    from torchvision import transforms  # type: ignore
    from torchvision.datasets import ImageFolder, FakeData  # type: ignore

    input_size = int(spec.get("input_size") or 224)
    use_default_aug = bool(spec.get("use_default_augmentation", True))

    train_transforms = [transforms.Resize((input_size, input_size))]
    if use_default_aug:
        train_transforms.append(transforms.RandomHorizontalFlip())
    nc = spec.get("novelty_component") or {}
    aug_callable = None
    try:
        from lab.codegen_utils import sanity_check_generated_aug  # type: ignore

        if sanity_check_generated_aug():
            from lab.generated_aug import GeneratedAug  # type: ignore

            aug_callable = GeneratedAug()
    except Exception:
        aug_callable = None
    if aug_callable is not None and bool(nc.get("enabled", False)):
        train_transforms.append(aug_callable)  # type: ignore[arg-type]
    train_transforms.append(transforms.ToTensor())

    tfm_train = transforms.Compose(train_transforms)
    tfm_val = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])

    data_dir = Path(spec.get("dataset_path") or dataset_path_for())
    splits = dataset_splits()
    train_dir = data_dir / splits.get("train", "train")
    val_dir = data_dir / splits.get("val", "val")
    test_dir = data_dir / splits.get("test", "test")

    if train_dir.exists() and val_dir.exists():
        ds_train = ImageFolder(str(train_dir), transform=tfm_train)
        ds_val = ImageFolder(str(val_dir), transform=tfm_val)
        num_classes = len(ds_train.classes)
        if test_dir.exists():
            ds_test = ImageFolder(str(test_dir), transform=tfm_val)
        else:
            ds_test = ds_val
        return ds_train, ds_val, ds_test, num_classes

    if fallback:
        num_classes = int(spec.get("num_classes", 2))
        ds_train = FakeData(
            size=int(spec.get("fallback_train_size", 200)),
            image_size=(3, input_size, input_size),
            num_classes=num_classes,
            transform=tfm_train,
        )
        ds_val = FakeData(
            size=int(spec.get("fallback_val_size", 50)),
            image_size=(3, input_size, input_size),
            num_classes=num_classes,
            transform=tfm_val,
        )
        ds_test = FakeData(
            size=int(spec.get("fallback_test_size", 50)),
            image_size=(3, input_size, input_size),
            num_classes=num_classes,
            transform=tfm_val,
        )
        return ds_train, ds_val, ds_test, num_classes

    raise FileNotFoundError(f"ImageFolder dataset not found under {data_dir} (expect {train_dir.name}/ and {val_dir.name}/ folders)")


def _prepare_datasets(spec: Dict[str, Any]) -> Tuple[Any, Any, Any, int]:
    choice = dataset_choice()
    fallback = get_bool("dataset.allow_fallback", False) or (str(os.getenv("ALLOW_FALLBACK_DATASET", "")).lower() in {"1", "true", "yes"})
    allow_download = get_bool("dataset.allow_download", False) or (str(os.getenv("ALLOW_DATASET_DOWNLOAD", "")).lower() in {"1", "true", "yes"})
    kind = dataset_kind(None) or ("cifar10" if choice == "cifar10" else "imagefolder")

    if kind == "imagefolder":
        return _imagefolder_datasets(spec, fallback=fallback)

    if kind == "cifar10":
        from torchvision import transforms  # type: ignore
        from torchvision.datasets import CIFAR10  # type: ignore

        if not allow_download:
            raise FileNotFoundError("CIFAR10 requested but download is disabled (set dataset.allow_download true).")
        input_size = int(spec.get("input_size") or 32)
        tfm = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
        root = Path(dataset_path_for("cifar10"))
        ds_train = CIFAR10(str(root), train=True, download=True, transform=tfm)
        ds_val = CIFAR10(str(root), train=False, download=True, transform=tfm)
        return ds_train, ds_val, ds_val, 10

    raise ValueError(f"Unsupported dataset_kind: {kind}")


def run_experiment(spec: Dict[str, Any], *, on_progress=None, should_stop=None) -> Dict[str, Any]:
    status = detect_environment()
    if not status.torch_available or not status.torchvision_available:
        reason = status.reason or "missing torch/torchvision"
        res = {
            "mode": "stub",
            "reason": reason,
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }
        if is_verbose():
            vprint(f"Runner mode=stub reason={reason}")
        return res

    if not os.getenv("ALLOW_CPU_FOR_TESTS") and not status.cuda_available:
        res = {
            "mode": "stub",
            "reason": "CUDA/GPU required. set ALLOW_CPU_FOR_TESTS=1 only for tests.",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }
        if is_verbose():
            vprint("Runner mode=stub reason=no CUDA")
        return res

    # Optional spec normalization
    try:
        from lab.spec_validation import normalize_spec as _normalize_spec

        spec = _normalize_spec(spec)
    except Exception:
        pass

    _seed_everything(int(spec.get("seed") or 42))

    try:
        ds_train, ds_val, ds_test, num_classes = _prepare_datasets(spec)
    except Exception as exc:
        return {
            "mode": "stub",
            "reason": str(exc),
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }

    if is_verbose():
        try:
            summary = json.dumps(
                {
                    "dataset_path": spec.get("dataset_path"),
                    "input_size": spec.get("input_size"),
                    "batch_size": spec.get("batch_size"),
                    "epochs": spec.get("epochs"),
                    "lr": spec.get("lr"),
                    "max_train_steps": spec.get("max_train_steps"),
                    "model": spec.get("model"),
                    "novelty_component": spec.get("novelty_component", {}),
                },
                ensure_ascii=False,
            )
            vprint("Spec summary: " + summary)
        except Exception:
            pass

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from torchvision import transforms  # type: ignore
    from torchvision.models import resnet18  # type: ignore

    input_size = int(spec.get("input_size") or 224)
    batch_size = int(spec.get("batch_size") or 16)
    epochs = int(spec.get("epochs") or 1)
    lr = float(spec.get("lr") or 1e-3)
    max_steps = int(spec.get("max_train_steps") or 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    steps = 0
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            steps += 1
            if steps > max_steps:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if steps > max_steps:
            break

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = correct / max(1, total)

    result = {
        "mode": "train",
        "started": _now(),
        "finished": _now(),
        "metrics": {
            "val_accuracy": acc,
        },
    }

    if is_verbose():
        vprint(f"Runner metrics: {result['metrics']}")

    return result


__all__ = [
    "detect_environment",
    "run_experiment",
    "dataset_choice",
]
