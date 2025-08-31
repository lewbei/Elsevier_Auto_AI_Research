import json
import time
import pathlib
from typing import Any, Dict
import random
import os


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


def dataset_choice() -> str:
    """Return selected dataset name based on env var DATASET; defaults to 'isic'."""
    ds = str(os.getenv("DATASET", "isic") or "isic").strip().lower()
    if ds in {"isic", "cifar10"}:
        return ds
    return "isic"


def _run_real(spec: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from torchvision import transforms  # type: ignore
        from torchvision.datasets import ImageFolder  # type: ignore
        from torchvision.models import resnet18  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {
            "mode": "stub",
            "reason": f"Missing torch/torchvision: {exc}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }

    seed = int(spec.get("seed") or 42)
    _seed_everything(seed)

    ds_choice = dataset_choice()
    use_fallback = str(os.getenv("ALLOW_FALLBACK_DATASET", "")).lower() in {"1", "true", "yes"}
    allow_download = str(os.getenv("ALLOW_DATASET_DOWNLOAD", "")).lower() in {"1", "true", "yes"}

    input_size = int(spec.get("input_size") or 224)
    batch_size = int(spec.get("batch_size") or 16)
    epochs = int(spec.get("epochs") or 1)

    # Optional generated augmentation (from engineer/codegen step)
    aug_callable = None
    try:
        # Guard: only enable if generated_aug imports and smokes fine
        from lab.codegen_utils import sanity_check_generated_aug  # type: ignore
        ok = sanity_check_generated_aug()
        if ok:
            from lab.generated_aug import GeneratedAug  # type: ignore
            aug_callable = GeneratedAug()
    except Exception:
        aug_callable = None

    train_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
    ]
    # Use generated augmentation only when novelty component is enabled
    nc = spec.get("novelty_component") or {}
    if aug_callable is not None and bool(nc.get("enabled", False)):
        train_transforms.append(aug_callable)  # type: ignore[arg-type]
    train_transforms.append(transforms.ToTensor())
    tfm_train = transforms.Compose(train_transforms)
    tfm_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    num_classes = None
    if ds_choice == "isic":
        data_dir = pathlib.Path(spec.get("dataset_path") or "data/isic")
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        if train_dir.exists() and val_dir.exists():
            ds_train = ImageFolder(str(train_dir), transform=tfm_train)
            ds_val = ImageFolder(str(val_dir), transform=tfm_val)
            num_classes = len(ds_train.classes)
        elif use_fallback:
            from torchvision.datasets import FakeData  # type: ignore
            num_classes = int(spec.get("num_classes", 2))
            ds_train = FakeData(size=int(spec.get("fallback_train_size", 200)), image_size=(3, input_size, input_size),
                                num_classes=num_classes, transform=tfm_train)
            ds_val = FakeData(size=int(spec.get("fallback_val_size", 50)), image_size=(3, input_size, input_size),
                              num_classes=num_classes, transform=tfm_val)
        else:
            return {
                "mode": "stub",
                "reason": f"Dataset not found under {data_dir} (expect train/ and val/ folders)",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0},
            }
    elif ds_choice == "cifar10":
        try:
            from torchvision.datasets import CIFAR10, FakeData  # type: ignore
        except Exception as exc:
            return {
                "mode": "stub",
                "reason": f"torchvision missing for CIFAR10: {exc}",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0},
            }
        root = pathlib.Path("data/cifar10")
        try:
            ds_train = CIFAR10(root=str(root), train=True, transform=tfm_train, download=allow_download)
            ds_val = CIFAR10(root=str(root), train=False, transform=tfm_val, download=allow_download)
            num_classes = 10
        except Exception:
            if use_fallback:
                num_classes = 10
                ds_train = FakeData(size=int(spec.get("fallback_train_size", 200)), image_size=(3, input_size, input_size),
                                    num_classes=num_classes, transform=tfm_train)
                ds_val = FakeData(size=int(spec.get("fallback_val_size", 50)), image_size=(3, input_size, input_size),
                                  num_classes=num_classes, transform=tfm_val)
            else:
                return {
                    "mode": "stub",
                    "reason": "CIFAR10 unavailable and download disabled; set ALLOW_DATASET_DOWNLOAD or ALLOW_FALLBACK_DATASET",
                    "started": _now(),
                    "finished": _now(),
                    "metrics": {"val_accuracy": 0.0},
                }
    else:
        return {
            "mode": "stub",
            "reason": f"Unknown dataset choice: {ds_choice}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    # Model selection (keep minimal)
    model_name = str(spec.get("model") or "resnet18").lower()
    if model_name == "resnet18":
        model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)
    # Use generated head if requested and available
    use_gen_head = bool(spec.get("use_generated_head", False))
    if use_gen_head:
        try:
            from lab.codegen_utils import sanity_check_generated_head  # type: ignore
            if sanity_check_generated_head():
                from lab.generated_head import GeneratedHead  # type: ignore
                p = float(spec.get("dropout_p", 0.2))
                model.fc = GeneratedHead(model.fc.in_features, int(num_classes), p)
            else:
                model.fc = nn.Linear(model.fc.in_features, int(num_classes))
        except Exception:
            model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    else:
        model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    model.to(device)

    lr = float(spec.get("lr") or 1e-3)
    optimizer_name = str(spec.get("optimizer") or "adam").lower()
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    started = _now()
    model.train()
    max_train_steps = int(spec.get("max_train_steps") or 50)
    step = 0
    for epoch in range(epochs):
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            step += 1
            if step >= max_train_steps:
                break
        if step >= max_train_steps:
            break

    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    acc = float(correct) / float(total) if total else 0.0

    finished = _now()
    return {
        "mode": "real",
        "reason": "ok",
        "started": started,
        "finished": finished,
        "metrics": {"val_accuracy": acc},
        "meta": {
            "device": device,
            "num_classes": num_classes,
            "seed": seed,
        },
    }


def run_experiment(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to run a minimal real experiment if torch+dataset present; otherwise return a stub
    result with explicit reason. This keeps the pipeline end-to-end even on limited setups.
    """
    try:
        return _run_real(spec)
    except Exception as exc:  # pragma: no cover
        return {
            "mode": "stub",
            "reason": f"Exception in real runner: {exc}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }
