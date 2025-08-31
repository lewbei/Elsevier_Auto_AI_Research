import json
import time
import pathlib
from typing import Any, Dict
import random
import os
import importlib
from lab.config import dataset_name, dataset_path_for, get_bool, dataset_kind, dataset_splits, get


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
    """Return selected dataset name.

    Resolution: explicit env DATASET (if set) > YAML config (dataset.name) > default 'isic'.
    This preserves env-based tests while enabling project-level YAML control.
    """
    env_ds = str(os.getenv("DATASET", "") or "").strip().lower()
    if env_ds in {"isic", "cifar10"}:
        return env_ds
    ds = dataset_name("isic")
    return ds if ds in {"isic", "cifar10"} else "isic"


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
    # YAML overrides env for flags
    use_fallback = get_bool("dataset.allow_fallback", False) or (str(os.getenv("ALLOW_FALLBACK_DATASET", "")).lower() in {"1", "true", "yes"})
    allow_download = get_bool("dataset.allow_download", False) or (str(os.getenv("ALLOW_DATASET_DOWNLOAD", "")).lower() in {"1", "true", "yes"})

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
    # Determine dataset kind (config first; fallback by choice)
    kind = dataset_kind(None)
    if not kind:
        kind = "cifar10" if ds_choice == "cifar10" else "imagefolder"

    if kind == "imagefolder":
        data_dir = pathlib.Path(spec.get("dataset_path") or dataset_path_for())
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        # allow custom split subfolder names via YAML
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
                # fallback: use val as test or FakeData if allowed
                if use_fallback:
                    from torchvision.datasets import FakeData  # type: ignore
                    ds_test = FakeData(size=int(spec.get("fallback_test_size", 50)), image_size=(3, input_size, input_size),
                                       num_classes=num_classes, transform=tfm_val)
                else:
                    ds_test = ds_val
        elif use_fallback:
            from torchvision.datasets import FakeData  # type: ignore
            num_classes = int(spec.get("num_classes", 2))
            ds_train = FakeData(size=int(spec.get("fallback_train_size", 200)), image_size=(3, input_size, input_size),
                                num_classes=num_classes, transform=tfm_train)
            ds_val = FakeData(size=int(spec.get("fallback_val_size", 50)), image_size=(3, input_size, input_size),
                              num_classes=num_classes, transform=tfm_val)
            ds_test = FakeData(size=int(spec.get("fallback_test_size", 50)), image_size=(3, input_size, input_size),
                               num_classes=num_classes, transform=tfm_val)
        else:
            return {
                "mode": "stub",
                "reason": f"ImageFolder dataset not found under {data_dir} (expect {train_dir.name}/ and {val_dir.name}/ folders)",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
            }
    elif kind == "cifar10":
        try:
            from torchvision.datasets import CIFAR10, FakeData  # type: ignore
            from torch.utils.data import random_split, Subset  # type: ignore
        except Exception as exc:
            return {
                "mode": "stub",
                "reason": f"torchvision missing for CIFAR10: {exc}",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
            }
        root = pathlib.Path(spec.get("dataset_path") or dataset_path_for("cifar10"))
        try:
            # Build train/val from training set via split, and use official test as test
            full_train = CIFAR10(root=str(root), train=True, transform=tfm_train, download=allow_download)
            full_val = CIFAR10(root=str(root), train=True, transform=tfm_val, download=False)
            num_classes = 10
            val_frac = float(get("dataset.val_fraction", 0.1) or 0.1)
            n_total = len(full_train)
            n_val = max(1, int(n_total * min(max(val_frac, 0.01), 0.5)))
            n_train = n_total - n_val
            import torch as _torch  # local alias for generator
            gen = _torch.Generator().manual_seed(int(spec.get("seed", 42)))
            train_subset, val_subset_tmp = random_split(full_train, [n_train, n_val], generator=gen)
            # Build val subset with val transforms using the same indices
            ds_val = Subset(full_val, val_subset_tmp.indices)  # type: ignore[attr-defined]
            ds_train = train_subset
            ds_test = CIFAR10(root=str(root), train=False, transform=tfm_val, download=allow_download)
        except Exception:
            if use_fallback:
                num_classes = 10
                ds_train = FakeData(size=int(spec.get("fallback_train_size", 200)), image_size=(3, input_size, input_size),
                                    num_classes=num_classes, transform=tfm_train)
                ds_val = FakeData(size=int(spec.get("fallback_val_size", 50)), image_size=(3, input_size, input_size),
                                  num_classes=num_classes, transform=tfm_val)
                ds_test = FakeData(size=int(spec.get("fallback_test_size", 50)), image_size=(3, input_size, input_size),
                                   num_classes=num_classes, transform=tfm_val)
            else:
                return {
                    "mode": "stub",
                    "reason": "CIFAR10 unavailable and download disabled; set ALLOW_DATASET_DOWNLOAD or ALLOW_FALLBACK_DATASET",
                    "started": _now(),
                    "finished": _now(),
                    "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
                }
    elif kind == "custom":
        # Optional: load custom dataset classes from config
        # Expect config keys under dataset.custom.{train,val} with module/class/kwargs
        def _load_custom(which: str, transform):
            entry = get(f"dataset.custom.{which}", None)
            if not isinstance(entry, dict):
                return None
            module = entry.get("module")
            cls = entry.get("class")
            kwargs = entry.get("kwargs") or {}
            if not module or not cls:
                return None
            try:
                mod = importlib.import_module(str(module))
                ctor = getattr(mod, str(cls))
                if transform is not None:
                    kwargs = dict(kwargs)
                    if "transform" not in kwargs:
                        kwargs["transform"] = transform
                return ctor(**kwargs)
            except Exception:
                return None

        ds_train = _load_custom("train", tfm_train)
        ds_val = _load_custom("val", tfm_val)
        if ds_train is None or ds_val is None:
            return {
                "mode": "stub",
                "reason": "Custom dataset not fully configured or failed to import. See dataset.custom.* in config.",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
            }
        # optional custom test dataset
        ds_test = None
        entry_test = get("dataset.custom.test", None)
        if isinstance(entry_test, dict):
            module = entry_test.get("module")
            cls = entry_test.get("class")
            kwargs = entry_test.get("kwargs") or {}
            try:
                mod = importlib.import_module(str(module))
                ctor = getattr(mod, str(cls))
                if "transform" not in kwargs:
                    kwargs["transform"] = tfm_val
                ds_test = ctor(**kwargs)
            except Exception:
                ds_test = None
        if ds_test is None:
            # fallback to val as test when not specified
            ds_test = ds_val
        try:
            # Try to infer num_classes by inspecting first target or attribute
            num_classes = int(get("dataset.num_classes", 0) or 0)
        except Exception:
            num_classes = 0
        if not num_classes:
            try:
                num_classes = len(getattr(ds_train, "classes"))  # type: ignore[arg-type]
            except Exception:
                num_classes = int(spec.get("num_classes", 2))
    else:
        return {
            "mode": "stub",
            "reason": f"Unknown dataset kind: {kind}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
        }

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

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

    # Validate (val split)
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
    val_acc = float(correct) / float(total) if total else 0.0

    # Test evaluation
    correct_t = 0
    total_t = 0
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct_t += (pred == yb).sum().item()
            total_t += yb.numel()
    test_acc = float(correct_t) / float(total_t) if total_t else 0.0

    finished = _now()
    return {
        "mode": "real",
        "reason": "ok",
        "started": started,
        "finished": finished,
        "metrics": {"val_accuracy": val_acc, "test_accuracy": test_acc},
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
