import json
import time
import pathlib
from typing import Any, Dict
import random
import os
import importlib
from lab.config import dataset_name, dataset_path_for, get_bool, dataset_kind, dataset_splits, get
from lab.logging_utils import vprint, is_verbose


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
    # Normalize/validate spec (optional dependency)
    try:
        from lab.spec_validation import normalize_spec as _normalize_spec
        spec = _normalize_spec(spec)
    except Exception:
        pass
    # Verbose: show spec summary before attempting real/stub run
    if is_verbose():
        try:
            import json as _json
            vprint("Spec summary: " + _json.dumps({
                "dataset_path": spec.get("dataset_path"),
                "input_size": spec.get("input_size"),
                "batch_size": spec.get("batch_size"),
                "epochs": spec.get("epochs"),
                "lr": spec.get("lr"),
                "max_train_steps": spec.get("max_train_steps"),
                "model": spec.get("model"),
                "novelty_component": spec.get("novelty_component", {}),
            }, ensure_ascii=False))
        except Exception:
            pass
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from torchvision import transforms  # type: ignore
        from torchvision.datasets import ImageFolder  # type: ignore
        from torchvision.models import resnet18  # type: ignore
    except Exception as exc:  # pragma: no cover
        res = {
            "mode": "stub",
            "reason": f"Missing torch/torchvision: {exc}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }
        if is_verbose():
            vprint("Runner mode=stub reason=missing torch/vision")
        return res

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

    # Optional generated train hooks (advanced codegen)
    gtrain = None
    try:
        import lab.generated_train as _gtrain  # type: ignore
        gtrain = _gtrain
    except Exception:
        gtrain = None

    # Control default augmentation via spec.use_default_augmentation (True by default)
    try:
        use_default_aug = bool(spec.get("use_default_augmentation", True))
    except Exception:
        use_default_aug = True
    train_transforms = [transforms.Resize((input_size, input_size))]
    if use_default_aug:
        train_transforms.append(transforms.RandomHorizontalFlip())
    # Use generated augmentation only when novelty component is enabled
    nc = spec.get("novelty_component") or {}
    if aug_callable is not None and bool(nc.get("enabled", False)):
        train_transforms.append(aug_callable)  # type: ignore[arg-type]
    train_transforms.append(transforms.ToTensor())
    # Allow advanced hook to override/extend train transforms
    tfm_train = None
    if gtrain is not None and hasattr(gtrain, "build_train_transforms"):
        try:
            tfm_candidate = gtrain.build_train_transforms(int(input_size))  # type: ignore[attr-defined]
            if tfm_candidate is not None:
                tfm_train = tfm_candidate
        except Exception:
            tfm_train = None
    if tfm_train is None:
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
            res = {
                "mode": "stub",
                "reason": f"ImageFolder dataset not found under {data_dir} (expect {train_dir.name}/ and {val_dir.name}/ folders)",
                "started": _now(),
                "finished": _now(),
                "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
            }
            if is_verbose():
                vprint(res["reason"])
            return res
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
                res = {
                    "mode": "stub",
                    "reason": "CIFAR10 unavailable and download disabled; set ALLOW_DATASET_DOWNLOAD or ALLOW_FALLBACK_DATASET",
                    "started": _now(),
                    "finished": _now(),
                    "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
                }
                if is_verbose():
                    vprint(res["reason"])
                return res
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
        res = {
            "mode": "stub",
            "reason": f"Unknown dataset kind: {kind}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0, "test_accuracy": 0.0},
        }
        if is_verbose():
            vprint(res["reason"])
        return res

    # DataLoader tuning via spec/config
    try:
        nworkers = int(spec.get("num_workers", int(get("dataset.num_workers", 0) or 0)))
    except Exception:
        nworkers = 0
    use_cuda = (hasattr(torch, "cuda") and torch.cuda.is_available())
    try:
        pin_memory = bool(spec.get("pin_memory", use_cuda))
    except Exception:
        pin_memory = bool(use_cuda)
    try:
        persistent_workers = bool(spec.get("persistent_workers", (nworkers > 0)))
    except Exception:
        persistent_workers = (nworkers > 0)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nworkers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if nworkers > 0 else False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, nworkers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if nworkers > 0 else False,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, nworkers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if nworkers > 0 else False,
    )

    # Infer task (classification|regression|detection|segmentation) with overrides
    def _infer_task_from_sample() -> str:
        try:
            sample = ds_train[0]
            yb = sample[1] if isinstance(sample, (list, tuple)) and len(sample) > 1 else sample
            if isinstance(yb, dict) and ("boxes" in yb or "labels" in yb):
                return "detection"
            import torch as _torch  # type: ignore
            if _torch.is_tensor(yb):
                if yb.dtype.is_floating_point and (yb.ndim == 0 or yb.ndim == 1):
                    return "regression"
                if yb.ndim >= 2:
                    return "segmentation"
            return "classification"
        except Exception:
            return "classification"

    task = str(spec.get("task") or "").strip().lower() or _infer_task_from_sample()

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    # Optional deterministic mode for cudnn
    try:
        if bool(spec.get("deterministic", False)) and hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # Model selection (keep minimal)
    model_name = str(spec.get("model") or "resnet18").lower()
    if model_name == "resnet18":
        model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)
    # Heads + losses per task
    if task == "regression":
        out_dim = int(spec.get("num_outputs", 1) or 1)
        model.fc = nn.Linear(model.fc.in_features, out_dim)
        # Optional loss override for regression
        loss_name = str(spec.get("loss") or "").strip().lower()
        if loss_name in {"l1", "mae"}:
            loss_fn = nn.L1Loss()
        elif loss_name in {"huber", "smooth_l1"}:
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()
    else:  # classification (default)
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
        # Advanced hook may override head entirely
        if gtrain is not None and hasattr(gtrain, "build_model_head"):
            try:
                head = gtrain.build_model_head(int(model.fc.in_features), int(num_classes))  # type: ignore[attr-defined]
                if head is not None:
                    model.fc = head
            except Exception:
                pass
        # Optional loss override for classification
        loss_name = str(spec.get("loss") or "").strip().lower()
        if loss_name in {"bce", "bcewithlogits"}:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    lr = float(spec.get("lr") or 1e-3)
    optimizer_name = str(spec.get("optimizer") or "adam").lower()
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional class weights for classification
    class_weights = None
    if task != "regression":
        cw = spec.get("class_weights")
        if isinstance(cw, list) and num_classes and len(cw) == int(num_classes):
            try:
                import torch as _torch
                class_weights = _torch.tensor([float(x) for x in cw], dtype=_torch.float32).to(device)
            except Exception:
                class_weights = None
        # Apply weights only for CE when no explicit loss override is requested
        try:
            loss_name = str(spec.get("loss") or "").strip().lower()
        except Exception:
            loss_name = ""
        if loss_name in {"", "ce", "cross_entropy"}:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optional LR scheduler (step or cosine)
    sched = None
    try:
        sched_name = str(spec.get("lr_scheduler") or "").lower()
        if sched_name == "step":
            step_size = int(spec.get("lr_step_size") or 50)
            gamma = float(spec.get("lr_gamma") or 0.1)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        elif sched_name == "cosine":
            tmax = int(spec.get("lr_tmax") or max(1, int(spec.get("max_train_steps") or 50)))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax)
    except Exception:
        sched = None

    # Mixed precision (AMP) optional
    use_amp = bool(spec.get("amp", False)) and device == "cuda"
    scaler = None
    if use_amp:
        try:
            from torch.cuda.amp import GradScaler  # type: ignore
            scaler = GradScaler()
        except Exception:
            scaler = None

    started = _now()
    model.train()
    max_train_steps = int(spec.get("max_train_steps") or 50)
    step = 0
    early_patience = int(spec.get("early_stopping_patience") or 0)
    best_val = None
    no_improve = 0
    grad_clip = float(spec.get("grad_clip_norm") or 0.0)
    for epoch in range(epochs):
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True) if hasattr(yb, "to") else yb
            opt.zero_grad()
            if use_amp and scaler is not None:
                try:
                    from torch.cuda.amp import autocast  # type: ignore
                except Exception:
                    autocast = None  # type: ignore
            else:
                autocast = None  # type: ignore
            if autocast is not None:
                with autocast():
                    logits = model(xb)
                    if task == "regression":
                        import torch as _torch  # type: ignore
                        yb_f = yb.float().view(logits.shape[0], -1)
                        if yb_f.shape[1] != logits.shape[1]:
                            want = logits.shape[1]
                            yb_f = yb_f[:, :want] if yb_f.shape[1] > want else _torch.nn.functional.pad(yb_f, (0, want - yb_f.shape[1]))
                        loss = loss_fn(logits, yb_f)
                    else:
                        loss = loss_fn(logits, yb)
            else:
                logits = model(xb)
                if task == "regression":
                    # Align shapes
                    import torch as _torch  # type: ignore
                    yb_f = yb.float().view(logits.shape[0], -1)
                    if yb_f.shape[1] != logits.shape[1]:
                        # best-effort broadcast/clip
                        want = logits.shape[1]
                        yb_f = yb_f[:, :want] if yb_f.shape[1] > want else _torch.nn.functional.pad(yb_f, (0, want - yb_f.shape[1]))
                    loss = loss_fn(logits, yb_f)
                else:
                    loss = loss_fn(logits, yb)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    try:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    except Exception:
                        pass
                scaler.step(opt)
                try:
                    scaler.update()
                except Exception:
                    pass
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    except Exception:
                        pass
                opt.step()
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass
            step += 1
            # Optional progress logging
            try:
                log_interval = int(spec.get("log_interval", 0) or 0)
            except Exception:
                log_interval = 0
            if log_interval and (step % log_interval == 0) and is_verbose():
                try:
                    vprint(f"step={step} loss={float(loss.item()):.4f} lr={float(opt.param_groups[0].get('lr', 0.0)):.2e}")
                except Exception:
                    vprint(f"step={step} loss={float(loss)}")
            if step >= max_train_steps:
                break
        # Early stopping (epoch-level)
        if early_patience > 0:
            try:
                model.eval()
                if task == "regression":
                    import torch as _torch
                    se_sum_e, n_val_e = 0.0, 0
                    with torch.no_grad():
                        for xb, yb in dl_val:
                            xb = xb.to(device)
                            y = (yb if not hasattr(yb, "to") else yb).detach().cpu().float()
                            yhat = model(xb).detach().cpu().float()
                            se_sum_e += float(((yhat.view_as(y) - y) ** 2).sum().item())
                            n_val_e += int(y.numel())
                    cur = se_sum_e / n_val_e if n_val_e else float("inf")
                    better = (best_val is None) or (cur < best_val)
                else:
                    correct_e, total_e = 0, 0
                    with torch.no_grad():
                        for xb, yb in dl_val:
                            xb = xb.to(device)
                            yb = yb.to(device)
                            pred = model(xb).argmax(dim=1)
                            correct_e += (pred == yb).sum().item()
                            total_e += yb.numel()
                    cur = float(correct_e) / float(total_e) if total_e else 0.0
                    better = (best_val is None) or (cur > best_val)
                if better:
                    best_val = cur
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
                if no_improve >= early_patience:
                    break
            except Exception:
                pass
        if step >= max_train_steps:
            break

    # Validate (val split)
    model.eval()
    if task == "regression":
        import torch as _torch  # type: ignore
        se_sum = 0.0
        ae_sum = 0.0
        n_val = 0
        y_all = []
        yhat_all = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                logits = model(xb)
                yhat = logits.detach().cpu().float().view(logits.shape[0], -1)
                y = (yb if not hasattr(yb, "to") else yb).detach().cpu().float().view(yhat.shape[0], -1)
                y_all.append(y)
                yhat_all.append(yhat)
                se_sum += float(((yhat - y) ** 2).sum().item())
                ae_sum += float((yhat - y).abs().sum().item())
                n_val += int(y.numel())
        mse = se_sum / n_val if n_val else 0.0
        mae = ae_sum / n_val if n_val else 0.0
        rmse = mse ** 0.5
        try:
            y_cat = _torch.cat(y_all, dim=0)
            yhat_cat = _torch.cat(yhat_all, dim=0)
            y_mean = y_cat.mean()
            ss_res = ((yhat_cat - y_cat) ** 2).sum()
            ss_tot = ((y_cat - y_mean) ** 2).sum()
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)).item())
        except Exception:
            r2 = 0.0
        val_acc = 0.0  # keep interface for downstream
    else:
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
    if task == "regression":
        import torch as _torch  # type: ignore
        se_sum_t = 0.0
        ae_sum_t = 0.0
        n_test = 0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb = xb.to(device)
                logits = model(xb)
                yhat = logits.detach().cpu().float().view(logits.shape[0], -1)
                y = (yb if not hasattr(yb, "to") else yb).detach().cpu().float().view(yhat.shape[0], -1)
                se_sum_t += float(((yhat - y) ** 2).sum().item())
                ae_sum_t += float((yhat - y).abs().sum().item())
                n_test += int(y.numel())
        mse_t = se_sum_t / n_test if n_test else 0.0
        mae_t = ae_sum_t / n_test if n_test else 0.0
        rmse_t = mse_t ** 0.5
        test_acc = 0.0
    else:
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
    # Build result with task-aware metrics
    metrics: Dict[str, Any]
    if task == "regression":
        metrics = {
            "val_accuracy": val_acc,  # kept for interface compatibility (0.0)
            "test_accuracy": test_acc,  # kept for interface compatibility (0.0)
            "val_mse": mse,
            "val_mae": mae,
            "val_rmse": rmse,
            "val_r2": r2,
            "test_mse": mse_t,
            "test_mae": mae_t,
            "test_rmse": rmse_t,
        }
    else:
        metrics = {"val_accuracy": val_acc, "test_accuracy": test_acc}

    res = {
        "mode": "real",
        "reason": "ok",
        "started": started,
        "finished": finished,
        "metrics": metrics,
        "meta": {
            "device": device,
            "num_classes": num_classes,
            "seed": seed,
            "task": task,
        },
    }
    if is_verbose():
        if task == "regression":
            vprint(f"Completed real run (regression): val_mse={mse:.4f} val_r2={r2:.4f} device={device}")
        else:
            vprint(f"Completed real run: val_acc={val_acc:.4f} test_acc={test_acc:.4f} device={device}")
    return res


def run_experiment(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to run a minimal real experiment if torch+dataset present; otherwise return a stub
    result with explicit reason. This keeps the pipeline end-to-end even on limited setups.
    """
    try:
        return _run_real(spec)
    except Exception as exc:  # pragma: no cover
        res = {
            "mode": "stub",
            "reason": f"Exception in real runner: {exc}",
            "started": _now(),
            "finished": _now(),
            "metrics": {"val_accuracy": 0.0},
        }
        if is_verbose():
            vprint(res["reason"])
        return res
