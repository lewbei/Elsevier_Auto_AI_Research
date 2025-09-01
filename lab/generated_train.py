import torchvision.transforms as T
import torch.nn as nn

def build_train_transforms(input_size):
    """
    Build a compact, deterministic training pipeline.
    Allowed transforms: Resize, RandomHorizontalFlip, ColorJitter, ToTensor.
    Keep augmentations small for medical images.
    """
    # clamp input_size to safe range (96..512)
    if input_size is None:
        input_size = 224
    input_size = int(max(96, min(512, input_size)))

    transforms = T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
    ])
    return transforms


def build_model_head(in_features, num_classes):
    """
    Small classification head suitable for EfficientNet pooled features.
    Uses only torch.nn layers.
    """
    # ensure integer intermediate size
    hidden = max(1, in_features // 2)
    head = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(hidden, num_classes),
    )
    return head


def update_spec(spec):
    """
    Update the experiment spec to reflect the 'Agreement-weighted CAM pseudo-box distillation'
    novelty. This function makes minimal, deterministic edits:
      - Enforce short-budget training settings (epochs, steps, batch, optimizer, lr).
      - Ensure input_size is in [96, 512].
      - Insert a concise novelty_component describing the agreement-weighting procedure
        and its numeric hyperparameters used to produce pseudo-boxes.
    Returns the updated spec dict.
    """
    # ensure spec is a dict-like object
    if spec is None:
        spec = {}

    # Enforce safe input size range
    input_size = spec.get("input_size", 224)
    input_size = int(max(96, min(512, input_size)))
    spec["input_size"] = input_size

    # Short-budget training settings
    spec["epochs"] = 1
    spec["batch_size"] = 2
    # clamp learning rate to a safe range (1e-5 .. 1e-1)
    lr = float(spec.get("lr", 0.001))
    lr = max(1e-5, min(1e-1, lr))
    spec["lr"] = lr
    # bounded train steps
    max_steps = int(spec.get("max_train_steps", 100))
    max_steps = max(10, min(1000, max_steps))
    # override to recommended short-budget
    spec["max_train_steps"] = 100

    # optimizer
    spec["optimizer"] = "AdamW"

    # Seed, keep deterministic default if missing
    spec["seed"] = int(spec.get("seed", 42))

    # Update title to reflect the novelty
    spec["title"] = "Novelty: Agreement-weighted CAM → pseudo-box distillation (EfficientNet_b3 + frozen ViT + YOLOv5s)"

    # Add concise, explicit novelty component description and parameters.
    spec["novelty_component"] = {
        "enabled": True,
        "description": (
            "Agreement-weighted CAM pseudo-box distillation. Compute Grad-CAM heatmap "
            "from EfficientNet_b3 classifier. Extract pooled feature vector from EfficientNet_b3 "
            "and compute cosine similarity with a frozen ViT_base_patch16_224 pooled vector. "
            "Agreement is computed as agreement = clip(sigmoid((cos_sim - center)/scale), clip_min, clip_max) "
            "with center=0.5, scale=0.1, clip_min=0.1, clip_max=0.9. Multiply Grad-CAM by agreement, "
            "threshold heatmap at cam_threshold to obtain binary mask, remove connected components whose area "
            "is below min_area_frac of image area, and convert remaining components into pseudo-boxes. "
            "Assign pseudo-box scores proportional to (agreement * CAM_mass). Train YOLOv5s on these pseudo-boxes. "
            "Primary evaluation: mAP@0.5 versus raw CAM baseline."
        ),
        # explicit numeric hyperparameters used by the procedure
        "params": {
            "cam_threshold": 0.25,
            "min_area_frac": 0.005,
            "agreement_center": 0.5,
            "agreement_scale": 0.1,
            "agreement_clip_min": 0.1,
            "agreement_clip_max": 0.9,
        },
        "models_involved": [
            "efficientnet_b3 (classifier, trainable for CAM)",
            "vit_base_patch16_224 (frozen pooled vector for agreement)",
            "yolov5s (detector trained on pseudo-boxes)"
        ],
        "pseudo_box_scoring": "score ≈ agreement * CAM_mass",
        "eval_metric": "mAP@0.5 (primary)",
        "baseline": "raw CAM → pseudo-box (no agreement weighting)"
    }

    # preserve other user-specified fields where reasonable
    # return the updated spec
    return spec