# training_hooks.py
# Small training hooks file with a minimal multimodal fusion head that
# implements learnable per-modality gating as the novelty component.
# The file exposes exactly three functions:
# - build_train_transforms(input_size)
# - build_model_head(in_features, num_classes)
# - update_spec(spec)

import torch
import torch.nn as nn
import torchvision.transforms as T

def build_train_transforms(input_size):
    """
    Build a small set of training transforms.
    Keep transforms simple and deterministic within randomness bounds.
    Allowed transforms used: Resize, RandomHorizontalFlip, ColorJitter, ToTensor.
    """
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.ToTensor(),
    ])


def build_model_head(in_features, num_classes):
    """
    Build a fusion head module that:
    - Holds two learnable scalar parameters s_closeup and s_hfus (initialized to 0).
      g = sigmoid(s) yields initial g=0.5.
    - Projects HFUS pooled features to the same pooled-dim (in_features).
    - Computes fused pooled = (g_c * pool_c + g_h * pool_h) / max(g_c + g_h, 0.01).
    - Adds an L1 regularizer on the gates (coeff = 1e-3).
    - Clamps gate values to [0.05, 0.95] for the first 30 steps to stabilize training.
    The returned module forward accepts (pool_closeup, pool_hfus, step=0) and returns
    (logits, reg_loss) where reg_loss is the L1 regularization scalar tensor.
    """
    class FusionHead(nn.Module):
        def __init__(self, in_features, num_classes, clamp_steps=30, l1_coeff=1e-3):
            super().__init__()
            # scalar parameters (initialized zero so sigmoid(0)=0.5)
            self.s_closeup = nn.Parameter(torch.zeros(1))
            self.s_hfus = nn.Parameter(torch.zeros(1))
            # project HFUS pooled vector to the same dim (if needed)
            self.hfus_proj = nn.Linear(in_features, in_features)
            # simple linear classifier on the fused pooled vector
            self.classifier = nn.Linear(in_features, num_classes)
            # configuration
            self.clamp_steps = int(clamp_steps)
            self.l1_coeff = float(l1_coeff)

        def forward(self, pool_closeup, pool_hfus, step=0):
            """
            pool_closeup: Tensor [B, in_features] pooled from ResNet18 global pool
            pool_hfus:    Tensor [B, in_features] pooled from UNet-small encoder (before projection)
            step:         int training step used to decide clamping
            Returns: (logits [B, num_classes], reg_loss scalar tensor)
            """
            # project HFUS pooled features
            pool_hfus_proj = self.hfus_proj(pool_hfus)

            # compute gates
            g_c = torch.sigmoid(self.s_closeup)  # scalar tensor
            g_h = torch.sigmoid(self.s_hfus)

            # clamp gates during early steps to stabilize (clamp in value space)
            if int(step) < self.clamp_steps:
                g_c = torch.clamp(g_c, min=0.05, max=0.95)
                g_h = torch.clamp(g_h, min=0.05, max=0.95)

            # safe denominator
            denom = torch.clamp(g_c + g_h, min=0.01)

            # fused pooled vector (broadcasting scalar gates over batch)
            fused = (g_c * pool_closeup + g_h * pool_hfus_proj) / denom

            logits = self.classifier(fused)

            # L1 regularizer on gate values (use the post-sigmoid values)
            reg_loss = self.l1_coeff * (g_c.abs() + g_h.abs())

            # return logits and the regularization term (scalar tensor)
            return logits, reg_loss

    return FusionHead(in_features, num_classes)


def update_spec(spec):
    """
    Update the experiment spec with concrete novelty hyperparameters and ensure
    safe ranges remain respected. Modifies the spec in-place and also returns it.
    """
    # Ensure essential fields exist
    spec = dict(spec)  # shallow copy to avoid surprising side-effects if needed

    nc = spec.get("novelty_component", {})
    nc = dict(nc)
    nc.setdefault("description", "Learnable per-modality gating (close-up + HFUS).")
    # Clamp duration and L1 coefficient used by the head
    nc["gating_clamp_steps"] = 30
    nc["gating_l1_coeff"] = 1e-3
    nc["enabled"] = True
    spec["novelty_component"] = nc

    # Safety: ensure ranges are within allowed constraints
    # input_size: 96..512
    if "input_size" in spec:
        spec["input_size"] = int(max(96, min(512, int(spec["input_size"]))))
    # learning rate safe range 1e-5 .. 1e-1
    if "lr" in spec:
        spec["lr"] = float(max(1e-5, min(1e-1, float(spec["lr"]))))
    # max_train_steps: 10..1000
    if "max_train_steps" in spec:
        spec["max_train_steps"] = int(max(10, min(1000, int(spec["max_train_steps"]))))
    # keep batch_size positive
    if "batch_size" in spec:
        spec["batch_size"] = int(max(1, int(spec["batch_size"])))

    return spec