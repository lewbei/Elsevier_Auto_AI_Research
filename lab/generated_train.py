import torch
import torch.nn as nn
from torchvision import transforms

def build_train_transforms(input_size):
    """
    Build a small, deterministic-but-useful training transform pipeline.
    Allowed building blocks are used (Resize, RandomHorizontalFlip, ColorJitter,
    RandomRotation, GaussianBlur, ToTensor, RandomErasing).

    Keeps input_size clamped to safe range (96..512).
    """
    # enforce safe range
    input_size = int(max(96, min(int(input_size), 512)))

    tr = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        # RandomErasing operates on tensor images; keep p modest
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3))
    ])
    return tr


def build_model_head(in_features, num_classes):
    """
    Build a compact model head that implements the ProtoAdapterMotion-Light idea
    as a tiny module (defined locally to avoid additional top-level classes):

    - appearance bottleneck: projects frozen backbone embedding (in_features) -> 64-D
    - motion FiLM MLP: motion summary (64-D) -> 2*64 (gamma, beta)
    - FiLM application: modulate bottleneck with gamma/beta
    - up-projection: project modulated 64-D back to a prototype space (in_features)
    - small linear classifier (proto_classifier) is provided for convenience; prototypes
      for episodic ProtoNet computations may be computed externally from the
      "projected" embeddings returned by forward().

    Returned object is an nn.Module instance whose forward(appearance_emb, motion_summary)
    returns a dict with keys: "modulated", "projected", "logits", "gamma", "beta".
    """
    # Keep dimensions small and deterministic:
    bottleneck_dim = 64
    motion_dim = 64

    class AdapterHead(nn.Module):
        def __init__(self, in_features, num_classes, bottleneck_dim=64, motion_dim=64):
            super().__init__()
            # appearance bottleneck: in_features -> 64
            self.bottleneck = nn.Linear(in_features, bottleneck_dim)

            # motion MLP: motion_dim -> 2 * bottleneck_dim (gamma and beta)
            # single hidden layer kept tiny to satisfy parameter budget
            self.film_mlp = nn.Sequential(
                nn.Linear(motion_dim, motion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(motion_dim, 2 * bottleneck_dim)
            )

            # up-project modulated bottleneck back to prototype space (in_features)
            self.upproj = nn.Linear(bottleneck_dim, in_features)

            # small classifier head (optional convenience, prototypes are expected
            # to be computed episodically outside this module for ProtoNet loss).
            self.proto_classifier = nn.Linear(in_features, num_classes)

            # initialize weights small and stable
            nn.init.normal_(self.bottleneck.weight, std=0.01)
            nn.init.zeros_(self.bottleneck.bias)
            nn.init.normal_(self.film_mlp[0].weight, std=0.01)
            nn.init.zeros_(self.film_mlp[0].bias)
            nn.init.normal_(self.film_mlp[2].weight, std=0.01)
            nn.init.zeros_(self.film_mlp[2].bias)
            nn.init.normal_(self.upproj.weight, std=0.01)
            nn.init.zeros_(self.upproj.bias)
            nn.init.normal_(self.proto_classifier.weight, std=0.01)
            nn.init.zeros_(self.proto_classifier.bias)

        def forward(self, appearance_emb, motion_summary):
            """
            appearance_emb: Tensor [B, in_features] (e.g., pooled ResNet-18 output)
            motion_summary: Tensor [B, motion_dim] (precomputed per-crop absolute frame-diff pooled to 64-D)

            Returns a dict:
              - "modulated": [B, bottleneck_dim] FiLM-modulated bottleneck embedding
              - "projected": [B, in_features] up-projected embedding usable for prototype computation
              - "logits": [B, num_classes] optional linear logits from projected embeddings
              - "gamma", "beta": FiLM parameters [B, bottleneck_dim] each
            """
            z = self.bottleneck(appearance_emb)  # [B, bottleneck_dim]
            film_params = self.film_mlp(motion_summary)  # [B, 2*bottleneck_dim]
            gamma, beta = film_params.chunk(2, dim=-1)
            z_mod = gamma * z + beta
            up = self.upproj(z_mod)
            logits = self.proto_classifier(up)
            return {"modulated": z_mod, "projected": up, "logits": logits, "gamma": gamma, "beta": beta}

    return AdapterHead(int(in_features), int(num_classes), bottleneck_dim=bottleneck_dim, motion_dim=motion_dim)


def update_spec(spec):
    """
    Sanitize and clamp the incoming spec to safe defaults required by the
    experimental budget and acceptance criteria.

    - lr clamped to [1e-5, 1e-1]
    - max_train_steps clamped to [10, 1000]
    - input_size clamped to [96, 512]
    - ensures episodes_per_step and validate_every_steps default values
      (episodes_per_step=4, validate_every_steps=10) required by the brief.
    - preserves other keys (including novelty description and sssa params)
    """
    # Copy to avoid mutating caller object
    out = dict(spec)

    if "lr" in out:
        try:
            out["lr"] = float(out["lr"])
        except Exception:
            out["lr"] = 0.0003
    out["lr"] = float(max(1e-5, min(out.get("lr", 1e-3), 1e-1)))

    if "max_train_steps" in out:
        try:
            out["max_train_steps"] = int(out["max_train_steps"])
        except Exception:
            out["max_train_steps"] = 100
    out["max_train_steps"] = int(max(10, min(out.get("max_train_steps", 100), 1000)))

    if "input_size" in out:
        try:
            out["input_size"] = int(out["input_size"])
        except Exception:
            out["input_size"] = 224
    out["input_size"] = int(max(96, min(out.get("input_size", 224), 512)))

    # enforce episodes_per_step and validate cadence required by the design
    out.setdefault("episodes_per_step", 4)
    try:
        out["episodes_per_step"] = int(out["episodes_per_step"])
    except Exception:
        out["episodes_per_step"] = 4

    out.setdefault("validate_every_steps", 10)
    try:
        out["validate_every_steps"] = int(out["validate_every_steps"])
    except Exception:
        out["validate_every_steps"] = 10

    # ensure use_generated_head is present (brief expects generated head behavior)
    out.setdefault("use_generated_head", True)

    return out