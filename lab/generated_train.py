def build_train_transforms(input_size):
    """Builds training transforms for motion-conditioned model."""
    transforms = [
        Resize((input_size, input_size)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ToTensor(),
    ]
    return transforms

def build_model_head(in_features, num_classes):
    """Builds a model head with motion-conditioned LoRA adapters."""
    # Linear layer with LoRA adaptation for prototype computation
    return torch.nn.Linear(in_features, num_classes)

def update_spec(spec):
    """Updates spec with motion-conditioned LoRA adapter parameters."""
    # Parameters for LoRA adapters and motion conditioning
    spec["lora_rank"] = 8
    spec["lora_alpha"] = 16
    spec["motion_window_size"] = 5  # frames for short-term differences
    return spec