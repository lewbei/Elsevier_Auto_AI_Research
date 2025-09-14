try:
    import torchvision.transforms as T
except Exception:
    T = None

class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            self.pipe = T.Compose([
                # Deterministic resize to the requested input size
                T.Resize((224, 224)),
                # Very light, fixed Gaussian blur (fixed sigma for determinism)
                T.GaussianBlur(kernel_size=3, sigma=0.1),
                # Simple horizontal flip augmentation (stochastic)
                T.RandomHorizontalFlip(p=0.5),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)