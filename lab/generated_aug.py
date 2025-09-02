try:
    import torchvision.transforms as T
except Exception:
    T = None

class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            # Short, mostly deterministic pipeline aligned with spectral-focused preprocessing:
            # Note: Do NOT include T.ToTensor() here; the runner appends it.
            # - Resize to model input size (keeps spatial scale consistent for DCT/block processing downstream)
            # - Gentle deterministic Gaussian blur to slightly emphasize low-frequency content
            self.pipe = T.Compose([
                T.Resize((256, 256)),
                T.GaussianBlur(kernel_size=3, sigma=0.1),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)
