try:
    import torchvision.transforms as T
except Exception:
    T = None

class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            # Keep the list short and deterministic where possible.
            # Resize to the input size referenced in the spec, apply a small fixed blur,
            # then convert to tensor.
            self.pipe = T.Compose([
                T.Resize((224, 224)),
                T.GaussianBlur(kernel_size=3, sigma=0.1),
                T.ToTensor(),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)