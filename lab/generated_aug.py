try:
    import torchvision.transforms as T
except Exception:
    T = None

class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            # Short, deterministic pipeline chosen to reflect the described edge-prior
            # preprocessing: downsample to ResNet layer2 spatial dims (28x28),
            # apply a fixed Gaussian smoothing (proxy for multi-scale edge ops),
            # then convert to tensor.
            self.pipe = T.Compose([
                T.Resize((28, 28)),
                T.GaussianBlur(kernel_size=5, sigma=1.0),
                T.ToTensor(),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)