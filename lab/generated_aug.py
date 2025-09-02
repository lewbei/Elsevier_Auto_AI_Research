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
                T.Resize((256, 256)),
                T.GaussianBlur(kernel_size=5, sigma=1.0),
                T.RandomHorizontalFlip(p=0.5),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)