
try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover
    T = None  # type: ignore


class GeneratedAug:
    def __init__(self):
        if T is None:
            self.pipe = None
        else:
            self.pipe = T.Compose([
                T.ColorJitter(0.1,0.1,0.1,0.05),
                T.RandomRotation(10),
                T.RandomErasing(p=0.25),
                T.GaussianBlur(3)
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)
