
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
                T.RandomHorizontalFlip()
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)
