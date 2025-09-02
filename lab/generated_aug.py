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
                T.Resize((224, 224)),
                T.ToTensor(),
            ])

    def __call__(self, x):
        if self.pipe is None:
            return x
        return self.pipe(x)