
import torch.nn as nn


class GeneratedHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)
