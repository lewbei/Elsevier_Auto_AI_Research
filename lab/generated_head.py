
import torch
import torch.nn as nn
from typing import Any


class GeneratedHead(nn.Module):
    """Minimal classification head: optional dropout followed by Linear.

    Parameters
    - in_features: input feature dimension
    - num_classes: number of output classes
    - p: dropout probability
    """

    def __init__(self, in_features: int, num_classes: int, p: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)
