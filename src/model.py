"""Компактна CNN для класифікації ключових слів за log-Mel спектрограмою."""
from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KWSCNN(nn.Module):
    """Вхід [B, 1, 64, 101] → логіти [B, n_classes]."""

    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, width // 2),      # 16 x 32 x 50
            conv_block(width // 2, width),  # 32 x 16 x 25
            conv_block(width, width * 2),   # 64 x 8 x 12
            conv_block(width * 2, width * 2),  # 64 x 4 x 6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str, device: str = "cpu") -> tuple[KWSCNN, dict]:
    """Завантажує model.pt, збережений train.py: {"state_dict", "config"}."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = KWSCNN(n_classes=len(cfg["classes"]), width=cfg.get("width", 32))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg
