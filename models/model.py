import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + 0.2 * self.conv2(self.act(self.conv1(x)))


class DefendSRLite(nn.Module):
    """Small x2 grayscale restoration network.

    The model works mostly at low resolution for speed, predicts a residual over
    bicubic x2 upsampling, and includes a lightweight confidence head.  The
    checkpoint shipped with this repo is trained only on procedurally generated
    semiconductor-like patterns, so it should be treated as a hackathon
    prototype / fallback rather than a production model.
    """

    def __init__(self, channels: int = 24, blocks: int = 4):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.fuse = nn.Conv2d(channels, channels, 3, padding=1)
        self.restore_head = nn.Conv2d(channels, 4, 3, padding=1)
        self.confidence_head = nn.Conv2d(channels, 4, 3, padding=1)

    def forward(self, x, return_confidence: bool = False):
        # Bound extreme speckle outliers without destroying the expected
        # overshoot behaviour around [0, 1].
        x_safe = torch.clamp(x, -0.5, 1.5)
        base = F.interpolate(x_safe, scale_factor=2.0, mode="bicubic", align_corners=False)

        feat0 = F.gelu(self.head(x_safe))
        feat = self.body(feat0)
        feat = feat0 + 0.2 * self.fuse(feat)

        residual = F.pixel_shuffle(self.restore_head(feat), 2)
        restored = torch.clamp(base + 0.20 * residual, 0.0, 1.0)

        if not return_confidence:
            return restored

        confidence = torch.sigmoid(F.pixel_shuffle(self.confidence_head(feat), 2))
        return restored, confidence
