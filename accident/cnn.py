"""Frame-level accident CNN.

Parameter names (``down1.conv1``, ``dens1.line1``, ...) are kept stable so that checkpoints trained
with earlier versions of this repository load without conversion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = (32, 64, 128, 256, 512)


class DownBlock(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU -> MaxPool(2)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool1(self.relu(self.batch1(self.conv1(x))))


class MLPHead(nn.Module):
    """Linear(in, 2*in) -> ReLU -> Linear(2*in, in) -> ReLU -> Linear(in, out)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.line1 = nn.Linear(in_ch, 2 * in_ch)
        self.line2 = nn.Linear(2 * in_ch, in_ch)
        self.line3 = nn.Linear(in_ch, out_ch)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.line3(self.relu2(self.line2(self.relu1(self.line1(x)))))


class AccidentCNN(nn.Module):
    """``depth`` down-sampling blocks with dropout, global average pooling and an MLP head (2 classes)."""

    def __init__(self, depth=4, num_classes=2, dropout=0.5):
        super().__init__()
        if not 1 <= depth <= len(CHANNELS):
            raise ValueError(f"depth must be in [1, {len(CHANNELS)}]")
        self.depth = depth
        in_ch = 3
        for i, out_ch in enumerate(CHANNELS[:depth], start=1):
            self.add_module(f"down{i}", DownBlock(in_ch, out_ch))
            in_ch = out_ch
        self.out_channels = in_ch
        self.dropout = nn.Dropout(dropout)
        self.dens1 = MLPHead(in_ch, num_classes)

    def blocks(self):
        return [getattr(self, f"down{i}") for i in range(1, self.depth + 1)]

    def features(self, x):
        """Feature map after the last block, ``[B, out_channels, H / 2^depth, W / 2^depth]``."""
        for block in self.blocks():
            x = self.dropout(block(x))
        return x

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.features(x), 1)
        return self.dens1(torch.flatten(x, 1))


def load_cnn(checkpoint_path, depth=4, device="cpu"):
    model = AccidentCNN(depth=depth)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()
