from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(in_channels)
        self.norm2 = nn.RMSNorm(out_channels)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample3d = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def __call__(self, x: mx.array, skip: mx.array) -> mx.array:
        x = self.upsample3d(x)
        x = self.up_conv(x)
        x = mx.concatenate([x, skip], axis=-1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.down1 = Downsample3D(base_channels)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.down2 = Downsample3D(base_channels * 2)

        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)

        self.mid_conv1 = ConvBlock3D(base_channels * 4, base_channels * 4)
        self.mid_attn = nn.MultiHeadAttention(base_channels * 4, num_heads=4)
        self.mid_conv2 = ConvBlock3D(base_channels * 4, base_channels * 4)

        self.up2 = UpBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock3D(base_channels * 2, base_channels, base_channels)

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))

        x3 = self.mid_conv1(x3)
        b, s, h, w, c = x3.shape

        x3 = mx.reshape(x3, (b, s * h * w, c))
        x3 = self.mid_attn(x3, x3, x3)
        x3 = mx.reshape(x3, (b, s, h, w, c))
        x3 = self.mid_conv2(x3)

        x = self.up2(x3, x2)
        x = self.up1(x, x1)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2, base_channels=16)
    x = mx.random.normal((1, 16, 32, 32, 1))
    y = model(x)
    print("input:", x.shape, "output:", y.shape)
