from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    if t.ndim == 0:
        t = mx.broadcast_to(t, (1,))

    t = t.astype(mx.float32)
    half = dim // 2

    if half == 0:
        return mx.zeros((t.shape[0], dim), dtype=t.dtype)

    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half) / half)
    args = t[:, None] * freqs[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros((t.shape[0], 1), dtype=emb.dtype)], axis=-1)

    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.lin1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden, dim)
        self.dim = dim

    def __call__(self, t: mx.array) -> mx.array:
        emb = timestep_embedding(t, self.dim)
        return self.lin2(self.act(self.lin1(emb)))


class ConvBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_embed_dim: int | None = None
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(in_channels)
        self.norm2 = nn.RMSNorm(out_channels)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if time_embed_dim is not None:
            self.to_film1 = nn.Linear(time_embed_dim, in_channels * 2)
            self.to_film2 = nn.Linear(time_embed_dim, out_channels * 2)
        else:
            self.to_film1 = None
            self.to_film2 = None

    def _apply_film(
        self, h: mx.array, t_emb: mx.array | None, proj: nn.Linear | None
    ) -> mx.array:
        if t_emb is None or proj is None:
            return h

        b = h.shape[0]
        c = h.shape[-1]
        scale_shift = proj(t_emb)
        scale = scale_shift[:, :c]
        shift = scale_shift[:, c:]
        scale = mx.reshape(scale, (b, 1, 1, 1, c))
        shift = mx.reshape(shift, (b, 1, 1, 1, c))
        return h * (1 + scale) + shift

    def __call__(self, x: mx.array, t_emb: mx.array | None = None) -> mx.array:
        h = self.norm1(x)
        h = self._apply_film(h, t_emb, self.to_film1)
        x = self.conv1(self.act(h))

        h = self.norm2(x)
        h = self._apply_film(h, t_emb, self.to_film2)
        x = self.conv2(self.act(h))
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        self.upsample3d = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock3D(
            out_channels + skip_channels, out_channels, time_embed_dim=time_embed_dim
        )

    def __call__(
        self, x: mx.array, skip: mx.array, t_emb: mx.array | None = None
    ) -> mx.array:
        x = self.upsample3d(x)
        x = self.up_conv(x)
        x = mx.concatenate([x, skip], axis=-1)
        return self.conv(x, t_emb)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()
        time_embed_dim = base_channels * 4
        self.time_embed = TimeEmbedding(time_embed_dim)

        self.enc1 = ConvBlock3D(
            in_channels, base_channels, time_embed_dim=time_embed_dim
        )
        self.down1 = Downsample3D(base_channels)

        self.enc2 = ConvBlock3D(
            base_channels, base_channels * 2, time_embed_dim=time_embed_dim
        )
        self.down2 = Downsample3D(base_channels * 2)

        self.enc3 = ConvBlock3D(
            base_channels * 2, base_channels * 4, time_embed_dim=time_embed_dim
        )

        self.mid_conv1 = ConvBlock3D(
            base_channels * 4, base_channels * 4, time_embed_dim=time_embed_dim
        )
        self.mid_attn = nn.MultiHeadAttention(base_channels * 4, num_heads=4)
        self.mid_conv2 = ConvBlock3D(
            base_channels * 4, base_channels * 4, time_embed_dim=time_embed_dim
        )

        self.up2 = UpBlock3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            time_embed_dim=time_embed_dim,
        )
        self.up1 = UpBlock3D(
            base_channels * 2,
            base_channels,
            base_channels,
            time_embed_dim=time_embed_dim,
        )

        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array, t: mx.array | None = None) -> mx.array:
        if t is None:
            t = mx.zeros((x.shape[0],), dtype=x.dtype)
        elif t.ndim == 0:
            t = mx.broadcast_to(t, (x.shape[0],))
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]

        t_emb = self.time_embed(t)

        x1 = self.enc1(x, t_emb)
        x2 = self.enc2(self.down1(x1), t_emb)
        x3 = self.enc3(self.down2(x2), t_emb)

        x3 = self.mid_conv1(x3, t_emb)
        b, s, h, w, c = x3.shape

        x3 = mx.reshape(x3, (b, s * h * w, c))
        x3 = self.mid_attn(x3, x3, x3)
        x3 = mx.reshape(x3, (b, s, h, w, c))
        x3 = self.mid_conv2(x3, t_emb)

        x = self.up2(x3, x2, t_emb)
        x = self.up1(x, x1, t_emb)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2, base_channels=16)
    x = mx.random.normal((1, 16, 32, 32, 1))
    t = mx.array([10])
    y = model(x, t)
    print("input:", x.shape, "output:", y.shape)
