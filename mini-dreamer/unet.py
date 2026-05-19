from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _is_mx_array(node: object) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _iter_param_tree(tree: object, prefix: str = ""):
    if _is_mx_array(tree):
        yield prefix, tree
        return

    if isinstance(tree, dict):
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_param_tree(value, name)
        return

    if isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            name = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_param_tree(value, name)
        return


def format_param_table(model: nn.Module, *, sort: bool = True) -> str:
    rows = []
    for name, param in _iter_param_tree(model.parameters()):
        shape = tuple(param.shape)
        count = int(math.prod(shape)) if shape else 1
        rows.append((name, shape, count))

    if sort:
        rows.sort(key=lambda row: row[0])

    if rows:
        name_width = max(len("Parameter"), max(len(name) for name, _, _ in rows))
        shape_width = max(len("Shape"), max(len(str(shape)) for _, shape, _ in rows))
        count_width = max(len("Count"), max(len(f"{count:,}") for _, _, count in rows))
    else:
        name_width = len("Parameter")
        shape_width = len("Shape")
        count_width = len("Count")

    lines = [
        f"{'Parameter'.ljust(name_width)}  {'Shape'.ljust(shape_width)}  {'Count'.rjust(count_width)}",
        f"{'-' * name_width}  {'-' * shape_width}  {'-' * count_width}",
    ]

    for name, shape, count in rows:
        lines.append(
            f"{name.ljust(name_width)}  {str(shape).ljust(shape_width)}  {f'{count:,}'.rjust(count_width)}"
        )

    total = sum(count for _, _, count in rows)
    lines.append(f"{'-' * name_width}  {'-' * shape_width}  {'-' * count_width}")
    lines.append(
        f"{'TOTAL'.ljust(name_width)}  {'-'.ljust(shape_width)}  {f'{total:,}'.rjust(count_width)}"
    )
    return "\n".join(lines)


def print_param_table(model: nn.Module) -> None:
    print(format_param_table(model))


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for Fourier embedding."
        self.dim = dim
        half = dim // 2
        self.weight = mx.random.normal((half,)) * scale

    def __call__(self, t: mx.array) -> mx.array:
        if t.ndim == 0:
            t = mx.broadcast_to(t, (1,))

        t = t.astype(mx.float32)
        args = t[:, None] * self.weight[None, :] * (2 * math.pi)
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, scale: float = 1.0):
        super().__init__()
        hidden = dim * hidden_mult
        self.fourier = GaussianFourierEmbedding(dim, scale=scale)
        self.lin1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(hidden, dim)

    def __call__(self, t: mx.array) -> mx.array:
        emb = self.fourier(t)
        return self.lin2(self.act(self.lin1(emb)))


class ConvResBlock3D(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int | None = None):
        super().__init__()
        out_channels: int = in_channels
        self.norm1 = nn.RMSNorm(in_channels)
        self.norm2 = nn.RMSNorm(out_channels)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if time_embed_dim is not None:
            self.to_film = nn.Linear(time_embed_dim, out_channels * 2)

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
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h)
        h = self._apply_film(h, t_emb, self.to_film)
        h = self.conv2(self.act(h))
        return h + x


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int | None = None,
    ):
        super().__init__()
        self.upsample3d = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvResBlock3D(out_channels, time_embed_dim=time_embed_dim)

    def __call__(
        self, x: mx.array, skip: mx.array, t_emb: mx.array | None = None
    ) -> mx.array:
        x = self.upsample3d(x)
        x = self.up_conv(x)
        x = (x + skip) * (2**-0.5)  # Normalize to prevent variance explosion
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

        self.conv_in = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        self.res1 = ConvResBlock3D(base_channels, time_embed_dim=time_embed_dim)
        self.down1 = Downsample3D(base_channels)  # Channels double

        self.res2 = ConvResBlock3D(base_channels * 2, time_embed_dim=time_embed_dim)
        self.down2 = Downsample3D(base_channels * 2)

        self.res3 = ConvResBlock3D(base_channels * 4, time_embed_dim=time_embed_dim)
        self.mid_attn = nn.MultiHeadAttention(base_channels * 4, num_heads=4)
        self.mid_conv = ConvResBlock3D(base_channels * 4, time_embed_dim=time_embed_dim)

        self.up2 = UpBlock3D(
            base_channels * 4,
            base_channels * 2,
            time_embed_dim=time_embed_dim,
        )
        self.up1 = UpBlock3D(
            base_channels * 2,
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
        x = self.conv_in(x)

        x1 = self.res1(x, t_emb)
        x2 = self.res2(self.down1(x1), t_emb)
        x3 = self.res3(self.down2(x2), t_emb)

        b, s, h, w, c = x3.shape
        x3 = mx.reshape(x3, (b, s * h * w, c))
        x3 = self.mid_attn(x3, x3, x3)
        x3 = mx.reshape(x3, (b, s, h, w, c))
        x3 = self.mid_conv(x3, t_emb)

        x = self.up2(x3, x2, t_emb)
        x = self.up1(x, x1, t_emb)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2, base_channels=16)
    print_param_table(model)
    x = mx.random.normal((1, 4, 32, 32, 1))
    t = mx.array([10])
    y = model(x, t)
    print("input:", x.shape, "output:", y.shape)
