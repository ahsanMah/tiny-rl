from __future__ import annotations

import math
import time

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


def format_param_table(model: nn.Module, *, sort: bool = False) -> str:
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


class ActionEmbedding(nn.Module):
    """Embeds actions into a fixed-size vector.

    Categorical actions (`num_actions` set) use an embedding lookup; continuous
    actions (`action_dim` set) use a linear projection. Exactly one of the two
    must be provided.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_actions: int | None = None,
        action_dim: int | None = None,
    ):
        super().__init__()
        if (num_actions is None) == (action_dim is None):
            raise ValueError(
                "Specify exactly one of num_actions (categorical) or action_dim (continuous)"
            )

        self.categorical = num_actions is not None
        if self.categorical:
            self.embed = nn.Embedding(num_actions, embed_dim)
        else:
            self.embed = nn.Linear(action_dim, embed_dim)

    def __call__(self, actions: mx.array) -> mx.array:
        if self.categorical:
            return self.embed(actions)
        return self.embed(actions.astype(mx.float32))


class ConvResBlock3D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_embed_dim: int | None = None
    ):
        super().__init__()

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
        self.norm1 = nn.RMSNorm(out_channels)
        self.norm2 = nn.RMSNorm(out_channels)
        zero_init = nn.init.constant(0.0)

        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if time_embed_dim is not None:
            self.to_film = nn.Linear(time_embed_dim, out_channels * 2)
            self.to_film.weight = zero_init(self.to_film.weight)
            self.to_film.bias = zero_init(self.to_film.bias)

        self.conv2.weight = zero_init(self.conv2.weight)
        self.conv2.bias = zero_init(self.conv2.bias)

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
        x = self.shortcut(x)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h)
        h = self._apply_film(h, t_emb, self.to_film)
        h = self.conv2(self.act(h))
        return h + x


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
        self.conv = ConvResBlock3D(
            out_channels, out_channels, time_embed_dim=time_embed_dim
        )

    def __call__(
        self, x: mx.array, skip: mx.array, t_emb: mx.array | None = None
    ) -> mx.array:
        x = self.upsample3d(x)
        x = self.up_conv(x)
        x = (x + skip) * (2**-0.5)  # Normalize to prevent variance explosion
        return self.conv(x, t_emb)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        """Implemenets MultiQuery Attention"""
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm = nn.RMSNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim // num_heads)
        self.to_v = nn.Linear(context_dim, dim // num_heads)
        self.to_out = nn.Linear(dim, dim)

        zero_init = nn.init.constant(0.0)
        self.to_out.weight = zero_init(self.to_out.weight)
        self.to_out.bias = zero_init(self.to_out.bias)

    def __call__(
        self, x: mx.array, context: mx.array, mask: str | None = None
    ) -> mx.array:
        num_heads = self.num_heads
        q = self.to_q(self.norm(x))
        k = self.to_k(context)
        v = self.to_v(context)
        # MLX SDPA expects (B, N_heads, T_seq, D_head); rearrange from (B, T, D).
        q = mx.unflatten(q, -1, (num_heads, -1)).transpose(0, 2, 1, 3)
        k = mx.unflatten(k, -1, (1, -1)).transpose(0, 2, 1, 3)
        v = mx.unflatten(v, -1, (1, -1)).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        # (B, H, T, D_head) -> (B, T, H*D_head)
        out = mx.flatten(out.transpose(0, 2, 1, 3), start_axis=-2)
        out = self.to_out(out)

        return out + x


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden_dim = dim * mult
        self.norm = nn.RMSNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, dim)

        zero_init = nn.init.constant(0.0)
        self.out_proj.weight = zero_init(self.out_proj.weight)
        self.out_proj.bias = zero_init(self.out_proj.bias)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.out_proj(self.act(self.in_proj(self.norm(x))))
        return h + x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.self_attn = CrossAttention(dim, context_dim=dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(
            dim, context_dim=context_dim, num_heads=num_heads
        )
        self.ff = FeedForward(dim, mult=ff_mult)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = self.self_attn(x, context=x)
        if context is not None:
            x = self.cross_attn(x, context=context)
        return self.ff(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        conv_block: nn.Module = ConvResBlock3D,
        num_actions: int = 1,
        max_context_size: int = 3,
        num_transformer_blocks: int = 2,
    ):
        super().__init__()
        time_embed_dim = base_channels * 4
        self.max_context_size = max_context_size
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.context_embed = ActionEmbedding(time_embed_dim, num_actions=num_actions)

        self.res1 = conv_block(
            in_channels, base_channels, time_embed_dim=time_embed_dim
        )
        self.down1 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.res2 = conv_block(
            base_channels, base_channels * 2, time_embed_dim=time_embed_dim
        )
        self.down2 = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        self.res3 = conv_block(
            base_channels * 2, base_channels * 4, time_embed_dim=time_embed_dim
        )

        self.mid_transformer_blocks = [
            TransformerBlock(
                base_channels * 4,
                context_dim=time_embed_dim,
                num_heads=4,
                ff_mult=4,
            )
            for _ in range(num_transformer_blocks)
        ]

        self.mid_conv = conv_block(
            base_channels * 4, base_channels * 4, time_embed_dim=time_embed_dim
        )

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

    def __call__(self, x: mx.array, t: mx.array, context: mx.array) -> mx.array:

        if t.ndim == 0:
            t = mx.broadcast_to(t, (x.shape[0],))
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]

        t_emb = self.time_embed(t)
        context = self.context_embed(context)
        action_emb = context[:, -1, :]
        time_context = (t_emb + action_emb) * 0.5

        x1 = self.res1(x, t_emb)
        x2 = self.res2(self.down1(x1), time_context)
        x3 = self.res3(self.down2(x2), time_context)

        b, s, h, w, c = x3.shape
        xmid = x3.reshape(b, s * h * w, c)
        for block in self.mid_transformer_blocks:
            xmid = block(xmid, context=context)
        xmid = mx.reshape(xmid, (b, s, h, w, c))

        xmid = self.mid_conv(xmid, time_context)

        x = self.up2(xmid, x2, time_context)
        x = self.up1(x, x1, time_context)

        return self.out_conv(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2, base_channels=16, num_actions=3)
    print_param_table(model)
    x = mx.random.normal((8, 4, 32, 32, 1))
    t = mx.ones((8, 1))
    a = mx.ones((8, 4), dtype=mx.uint8)
    print(f"Testing forward pass with input shape {x.shape}")

    y = model(x, t, a)
    mx.eval(y)
    print("input:", x.shape, "output:", y.shape)

    # quick timing (includes compute sync with mx.eval)
    num_runs = 20
    start = time.perf_counter()
    for _ in range(num_runs):
        y = model(x, t, a)
        mx.eval(y)
    elapsed = time.perf_counter() - start
    print(f"avg forward time over {num_runs} runs: {elapsed * 1000 / num_runs:.2f} ms")
