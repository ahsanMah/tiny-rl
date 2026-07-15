"""JAX/Flax NNX port of unet.py (UNet3D and its building blocks).

Layout notes (mirroring vae_jax.py):
- MLX and JAX both use channels-last, so clips stay ``(B, S, H, W, C)`` and
  ``nnx.Conv`` with a 3-tuple kernel is a drop-in for ``nn.Conv3d``.
- MLX RMSNorm defaults to ``eps=1e-5`` vs NNX's ``1e-6``; safetensors doesn't
  carry the epsilon, so it is baked into the constructors here.
- Zero-inits (FiLM proj, residual conv2, attention/FF out projections, reward
  head tail) are reproduced with ``nnx.initializers.zeros_init()``.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from vae_jax import WaveletDownsampleConv, WaveletUpsample

RMS_NORM_EPS = 1e-5  # MLX nn.RMSNorm default; NNX defaults to 1e-6.

def _is_tensor(node: object) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _iter_param_tree(tree: object, prefix: str = ""):
    if _is_tensor(tree):
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

def format_param_table(model: nnx.Module, *, sort: bool = False) -> str:
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

class GaussianFourierEmbedding(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs, scale: float = 1.0):
        assert dim % 2 == 0, "Dimension must be even for Fourier embedding."
        self.dim = dim
        half = dim // 2
        # A parameter (not a constant): MLX stores it in the checkpoint and the
        # optimizer/EMA touch it, so the port must track it as nnx.Param too.
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (half,)) * scale)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (1,))

        t = t.astype(jnp.float32)
        args = t[:, None] * self.weight[None, :] * (2 * math.pi)
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        return emb


class TimeEmbedding(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs, hidden_mult: int = 4, scale: float = 1.0):
        hidden = dim * hidden_mult
        self.fourier = GaussianFourierEmbedding(dim, scale=scale, rngs=rngs)
        self.lin1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.act = nnx.silu
        self.lin2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        emb = self.fourier(t)
        return self.lin2(self.act(self.lin1(emb)))


class ActionEmbedding(nnx.Module):
    """Embeds actions into a fixed-size vector.

    Categorical actions (`num_actions` set) use an embedding lookup; continuous
    actions (`action_dim` set) use a linear projection. Exactly one of the two
    must be provided.

    Categorical embeddings reserve one extra slot (index `num_actions`) as a
    learned NULL action, usable for unconditional generation (CFG dropout) and
    for "no action yet" contexts when extracting policy/value embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_actions: int | None = None,
        action_dim: int | None = None,
        rngs: nnx.Rngs,
    ):
        if (num_actions is None) == (action_dim is None):
            raise ValueError(
                "Specify exactly one of num_actions (categorical) or action_dim (continuous)"
            )

        self.categorical = num_actions is not None
        if self.categorical:
            self.null_action = num_actions
            self.embed = nnx.Embed(num_actions + 1, embed_dim, rngs=rngs)
        else:
            self.null_action = None
            self.embed = nnx.Linear(action_dim, embed_dim, rngs=rngs)

    def __call__(self, actions: jnp.ndarray) -> jnp.ndarray:
        if self.categorical:
            return self.embed(actions)
        return self.embed(actions.astype(jnp.float32))


class ConvResBlock3D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int | None = None,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        zero_init = nnx.initializers.zeros_init()

        self.shortcut = (
            nnx.identity
            if in_channels == out_channels
            else nnx.Conv(
                in_channels, out_channels, kernel_size=(1, 1, 1), dtype=dtype, rngs=rngs
            )
        )
        # Norms compute in fp32 regardless of the block's compute dtype; the
        # following conv casts back down.
        self.norm1 = nnx.RMSNorm(
            out_channels, epsilon=RMS_NORM_EPS, dtype=jnp.float32, rngs=rngs
        )
        self.norm2 = nnx.RMSNorm(
            out_channels, epsilon=RMS_NORM_EPS, dtype=jnp.float32, rngs=rngs
        )

        self.conv1 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            kernel_init=zero_init,
            bias_init=zero_init,
            dtype=dtype,
            rngs=rngs,
        )
        self.act = nnx.silu

        self.to_film = (
            nnx.Linear(
                time_embed_dim,
                out_channels * 2,
                kernel_init=zero_init,
                bias_init=zero_init,
                dtype=dtype,
                rngs=rngs,
            )
            if time_embed_dim is not None
            else None
        )

    def _apply_film(self, h: jnp.ndarray, t_emb: jnp.ndarray | None) -> jnp.ndarray:
        if t_emb is None or self.to_film is None:
            return h

        b = h.shape[0]
        c = h.shape[-1]
        scale_shift = self.to_film(t_emb)
        scale = scale_shift[:, :c].reshape(b, 1, 1, 1, c)
        shift = scale_shift[:, c:].reshape(b, 1, 1, 1, c)
        return h * (1 + scale) + shift

    def __call__(self, x: jnp.ndarray, t_emb: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.shortcut(x)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h)
        h = self._apply_film(h, t_emb)
        h = self.conv2(self.act(h))
        return h + x


def _avg_pool_3d(
    x: jnp.ndarray, kernel_size: int = 3, stride: int = 2, padding: int = 1
) -> jnp.ndarray:
    """``nn.AvgPool3d`` analogue for ``(B, S, H, W, C)``; padded zeros count
    toward the average (count_include_pad=True), matching MLX/PyTorch."""
    window = (1, kernel_size, kernel_size, kernel_size, 1)
    strides = (1, stride, stride, stride, 1)
    pads = ((0, 0), *(((padding, padding),) * 3), (0, 0))
    summed = lax.reduce_window(x, 0.0, lax.add, window, strides, pads)
    return summed / float(kernel_size**3)


def _nearest_upsample_3d(x: jnp.ndarray, scale_factor: int = 2) -> jnp.ndarray:
    """``nn.Upsample(mode='nearest')`` on ``(B, S, H, W, C)``: scales S, H, W."""
    B, S, H, W, C = x.shape
    return jax.image.resize(
        x,
        (B, S * scale_factor, H * scale_factor, W * scale_factor, C),
        method="nearest",
    )


class UpBlock3D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int | None = None,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        self.up_conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(1, 1, 1), dtype=dtype, rngs=rngs
        )
        self.conv = ConvResBlock3D(
            out_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self, x: jnp.ndarray, skip: jnp.ndarray, t_emb: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = _nearest_upsample_3d(x, scale_factor=2)
        x = self.up_conv(x)
        x = (x + skip) * (2**-0.5)  # Normalize to prevent variance explosion
        return self.conv(x, t_emb)


class CrossAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 4,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        """Implements MultiQuery Attention"""
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        zero_init = nnx.initializers.zeros_init()
        self.norm = nnx.RMSNorm(
            dim, epsilon=RMS_NORM_EPS, dtype=jnp.float32, rngs=rngs
        )
        self.to_q = nnx.Linear(dim, dim, dtype=dtype, rngs=rngs)
        self.to_k = nnx.Linear(context_dim, dim // num_heads, dtype=dtype, rngs=rngs)
        self.to_v = nnx.Linear(context_dim, dim // num_heads, dtype=dtype, rngs=rngs)
        self.to_out = nnx.Linear(
            dim, dim, kernel_init=zero_init, bias_init=zero_init, dtype=dtype, rngs=rngs
        )

    def __call__(
        self, x: jnp.ndarray, context: jnp.ndarray, mask: str | None = None
    ) -> jnp.ndarray:
        assert mask is None, "masking is not used by UNet3D and is not ported"
        B, T, D = x.shape
        num_heads = self.num_heads
        q = self.to_q(self.norm(x))
        k = self.to_k(context)
        v = self.to_v(context)
        # (B, T, D) -> (B, H, T, D_head); the single K/V head broadcasts over H
        # via matmul's leading-dim broadcasting (MQA).
        q = q.reshape(B, T, num_heads, -1).transpose(0, 2, 1, 3)
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        scores = (q @ jnp.swapaxes(k, -1, -2)) * self.scale
        # Softmax in fp32 for numerical stability under low-precision compute.
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = attn @ v
        # (B, H, T, D_head) -> (B, T, H*D_head)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = self.to_out(out)

        return out + x


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        hidden_dim = dim * mult
        zero_init = nnx.initializers.zeros_init()
        self.norm = nnx.RMSNorm(
            dim, epsilon=RMS_NORM_EPS, dtype=jnp.float32, rngs=rngs
        )
        self.in_proj = nnx.Linear(dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.act = nnx.silu
        self.out_proj = nnx.Linear(
            hidden_dim, dim, kernel_init=zero_init, bias_init=zero_init, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.out_proj(self.act(self.in_proj(self.norm(x))))
        return h + x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        self.self_attn = CrossAttention(
            dim, context_dim=dim, num_heads=num_heads, dtype=dtype, rngs=rngs
        )
        self.cross_attn = CrossAttention(
            dim, context_dim=context_dim, num_heads=num_heads, dtype=dtype, rngs=rngs
        )
        self.ff = FeedForward(dim, mult=ff_mult, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        x = self.self_attn(x, context=x)
        x = self.cross_attn(x, context=context)
        return self.ff(x)


class UNet3D(nnx.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_actions: int = 1,
        max_context_size: int = 3,
        base_channels: int = 16,
        num_transformer_blocks: int = 2,
        use_wavelet: bool = False,
        predict_reward: bool = False,
        dtype: str | jnp.dtype = "float32",
        rngs: nnx.Rngs,
        **kwargs,
    ):
        # Mixed precision: ``dtype`` is the compute dtype of the conv/attention
        # trunk (a string like "bfloat16" so it JSON round-trips through
        # config.json); params, norms, softmax, and the time/action embeddings
        # stay fp32. ``None`` keeps flax's default promotion (plain fp32).
        dtype = jnp.dtype(dtype)
        self.dtype = None if dtype == jnp.float32 else dtype
        time_embed_dim = base_channels * 4
        self.max_context_size = max_context_size
        self.use_wavelet = use_wavelet
        self.null_action = num_actions
        self.time_embed = TimeEmbedding(time_embed_dim, rngs=rngs)
        # Noise level of the conditioning frames (1.0 = clean), embedded so the
        # model can calibrate how much to trust an imperfect history.
        self.ctx_noise_embed = TimeEmbedding(time_embed_dim, rngs=rngs)
        self.context_embed = ActionEmbedding(
            time_embed_dim, num_actions=num_actions, rngs=rngs
        )

        if use_wavelet:
            self.prepool = WaveletDownsampleConv(in_channels)
            self.unpool = WaveletUpsample()
            in_channels = in_channels * 4

        out_channels = in_channels
        compute_dtype = self.dtype
        self.res1 = ConvResBlock3D(
            in_channels,
            base_channels,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.res2 = ConvResBlock3D(
            base_channels,
            base_channels * 2,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.res3 = ConvResBlock3D(
            base_channels * 2,
            base_channels * 4,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )

        self.mid_transformer_blocks = nnx.List(
            [
                TransformerBlock(
                    base_channels * 4,
                    context_dim=time_embed_dim,
                    num_heads=4,
                    ff_mult=4,
                    dtype=compute_dtype,
                    rngs=rngs,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.mid_conv = ConvResBlock3D(
            base_channels * 4,
            base_channels * 4,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )

        self.up2 = UpBlock3D(
            base_channels * 4,
            base_channels * 2,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.up1 = UpBlock3D(
            base_channels * 2,
            base_channels,
            time_embed_dim=time_embed_dim,
            dtype=compute_dtype,
            rngs=rngs,
        )

        self.out_conv = nnx.Conv(
            base_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            dtype=compute_dtype,
            rngs=rngs,
        )

        self.has_reward_head = predict_reward
        if predict_reward:
            mid_channels = base_channels * 4
            zero_init = nnx.initializers.zeros_init()
            self.reward_head = nnx.Sequential(
                nnx.RMSNorm(mid_channels, epsilon=RMS_NORM_EPS, rngs=rngs),
                nnx.Linear(mid_channels, mid_channels, rngs=rngs),
                nnx.silu,
                nnx.Linear(
                    mid_channels,
                    1,
                    kernel_init=zero_init,
                    bias_init=zero_init,
                    rngs=rngs,
                ),
            )

    def encode(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        t_ctx: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Down + mid path. Returns (xmid, skips, time_context).

        xmid is the post-mid_conv bottleneck, shape
        (B, S/4, H/4, W/4, 4*base_channels): the embedding consumed by
        downstream heads (value, policy, reward). Call with clean frames,
        t=1, and a context ending in the NULL action to get a deterministic
        state embedding for acting.

        ``t_ctx`` is the noise level of the conditioning frames (1.0 = clean);
        when ``None`` the context is treated as clean.
        """
        if self.use_wavelet:
            x = self.prepool(x)

        if t.ndim == 0:
            t = jnp.broadcast_to(t, (x.shape[0],))
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t[:, 0]

        if t_ctx is None:
            t_ctx = jnp.ones_like(t)
        elif t_ctx.ndim == 0:
            t_ctx = jnp.broadcast_to(t_ctx, (x.shape[0],))

        t_emb = self.time_embed(t)
        ctx_emb = self.ctx_noise_embed(t_ctx)
        context = self.context_embed(context)
        action_emb = context[:, -1, :]
        time_context = (t_emb + ctx_emb + action_emb) / 3.0

        x1 = self.res1(x, t_emb)
        x2 = self.res2(_avg_pool_3d(x1), time_context)
        x3 = self.res3(_avg_pool_3d(x2), time_context)

        b, s, h, w, c = x3.shape
        xmid = x3.reshape(b, s * h * w, c)
        for block in self.mid_transformer_blocks:
            xmid = block(xmid, context=context)
        xmid = xmid.reshape(b, s, h, w, c)
        xmid = self.mid_conv(xmid, time_context)
        return xmid, (x1, x2), time_context

    def decode(
        self,
        xmid: jnp.ndarray,
        skips: tuple[jnp.ndarray, jnp.ndarray],
        time_context: jnp.ndarray,
    ) -> jnp.ndarray:
        """Up path: bottleneck features + skips -> predicted velocity."""
        x1, x2 = skips
        x = self.up2(xmid, x2, time_context)
        x = self.up1(x, x1, time_context)

        out = self.out_conv(x)
        if self.use_wavelet:
            out = self.unpool(out)
        return out

    def predict_reward(self, xmid: jnp.ndarray) -> jnp.ndarray:
        """Predict the final frame's reward from bottleneck features.

        Pools xmid (B, S', H', W', C) over space and time, returns (B,).
        """
        pooled = jnp.mean(xmid, axis=(1, 2, 3))
        return self.reward_head(pooled)[:, 0]

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        context: jnp.ndarray,
        t_ctx: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        xmid, skips, time_context = self.encode(x, t, context, t_ctx=t_ctx)
        return self.decode(xmid, skips, time_context)


if __name__ == "__main__":
    for use_wavelet in (False, True):
        label = "wavelet" if use_wavelet else "no wavelet"
        print(f"\n{'=' * 40}\n{label}\n{'=' * 40}")
        model = UNet3D(
            in_channels=3,
            out_channels=3,
            base_channels=16,
            num_actions=3,
            use_wavelet=use_wavelet,
            rngs=nnx.Rngs(0),
        )
        x = jnp.zeros((2, 8, 96, 96, 3))
        t = jnp.ones((2, 1))
        a = jnp.ones((2, 4), dtype=jnp.uint8)
        y = model(x, t, a)
        print(f"input: {x.shape}  output: {y.shape}")

    print(nnx.tabulate(model, x, t, a, depth=2))
