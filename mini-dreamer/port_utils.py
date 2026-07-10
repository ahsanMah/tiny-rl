"""Utilities for incrementally checking the JAX port against the MLX code.

Current scope:
- create a small fixed VizDoom test set (default 500 environment steps),
- load the existing MLX VAE checkpoint and save reconstruction samples,
- reserve a placeholder for comparing against a future ``vae_jax.py`` model.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import imageio.v2 as imageio
import mlx.core as mx
import numpy as np

from data import make_env, record_rollouts
from vae import load_vae
from video_utils import save_clip_previews, to_uint8_video


def _max_abs_diff(a: object, b: object) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

DEFAULT_MLX_VAE_DIR = Path(
    "/Users/smaug/dev/mini-dreamer-workspace/mini-dreamer/logs/vizdoom-vae"
)
DEFAULT_MLX_UNET_DIR = Path(
    "/Users/smaug/dev/mini-dreamer-workspace/mini-dreamer/logs/vizdoom-latent-v2"
)
DEFAULT_OUTPUT_DIR = Path("logs/port-utils/vizdoom-vae")
DEFAULT_DATA_DIR = DEFAULT_OUTPUT_DIR / "test-set"


def create_vizdoom_test_set(
    *,
    env_id: str = "VizdoomBasic-v1",
    output_dir: str | Path = DEFAULT_DATA_DIR,
    rollout_steps: int = 500,
    seed: int = 0,
    clip_length: int = 8,
    clip_stride: int | None = 1,
    frame_skip: int = 4,
    pad_multiple: int = 16,
    recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Record a fixed VizDoom rollout and save clip tensors to disk."""
    output_dir = Path(output_dir)
    env = make_env(env_id, frame_skip=frame_skip)
    frames, actions, rewards, save_dir = record_rollouts(
        env,
        num_steps=rollout_steps,
        seed=seed,
        clip_length=clip_length,
        clip_stride=clip_stride,
        save_to_disk=True,
        save_dir=output_dir,
        pad_multiple=pad_multiple,
        recompute=recompute,
    )
    assert save_dir is not None
    return frames, actions, rewards, Path(save_dir)


def _reconstruction_sheet(original: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Make a PNG sheet with original/reconstruction rows for each sample."""
    originals = to_uint8_video(original)
    recons = to_uint8_video(recon)
    samples, frames, height, width, channels = originals.shape
    sheet = np.zeros((samples * 2 * height, frames * width, channels), dtype=np.uint8)
    for sample_idx in range(samples):
        for frame_idx in range(frames):
            x0 = frame_idx * width
            y_orig = sample_idx * 2 * height
            y_recon = y_orig + height
            sheet[y_orig : y_orig + height, x0 : x0 + width] = originals[
                sample_idx, frame_idx
            ]
            sheet[y_recon : y_recon + height, x0 : x0 + width] = recons[
                sample_idx, frame_idx
            ]
    if channels == 1:
        return sheet[..., 0]
    return sheet


def save_mlx_vae_reconstructions(
    clips: np.ndarray,
    *,
    vae_dir: str | Path = DEFAULT_MLX_VAE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "mlx-reconstructions",
    num_samples: int = 8,
    prefer_ema: bool = True,
) -> np.ndarray:
    """Load the MLX VAE checkpoint and save sample reconstructions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae = load_vae(vae_dir, prefer_ema=prefer_ema)
    batch = mx.array(clips[:num_samples])
    mean, _ = vae.encode(batch)
    recon = vae.decode(mean)
    mx.eval(recon)
    recon_np = np.asarray(recon)

    imageio.imwrite(
        output_dir / "reconstruction_sheet.png",
        _reconstruction_sheet(clips[:num_samples], recon_np),
    )
    save_clip_previews(
        mx.array(recon_np), output_dir / "recon_clips", max_clips=num_samples, fps=8.0
    )
    return recon_np


def _set_nnx_param(param, value) -> None:
    import jax.numpy as jnp

    param.value = jnp.asarray(value)


def _copy_linear_to_nnx(mlx_linear, jax_linear) -> None:
    # MLX Linear stores O,I; Flax NNX Linear stores I,O.
    _set_nnx_param(jax_linear.kernel, np.asarray(mlx_linear.weight).T)
    if mlx_linear.bias is not None:
        _set_nnx_param(jax_linear.bias, np.asarray(mlx_linear.bias))


def _copy_embedding_to_nnx(mlx_embedding, jax_embedding) -> None:
    # Both frameworks store embeddings as V,D.
    _set_nnx_param(jax_embedding.embedding, np.asarray(mlx_embedding.weight))


def _copy_conv_to_nnx(mlx_conv, jax_conv) -> None:
    # MLX Conv2d stores O,H,W,I; Flax NNX Conv stores H,W,I,O.
    _set_nnx_param(jax_conv.kernel, np.asarray(mlx_conv.weight).transpose(1, 2, 3, 0))
    if mlx_conv.bias is not None:
        _set_nnx_param(jax_conv.bias, np.asarray(mlx_conv.bias))


def _copy_conv3d_to_nnx(mlx_conv, jax_conv) -> None:
    # MLX Conv3d stores O,D,H,W,I; Flax NNX Conv stores D,H,W,I,O.
    _set_nnx_param(
        jax_conv.kernel, np.asarray(mlx_conv.weight).transpose(1, 2, 3, 4, 0)
    )
    if mlx_conv.bias is not None:
        _set_nnx_param(jax_conv.bias, np.asarray(mlx_conv.bias))


def _copy_rmsnorm_to_nnx(mlx_norm, jax_norm) -> None:
    _set_nnx_param(jax_norm.scale, np.asarray(mlx_norm.weight))
    # MLX defaults to eps=1e-5, NNX to 1e-6; the checkpoint was trained with
    # MLX's value, so carry it over or outputs drift by ~1e-4.
    jax_norm.epsilon = mlx_norm.eps


def _copy_resblock_to_nnx(mlx_block, jax_block) -> None:
    if hasattr(mlx_block.shortcut, "weight"):
        _copy_conv_to_nnx(mlx_block.shortcut, jax_block.shortcut)
    _copy_rmsnorm_to_nnx(mlx_block.norm1, jax_block.norm1)
    _copy_rmsnorm_to_nnx(mlx_block.norm2, jax_block.norm2)
    _copy_conv_to_nnx(mlx_block.conv1, jax_block.conv1)
    _copy_conv_to_nnx(mlx_block.conv2, jax_block.conv2)


def _copy_mlx_vae_to_jax(mlx_vae, jax_vae) -> None:
    _copy_conv_to_nnx(mlx_vae.enc_stem, jax_vae.enc_stem)
    for mlx_block, jax_block in zip(mlx_vae.enc_blocks, jax_vae.enc_blocks):
        _copy_resblock_to_nnx(mlx_block, jax_block)
    for mlx_down, jax_down in zip(mlx_vae.enc_downs, jax_vae.enc_downs):
        _copy_conv_to_nnx(mlx_down, jax_down)
    _copy_resblock_to_nnx(mlx_vae.enc_mid, jax_vae.enc_mid)
    _copy_conv_to_nnx(mlx_vae.to_moments, jax_vae.to_moments)
    _copy_conv_to_nnx(mlx_vae.from_latent, jax_vae.from_latent)
    _copy_resblock_to_nnx(mlx_vae.dec_mid, jax_vae.dec_mid)
    for mlx_block, jax_block in zip(mlx_vae.dec_blocks, jax_vae.dec_blocks):
        _copy_resblock_to_nnx(mlx_block, jax_block)
    _copy_conv_to_nnx(mlx_vae.dec_out, jax_vae.dec_out)


def export_flax_vae_checkpoint(
    *,
    vae_dir: str | Path = DEFAULT_MLX_VAE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "flax-vae",
) -> Path:
    """Convert the MLX VAE checkpoint into a ``vae_jax.load_vae``-loadable one.

    Loads the MLX weights (model and, if present, EMA) into JAX ``WaveletVAE``s
    via the copy helpers (transposing conv kernels to NNX layout) and re-saves
    them with ``vae_jax.save_vae``, preserving the checkpoint layout
    (``model.safetensors`` / ``ema_model.safetensors`` / ``config.json``).

    Note: safetensors only carries tensors, so the RMSNorm epsilon fix applied
    by ``_copy_rmsnorm_to_nnx`` (MLX 1e-5 vs NNX default 1e-6) is *not* part of
    the exported checkpoint; ``vae_jax.load_vae`` reconstructs with NNX defaults.
    """
    from flax import nnx

    from vae_jax import WaveletVAE as JaxWaveletVAE
    from vae_jax import save_vae as save_jax_vae

    vae_dir = Path(vae_dir)
    config = json.loads((vae_dir / "config.json").read_text())
    model_config = {k: v for k, v in config.items() if k != "step"}

    def _to_jax(prefer_ema: bool) -> JaxWaveletVAE:
        mlx_vae = load_vae(vae_dir, prefer_ema=prefer_ema)
        jax_vae = JaxWaveletVAE(**model_config, rngs=nnx.Rngs(0, reparam=1))
        _copy_mlx_vae_to_jax(mlx_vae, jax_vae)
        return jax_vae

    ema_exists = (vae_dir / "ema_model.safetensors").exists()
    save_jax_vae(
        _to_jax(prefer_ema=False),
        output_dir,
        config=config,
        ema_model=_to_jax(prefer_ema=True) if ema_exists else None,
    )
    return Path(output_dir)


def _copy_unet_time_embedding_to_nnx(mlx_time, jax_time) -> None:
    _set_nnx_param(jax_time.fourier.weight, np.asarray(mlx_time.fourier.weight))
    _copy_linear_to_nnx(mlx_time.lin1, jax_time.lin1)
    _copy_linear_to_nnx(mlx_time.lin2, jax_time.lin2)


def _copy_action_embedding_to_nnx(mlx_embed, jax_embed) -> None:
    if mlx_embed.categorical:
        _copy_embedding_to_nnx(mlx_embed.embed, jax_embed.embed)
    else:
        _copy_linear_to_nnx(mlx_embed.embed, jax_embed.embed)


def _copy_resblock3d_to_nnx(mlx_block, jax_block) -> None:
    if hasattr(mlx_block.shortcut, "weight"):
        _copy_conv3d_to_nnx(mlx_block.shortcut, jax_block.shortcut)
    _copy_rmsnorm_to_nnx(mlx_block.norm1, jax_block.norm1)
    _copy_rmsnorm_to_nnx(mlx_block.norm2, jax_block.norm2)
    _copy_conv3d_to_nnx(mlx_block.conv1, jax_block.conv1)
    _copy_conv3d_to_nnx(mlx_block.conv2, jax_block.conv2)
    if hasattr(mlx_block, "to_film"):
        _copy_linear_to_nnx(mlx_block.to_film, jax_block.to_film)


def _copy_cross_attention_to_nnx(mlx_attn, jax_attn) -> None:
    _copy_rmsnorm_to_nnx(mlx_attn.norm, jax_attn.norm)
    for name in ("to_q", "to_k", "to_v", "to_out"):
        _copy_linear_to_nnx(getattr(mlx_attn, name), getattr(jax_attn, name))


def _copy_feedforward_to_nnx(mlx_ff, jax_ff) -> None:
    _copy_rmsnorm_to_nnx(mlx_ff.norm, jax_ff.norm)
    _copy_linear_to_nnx(mlx_ff.in_proj, jax_ff.in_proj)
    _copy_linear_to_nnx(mlx_ff.out_proj, jax_ff.out_proj)


def _copy_transformer_block_to_nnx(mlx_block, jax_block) -> None:
    _copy_cross_attention_to_nnx(mlx_block.self_attn, jax_block.self_attn)
    _copy_cross_attention_to_nnx(mlx_block.cross_attn, jax_block.cross_attn)
    _copy_feedforward_to_nnx(mlx_block.ff, jax_block.ff)


def _copy_upblock3d_to_nnx(mlx_block, jax_block) -> None:
    _copy_conv3d_to_nnx(mlx_block.up_conv, jax_block.up_conv)
    _copy_resblock3d_to_nnx(mlx_block.conv, jax_block.conv)


def _copy_mlx_unet_to_jax(mlx_model, jax_model) -> None:
    _copy_unet_time_embedding_to_nnx(mlx_model.time_embed, jax_model.time_embed)
    _copy_unet_time_embedding_to_nnx(mlx_model.ctx_noise_embed, jax_model.ctx_noise_embed)
    _copy_action_embedding_to_nnx(mlx_model.context_embed, jax_model.context_embed)
    for name in ("res1", "res2", "res3", "mid_conv"):
        _copy_resblock3d_to_nnx(getattr(mlx_model, name), getattr(jax_model, name))
    for mlx_block, jax_block in zip(
        mlx_model.mid_transformer_blocks, jax_model.mid_transformer_blocks
    ):
        _copy_transformer_block_to_nnx(mlx_block, jax_block)
    _copy_upblock3d_to_nnx(mlx_model.up2, jax_model.up2)
    _copy_upblock3d_to_nnx(mlx_model.up1, jax_model.up1)
    _copy_conv3d_to_nnx(mlx_model.out_conv, jax_model.out_conv)
    if mlx_model.has_reward_head:
        # Sequential(RMSNorm, Linear, SiLU, Linear) in both frameworks.
        _copy_rmsnorm_to_nnx(mlx_model.reward_head.layers[0], jax_model.reward_head.layers[0])
        _copy_linear_to_nnx(mlx_model.reward_head.layers[1], jax_model.reward_head.layers[1])
        _copy_linear_to_nnx(mlx_model.reward_head.layers[3], jax_model.reward_head.layers[3])


def compare_unet_jax_components(
    *,
    batch_size: int = 4,
    dim: int = 16,
    seed: int = 0,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Run deterministic MLX-vs-JAX parity checks for ported ``unet.py`` pieces.

    This intentionally tests only components that exist in ``unet_jax.py`` so it
    can be used incrementally while the port is still in progress. Add new blocks
    here as their JAX counterparts are implemented.
    """
    import jax.numpy as jnp
    from flax import nnx

    import unet as mlx_unet
    import unet_jax

    metrics: dict[str, float] = {}
    rngs = nnx.Rngs(seed, reparam=seed + 1, params=seed + 2)

    from mlx.utils import tree_map

    rng = np.random.default_rng(seed)

    def _randomize_zero_params(mlx_module, scale: float = 0.02) -> None:
        """Fill all-zero (zero-initialized) params with small noise so parity
        checks exercise those branches instead of comparing zeros to zeros."""

        def repl(p):
            if p.size > 0 and float(mx.abs(p).max()) == 0.0:
                return mx.random.normal(p.shape) * scale
            return p

        mlx_module.update(tree_map(repl, mlx_module.parameters()))

    # GaussianFourierEmbedding has no layout differences; copy the sampled MLX
    # frequencies into the JAX module and compare scalar and batched calls.
    mlx_fourier = mlx_unet.GaussianFourierEmbedding(dim)
    jax_fourier = unet_jax.GaussianFourierEmbedding(dim, rngs=rngs)
    _set_nnx_param(jax_fourier.weight, np.asarray(mlx_fourier.weight))

    t_np = np.linspace(0.0, 1.0, batch_size, dtype=np.float32)
    mlx_out = mlx_fourier(mx.array(t_np))
    jax_out = jax_fourier(jnp.asarray(t_np))
    mx.eval(mlx_out)
    metrics["gaussian_fourier_max_abs_diff"] = _max_abs_diff(mlx_out, jax_out)

    mlx_scalar = mlx_fourier(mx.array(0.25, dtype=mx.float32))
    jax_scalar = jax_fourier(jnp.asarray(0.25, dtype=jnp.float32))
    mx.eval(mlx_scalar)
    metrics["gaussian_fourier_scalar_max_abs_diff"] = _max_abs_diff(
        mlx_scalar, jax_scalar
    )

    mlx_time = mlx_unet.TimeEmbedding(dim)
    jax_time = unet_jax.TimeEmbedding(dim, rngs=rngs)
    _copy_unet_time_embedding_to_nnx(mlx_time, jax_time)
    mlx_time_out = mlx_time(mx.array(t_np))
    jax_time_out = jax_time(jnp.asarray(t_np))
    mx.eval(mlx_time_out)
    metrics["time_embedding_max_abs_diff"] = _max_abs_diff(mlx_time_out, jax_time_out)

    # ActionEmbedding: categorical lookup incl. the NULL slot, and continuous.
    num_actions = 3
    mlx_act = mlx_unet.ActionEmbedding(dim, num_actions=num_actions)
    jax_act = unet_jax.ActionEmbedding(dim, num_actions=num_actions, rngs=rngs)
    _copy_action_embedding_to_nnx(mlx_act, jax_act)
    actions_np = rng.integers(0, num_actions + 1, size=(batch_size, 4))
    metrics["action_embedding_max_abs_diff"] = _max_abs_diff(
        mlx_act(mx.array(actions_np)), jax_act(jnp.asarray(actions_np))
    )

    mlx_act_cont = mlx_unet.ActionEmbedding(dim, action_dim=2)
    jax_act_cont = unet_jax.ActionEmbedding(dim, action_dim=2, rngs=rngs)
    _copy_action_embedding_to_nnx(mlx_act_cont, jax_act_cont)
    cont_np = rng.standard_normal((batch_size, 4, 2)).astype(np.float32)
    metrics["action_embedding_continuous_max_abs_diff"] = _max_abs_diff(
        mlx_act_cont(mx.array(cont_np)), jax_act_cont(jnp.asarray(cont_np))
    )

    # ConvResBlock3D with FiLM conditioning and a non-identity shortcut.
    x_np = rng.standard_normal((batch_size, 4, 8, 8, 3)).astype(np.float32)
    temb_np = rng.standard_normal((batch_size, dim)).astype(np.float32)
    mlx_res = mlx_unet.ConvResBlock3D(3, dim, time_embed_dim=dim)
    jax_res = unet_jax.ConvResBlock3D(3, dim, time_embed_dim=dim, rngs=rngs)
    _randomize_zero_params(mlx_res)
    _copy_resblock3d_to_nnx(mlx_res, jax_res)
    metrics["conv_resblock3d_max_abs_diff"] = _max_abs_diff(
        mlx_res(mx.array(x_np), mx.array(temb_np)),
        jax_res(jnp.asarray(x_np), jnp.asarray(temb_np)),
    )

    # UpBlock3D: nearest upsample + 1x1 conv + skip + resblock.
    skip_np = rng.standard_normal((batch_size, 8, 16, 16, dim)).astype(np.float32)
    xin_np = rng.standard_normal((batch_size, 4, 8, 8, 2 * dim)).astype(np.float32)
    mlx_up = mlx_unet.UpBlock3D(2 * dim, dim, time_embed_dim=dim)
    jax_up = unet_jax.UpBlock3D(2 * dim, dim, time_embed_dim=dim, rngs=rngs)
    _randomize_zero_params(mlx_up)
    _copy_upblock3d_to_nnx(mlx_up, jax_up)
    metrics["upblock3d_max_abs_diff"] = _max_abs_diff(
        mlx_up(mx.array(xin_np), mx.array(skip_np), mx.array(temb_np)),
        jax_up(jnp.asarray(xin_np), jnp.asarray(skip_np), jnp.asarray(temb_np)),
    )

    # TransformerBlock (MQA self-attn + cross-attn + FF).
    seq_np = rng.standard_normal((batch_size, 10, dim)).astype(np.float32)
    ctx_np = rng.standard_normal((batch_size, 4, 2 * dim)).astype(np.float32)
    mlx_tf = mlx_unet.TransformerBlock(dim, context_dim=2 * dim, num_heads=4)
    jax_tf = unet_jax.TransformerBlock(dim, context_dim=2 * dim, num_heads=4, rngs=rngs)
    _randomize_zero_params(mlx_tf)
    _copy_transformer_block_to_nnx(mlx_tf, jax_tf)
    metrics["transformer_block_max_abs_diff"] = _max_abs_diff(
        mlx_tf(mx.array(seq_np), context=mx.array(ctx_np)),
        jax_tf(jnp.asarray(seq_np), context=jnp.asarray(ctx_np)),
    )

    # Full UNet3D forward (covers avg-pool/upsample edge behavior), both
    # wavelet modes, plus encode() bottleneck and the reward head.
    clip_np = rng.standard_normal((2, 8, 16, 16, 3)).astype(np.float32)
    t_full_np = rng.uniform(0.0, 1.0, size=(2, 1)).astype(np.float32)
    unet_actions_np = rng.integers(0, num_actions + 1, size=(2, 4))
    for use_wavelet in (False, True):
        label = "wavelet" if use_wavelet else "pixel"
        mlx_model = mlx_unet.UNet3D(
            in_channels=3,
            base_channels=16,
            num_actions=num_actions,
            use_wavelet=use_wavelet,
            predict_reward=True,
        )
        jax_model = unet_jax.UNet3D(
            in_channels=3,
            base_channels=16,
            num_actions=num_actions,
            use_wavelet=use_wavelet,
            predict_reward=True,
            rngs=rngs,
        )
        _randomize_zero_params(mlx_model)
        _copy_mlx_unet_to_jax(mlx_model, jax_model)

        mlx_full = mlx_model(
            mx.array(clip_np),
            mx.array(t_full_np),
            mx.array(unet_actions_np),
            t_ctx=mx.array(0.5, dtype=mx.float32),
        )
        jax_full = jax_model(
            jnp.asarray(clip_np),
            jnp.asarray(t_full_np),
            jnp.asarray(unet_actions_np),
            t_ctx=jnp.asarray(0.5, dtype=jnp.float32),
        )
        mx.eval(mlx_full)
        metrics[f"unet3d_{label}_max_abs_diff"] = _max_abs_diff(mlx_full, jax_full)

        mlx_xmid, _, _ = mlx_model.encode(
            mx.array(clip_np), mx.array(t_full_np), mx.array(unet_actions_np)
        )
        jax_xmid, _, _ = jax_model.encode(
            jnp.asarray(clip_np), jnp.asarray(t_full_np), jnp.asarray(unet_actions_np)
        )
        metrics[f"unet3d_{label}_xmid_max_abs_diff"] = _max_abs_diff(mlx_xmid, jax_xmid)
        metrics[f"unet3d_{label}_reward_max_abs_diff"] = _max_abs_diff(
            mlx_model.predict_reward(mlx_xmid), jax_model.predict_reward(jax_xmid)
        )

    print("JAX UNet component comparison:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8g}")
    failures = {key: value for key, value in metrics.items() if value > atol}
    if failures:
        raise AssertionError(f"JAX UNet component comparison exceeded atol={atol}: {failures}")
    return metrics


def compare_unet_jax_checkpoint(
    clips: np.ndarray,
    actions: np.ndarray,
    *,
    unet_dir: str | Path = DEFAULT_MLX_UNET_DIR,
    vae_dir: str | Path = DEFAULT_MLX_VAE_DIR,
    num_samples: int = 4,
    prefer_ema: bool = True,
    atol: float = 1e-4,
    pad_multiple: int = 32,
) -> dict[str, float]:
    """Load the MLX UNet3D checkpoint into the JAX port and compare
    ``encode()``/``decode()``/``predict_reward()`` on the fixed VizDoom test set.

    Deterministic by construction: clips are encoded with the VAE posterior
    mean (``encode_clips``), and the UNet gets a fixed ``t``, fixed ``t_ctx``
    and the recorded actions — no sampling anywhere.

    ``pad_multiple`` must match the training config (vizdoom-latent used 32);
    smaller pads give latent sizes that don't round-trip the UNet's down/up path.
    """
    import jax.numpy as jnp
    from flax import nnx

    import unet_jax
    from data import pad_frames_to_multiple
    from diffusion import load_model as load_mlx_unet
    from vae import encode_clips

    unet_dir = Path(unet_dir)
    mlx_model = load_mlx_unet(unet_dir, prefer_ema=prefer_ema)
    config = json.loads((unet_dir / "config.json").read_text())
    jax_model = unet_jax.UNet3D(**config, rngs=nnx.Rngs(0))
    _copy_mlx_unet_to_jax(mlx_model, jax_model)

    vae = load_vae(vae_dir, prefer_ema=prefer_ema)
    # pad_frames_to_multiple takes (T, H, W, C); flatten the clip dims around it.
    raw = clips[:num_samples]
    padded = pad_frames_to_multiple(
        raw.reshape(-1, *raw.shape[2:]), multiple=pad_multiple
    )
    batch_np = padded.reshape(*raw.shape[:2], *padded.shape[1:])
    latents = encode_clips(vae, mx.array(batch_np))
    mx.eval(latents)
    latents_np = np.asarray(latents)
    actions_np = np.asarray(actions[:num_samples])

    t_np = np.full((num_samples, 1), 0.7, dtype=np.float32)
    t_ctx = 0.5

    mlx_xmid, mlx_skips, mlx_tc = mlx_model.encode(
        mx.array(latents_np),
        mx.array(t_np),
        mx.array(actions_np),
        t_ctx=mx.array(t_ctx, dtype=mx.float32),
    )
    mlx_out = mlx_model.decode(mlx_xmid, mlx_skips, mlx_tc)
    mlx_reward = mlx_model.predict_reward(mlx_xmid)
    mx.eval(mlx_xmid, mlx_out, mlx_reward)

    jax_xmid, jax_skips, jax_tc = jax_model.encode(
        jnp.asarray(latents_np),
        jnp.asarray(t_np),
        jnp.asarray(actions_np),
        t_ctx=jnp.asarray(t_ctx, dtype=jnp.float32),
    )
    jax_out = jax_model.decode(jax_xmid, jax_skips, jax_tc)
    jax_reward = jax_model.predict_reward(jax_xmid)

    metrics = {
        "xmid_max_abs_diff": _max_abs_diff(mlx_xmid, jax_xmid),
        "decode_max_abs_diff": _max_abs_diff(mlx_out, jax_out),
        "reward_max_abs_diff": _max_abs_diff(mlx_reward, jax_reward),
    }
    print(f"JAX UNet checkpoint comparison ({unet_dir}):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8g}")
    failures = {key: value for key, value in metrics.items() if value > atol}
    if failures:
        raise AssertionError(
            f"JAX UNet checkpoint comparison exceeded atol={atol}: {failures}"
        )
    return metrics


def compare_with_jax_vae(
    clips: np.ndarray,
    *,
    vae_dir: str | Path = DEFAULT_MLX_VAE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "jax-comparison",
    num_samples: int = 8,
    prefer_ema: bool = True,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Load MLX weights into the JAX VAE and compare deterministic outputs."""
    import jax.numpy as jnp
    from flax import nnx

    from vae_jax import WaveletVAE as JaxWaveletVAE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_vae = load_vae(vae_dir, prefer_ema=prefer_ema)
    config = json.loads((Path(vae_dir) / "config.json").read_text())
    config = {k: v for k, v in config.items() if k != "step"}
    jax_vae = JaxWaveletVAE(**config, rngs=nnx.Rngs(0, reparam=1))
    _copy_mlx_vae_to_jax(mlx_vae, jax_vae)

    batch_np = clips[:num_samples]
    batch_mx = mx.array(batch_np)
    mlx_mean, mlx_logvar = mlx_vae.encode(batch_mx)
    mlx_recon = mlx_vae.decode(mlx_mean)
    mx.eval(mlx_mean, mlx_logvar, mlx_recon)

    jax_mean, jax_logvar = jax_vae.encode(jnp.asarray(batch_np))
    jax_recon = jax_vae.decode(jax_mean)

    mlx_mean_np = np.asarray(mlx_mean)
    mlx_logvar_np = np.asarray(mlx_logvar)
    mlx_recon_np = np.asarray(mlx_recon)
    jax_mean_np = np.asarray(jax_mean)
    jax_logvar_np = np.asarray(jax_logvar)
    jax_recon_np = np.asarray(jax_recon)

    metrics = {
        "mean_max_abs_diff": float(np.max(np.abs(mlx_mean_np - jax_mean_np))),
        "logvar_max_abs_diff": float(np.max(np.abs(mlx_logvar_np - jax_logvar_np))),
        "recon_max_abs_diff": float(np.max(np.abs(mlx_recon_np - jax_recon_np))),
        "recon_mean_abs_diff": float(np.mean(np.abs(mlx_recon_np - jax_recon_np))),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imageio.imwrite(
        output_dir / "jax_reconstruction_sheet.png",
        _reconstruction_sheet(batch_np, jax_recon_np),
    )
    imageio.imwrite(
        output_dir / "diff_sheet.png",
        to_uint8_video(np.abs(mlx_recon_np - jax_recon_np) * 2.0 - 1.0)[0, 0],
    )

    print("JAX VAE comparison:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.8g}")
    if (
        metrics["mean_max_abs_diff"] > atol
        or metrics["logvar_max_abs_diff"] > atol
        or metrics["recon_max_abs_diff"] > atol
    ):
        raise AssertionError(f"JAX VAE comparison exceeded atol={atol}: {metrics}")
    return metrics


@click.command()
@click.option("--env-id", default="VizdoomBasic-v1", show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "--vae-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_MLX_VAE_DIR,
    show_default=True,
)
@click.option("--rollout-steps", default=500, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--clip-length", default=8, show_default=True)
@click.option("--clip-stride", default=1, show_default=True)
@click.option("--frame-skip", default=4, show_default=True)
@click.option("--num-samples", default=8, show_default=True)
@click.option("--compare-atol", default=1e-3, show_default=True)
@click.option(
    "--run-unet-tests",
    is_flag=True,
    help="Run incremental MLX-vs-JAX parity checks for ported unet.py components.",
)
@click.option("--unet-atol", default=1e-5, show_default=True)
@click.option(
    "--unet-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_MLX_UNET_DIR,
    show_default=True,
)
@click.option("--unet-ckpt-atol", default=1e-4, show_default=True)
@click.option(
    "--recompute",
    is_flag=True,
    help="Regenerate the VizDoom test set even if it already exists.",
)
def main(
    env_id: str,
    output_dir: Path,
    vae_dir: Path,
    rollout_steps: int,
    seed: int,
    clip_length: int,
    clip_stride: int,
    frame_skip: int,
    num_samples: int,
    compare_atol: float,
    run_unet_tests: bool,
    unet_atol: float,
    unet_dir: Path,
    unet_ckpt_atol: float,
    recompute: bool,
) -> None:
    clips, actions, rewards, data_dir = create_vizdoom_test_set(
        env_id=env_id,
        output_dir=output_dir / "test-set",
        rollout_steps=rollout_steps,
        seed=seed,
        clip_length=clip_length,
        clip_stride=clip_stride,
        frame_skip=frame_skip,
        recompute=recompute,
    )
    print(
        f"test set: {data_dir} clips={clips.shape} actions={actions.shape} rewards={rewards.shape}"
    )
    save_clip_previews(
        mx.array(clips),
        output_dir / "test-set-previews",
        max_clips=min(num_samples, len(clips)),
        fps=8.0,
        actions=actions,
    )
    save_mlx_vae_reconstructions(
        clips,
        vae_dir=vae_dir,
        output_dir=output_dir / "mlx-reconstructions",
        num_samples=num_samples,
    )
    compare_with_jax_vae(
        clips,
        vae_dir=vae_dir,
        output_dir=output_dir / "jax-comparison",
        num_samples=num_samples,
        atol=compare_atol,
    )
    if run_unet_tests:
        compare_unet_jax_components(atol=unet_atol)
        compare_unet_jax_checkpoint(
            clips,
            actions,
            unet_dir=unet_dir,
            vae_dir=vae_dir,
            atol=unet_ckpt_atol,
        )


if __name__ == "__main__":
    main()
