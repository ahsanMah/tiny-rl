from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax import random
from safetensors.flax import save_file

from jax_utils import flat_params, load_flat_params
from unet_jax import UNet3D
from video_utils import save_clip_previews, save_diffusion_mp4

NoiseDistribution = Literal["uniform", "logitnorm", "normal"]


def make_final_frame_mask(x: jnp.ndarray) -> jnp.ndarray:
    frame_mask = (jnp.arange(x.shape[1]) == (x.shape[1] - 1)).astype(x.dtype)
    return jnp.reshape(frame_mask, (1, x.shape[1], 1, 1, 1))


def sample_t_logit_normal(
    key: jax.Array,
    shape: tuple[int, ...],
    mu: float = 0.0,
    s: float = 1.0,
    eps: float = 1e-6,
) -> jnp.ndarray:
    # logit(t) ~ N(mu, s^2)  =>  t = sigmoid(N(...))
    z = mu + s * random.normal(key, shape)
    t = jax.nn.sigmoid(z)  # t in (0, 1)
    return jnp.clip(t, eps, 1.0 - eps)  # avoid exact 0/1


def sample_noise(
    key: jax.Array,
    shape: tuple[int, ...],
    noise_distribution: NoiseDistribution,
    mu: float = 0.0,
    s: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Unlike the MLX version, the RNG key is explicit: JAX has no global
    random state, so every draw consumes a key the caller controls."""
    match noise_distribution:
        case "logitnorm":
            return sample_t_logit_normal(key, shape, mu, s)

        case "normal":
            return random.normal(key, shape, dtype=dtype)

        case "uniform":
            return random.uniform(key, shape, dtype=dtype)

        case _:
            raise ValueError(f"Unknown noise distribution: {noise_distribution}")

def sample_euler(
    model: UNet3D,
    conditioning_clips: jnp.ndarray,
    actions: jnp.ndarray,
    num_steps: int = 32,
    seed: int = 0,
    return_intermediates: bool = False,
) -> jnp.ndarray | list[jnp.ndarray]:

    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    batch_size = conditioning_clips.shape[0]
    x_shape = (
        batch_size,
        1,
        int(conditioning_clips.shape[2]),
        int(conditioning_clips.shape[3]),
        int(conditioning_clips.shape[4]),
    )
    key = random.key(seed)
    x = random.normal(key, x_shape, dtype=conditioning_clips.dtype)
    eps = 1e-3
    # [eps, 1-eps) with num_steps equal partitions
    dt = (1 - 2 * eps) / num_steps
    # [eps, 1 - eps - eps] so that after final step x is at t = 1-eps
    ts = eps + jnp.arange(num_steps) * dt

    def _step(xt_prev, step_i):
        t = jnp.full((batch_size,), step_i, dtype=conditioning_clips.dtype)
        xt = jnp.concatenate([conditioning_clips, xt_prev], axis=1)
        v = model(xt, t, context=actions)
        x = xt_prev + dt * v[:, -1:]
        return x, x

    ts = jnp.linspace(0,1,num_steps)
    x, intermediates = lax.scan(_step, init=x, xs=ts)

    if return_intermediates:
        return intermediates

    samples = jnp.concatenate([conditioning_clips, x], axis=1)
    return samples


def sample_euler_to_mp4(
    model: UNet3D,
    *,
    conditioning_clips: jnp.ndarray,
    actions: jnp.ndarray,
    output_path: str | Path,
    num_steps: int = 32,
    fps: float = 8.0,
    seed: int = 0,
    decode_fn: Callable | None = None,
) -> None:
    """Generate one new frame and save the full denoising trajectory as an MP4.

    If `decode_fn` is set, the latent conditioning clip and each latent
    intermediate are decoded to pixels before the MP4 is written.

    The MP4 shows the context frames as static columns on the left and the
    evolving generated frame on the right — one MP4 frame per Euler step.

    Args:
        conditioning_clips: ``(B, L, H, W, C)`` context frames.
        actions: ``(B, L+1)`` action tokens.
        output_path: destination ``.mp4`` file.
        num_steps: Euler integration steps (= number of MP4 frames).
        fps: playback speed of the saved video.
    """
    intermediates = sample_euler_jax(
        model,
        conditioning_clips,
        actions,
        num_steps=num_steps,
        seed=seed,
        return_intermediates=True,
    )
    # lax.scan stacks the per-step snapshots into one (num_steps, B, 1, H, W, C)
    # array; save_diffusion_mp4 wants the MLX-style list of (B, 1, H, W, C).
    intermediates = list(intermediates)
    if decode_fn is not None:
        conditioning_clips = decode_fn(conditioning_clips)
        intermediates = [decode_fn(x) for x in intermediates]

    save_diffusion_mp4(conditioning_clips, intermediates, output_path, fps=fps)


def save_model(
    model: UNet3D,
    save_dir: str | Path,
    *,
    config: dict,
    ema_model: UNet3D | None = None,
) -> None:
    """Save model weights and config.

    Writes:
    - `model.safetensors` (always; the non-EMA model)
    - `ema_model.safetensors` (optional; when `ema_model` is provided)
    - `config.json` (written exactly as provided; useful for resume metadata)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(flat_params(model), str(save_dir / "model.safetensors"))
    if ema_model is not None:
        save_file(flat_params(ema_model), str(save_dir / "ema_model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_model(
    save_dir: str | Path, *, prefer_ema: bool = True, seed: int = 0
) -> UNet3D:
    """Load a `UNet3D` previously saved via `save_model` (JAX-native param
    names, i.e. `kernel`/`scale` — not an MLX checkpoint).

    If `prefer_ema` is True and `ema_model.safetensors` exists, EMA weights are
    loaded. Otherwise `model.safetensors` is loaded. The model is first built
    with fresh `nnx.Rngs(seed)` init, then every param is overwritten.
    """
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    model = UNet3D(**config, rngs=nnx.Rngs(seed))

    model_path = save_dir / "model.safetensors"
    ema_path = save_dir / "ema_model.safetensors"
    weights_path = ema_path if prefer_ema and ema_path.exists() else model_path
    load_flat_params(model, weights_path)
    return model


def generate_video(
    model: UNet3D,
    *,
    initial_clip: jnp.ndarray,
    num_new_frames: int,
    actions: jnp.ndarray,
    num_steps: int = 32,
    seed: int = 0,
) -> jnp.ndarray:
    """Autoregressively extend `initial_clip` by `num_new_frames` new frames.

    Args:
        initial_clip: (B, L, H, W, C) seed frames.
        num_new_frames: number of frames to generate after the seed.
        actions: (B, L + num_new_frames) action stream. Each Euler
            call receives an (L - 1)-frame conditioning window and an L-action
            window aligned to the generated clip.
        num_steps: Euler integration steps per generated frame.
        seed: base RNG seed; each generated frame draws its initial noise from
            `seed + step` so frames don't share the same noise sample.

    Returns:
        (B, L + num_new_frames, H, W, C) array of frames.
    """
    if num_new_frames <= 0:
        raise ValueError(f"num_new_frames must be > 0, got {num_new_frames}")

    clip_length = int(initial_clip.shape[1])
    expected = clip_length + num_new_frames

    print(f"Generating {num_new_frames} frames from {clip_length} initial frames...")
    if int(actions.shape[1]) != expected:
        raise ValueError(
            f"actions must have shape (B, {expected}), got {tuple(actions.shape)}"
        )

    if clip_length < 2:
        raise ValueError(
            f"initial_clip must contain at least 2 frames, got {clip_length}"
        )
    print(f"Using max_context_size={model.max_context_size}")
    frames = initial_clip
    max_context_size = model.max_context_size
    for step in range(num_new_frames):
        window = frames[:, -max_context_size:]
        print(
            f"generating frame {step + 1}/{num_new_frames} with conditioning window shape {window.shape}..."
        )
        end = frames.shape[1] + 1  # frame index being generated, + 1 inclusive
        action_window = actions[:, max(0, end - max_context_size - 1) : end]

        sample = sample_euler_jax(
            model,
            window,
            action_window,
            num_steps=num_steps,
            seed=seed + step,
        )
        frames = jnp.concatenate([frames, sample[:, -1:]], axis=1)

    return frames


def generate_env_video(
    model: UNet3D,
    *,
    initial_clip: jnp.ndarray,
    initial_actions: jnp.ndarray,
    num_actions: int,
    num_new_frames: int,
    num_steps: int = 32,
    sample_fps: float = 2.0,
    save_dir: str | Path,
    seed: int = 0,
    actions_pool: list[int] | None = None,
    decode_fn: Callable | None = None,
) -> jnp.ndarray:
    """Autoregressively extend `initial_clip` using random actions for new frames.

    If `decode_fn` is set, generation runs in latent space and `decode_fn` maps
    latents -> pixels right before previews are saved (default None = pixel space).
    """
    rng = np.random.default_rng(seed)
    initial_actions_np = np.asarray(initial_actions)
    batch_size = int(initial_clip.shape[0])
    if actions_pool is not None:
        extra_actions_np = rng.choice(actions_pool, size=(batch_size, num_new_frames))
    else:
        extra_actions_np = rng.integers(
            0, num_actions, size=(batch_size, num_new_frames)
        ).astype(np.int32)
    full_actions = jnp.asarray(
        np.concatenate([initial_actions_np, extra_actions_np], axis=1)
    )
    generated = generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=full_actions,
        num_steps=num_steps,
        seed=seed,
    )
    preview_clips = decode_fn(generated) if decode_fn is not None else generated
    save_clip_previews(
        preview_clips,
        save_dir,
        max_clips=batch_size,
        fps=sample_fps,
        actions=full_actions,
    )
    sample_euler_to_mp4(
        model,
        conditioning_clips=initial_clip[:, : model.max_context_size],
        actions=full_actions[:, : model.max_context_size + 1],
        output_path=f"{save_dir}/denoising.mp4",
        num_steps=num_steps,
        fps=num_steps / 4.0,
        seed=seed,
        decode_fn=decode_fn,
    )
    print(f"saved generated video to: {save_dir}")
    return generated
