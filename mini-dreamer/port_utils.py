"""Utilities for incrementally checking the JAX port against the MLX code.

Current scope:
- create a small fixed VizDoom test set (default 500 environment steps),
- load the existing MLX VAE checkpoint and save reconstruction samples,
- reserve a placeholder for comparing against a future ``vae_jax.py`` model.
"""

from __future__ import annotations

from pathlib import Path

import click
import imageio.v2 as imageio
import mlx.core as mx
import numpy as np

from data import make_env, record_rollouts
from vae import load_vae
from video_utils import save_clip_previews, to_uint8_video

DEFAULT_MLX_VAE_DIR = Path(
    "/Users/smaug/dev/mini-dreamer-workspace/mini-dreamer/logs/vizdoom-vae"
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

    imageio.imwrite(output_dir / "reconstruction_sheet.png", _reconstruction_sheet(clips[:num_samples], recon_np))
    save_clip_previews(mx.array(recon_np), output_dir / "recon_clips", max_clips=num_samples, fps=8.0)
    return recon_np


def compare_with_jax_placeholder(*_args, **_kwargs) -> None:
    """Placeholder until ``vae_jax.py`` exists."""
    try:
        import vae_jax  # noqa: F401
    except ImportError:
        print("vae_jax.py is not available yet; skipping JAX comparison.")
        return
    raise NotImplementedError("Wire the JAX VAE comparison here once vae_jax.py exists.")


@click.command()
@click.option("--env-id", default="VizdoomBasic-v1", show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option("--vae-dir", type=click.Path(path_type=Path), default=DEFAULT_MLX_VAE_DIR, show_default=True)
@click.option("--rollout-steps", default=500, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--clip-length", default=8, show_default=True)
@click.option("--clip-stride", default=1, show_default=True)
@click.option("--frame-skip", default=4, show_default=True)
@click.option("--num-samples", default=8, show_default=True)
@click.option("--recompute", is_flag=True, help="Regenerate the VizDoom test set even if it already exists.")
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
    print(f"test set: {data_dir} clips={clips.shape} actions={actions.shape} rewards={rewards.shape}")
    save_clip_previews(mx.array(clips), output_dir / "test-set-previews", max_clips=min(num_samples, len(clips)), fps=8.0, actions=actions)
    save_mlx_vae_reconstructions(
        clips,
        vae_dir=vae_dir,
        output_dir=output_dir / "mlx-reconstructions",
        num_samples=num_samples,
    )
    compare_with_jax_placeholder()


if __name__ == "__main__":
    main()
