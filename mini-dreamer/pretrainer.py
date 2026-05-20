from __future__ import annotations

from pathlib import Path

import click
import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from diffusion import generate_video, load_model, save_model, train_on_dataset
from unet import UNet3D
from video_utils import frames_to_clips, save_clip_previews


def rollout_minigrid_frames(
    env: gym.Env,
    *,
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    highlight: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out random actions in a MiniGrid env and capture tile-rendered RGB frames.

    Returns:
        frames: array of shape (num_steps, H, W, 3) of float32 in [-1, 1].
        actions: array of shape (num_steps,) of int32. `actions[i]` is the
            action taken at `frames[i]` (which produced the next frame).
    Episodes auto-reset on terminate/truncate so the frame stream is contiguous.
    """

    env = RGBImgObsWrapper(env, tile_size=tile_size)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []

    for _ in range(num_steps):
        action = env.action_space.sample() % 3
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)

        frame = obs["image"]
        frames.append(np.asarray(frame))
        if terminated or truncated:
            env.reset()

    env.close()

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    return stacked, action_array


def actions_to_clips(
    actions: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> mx.array:
    """Slice a 1-D action stream into (num_clips, clip_length) windows."""
    if actions.ndim != 1:
        raise ValueError(f"Expected actions with shape (T,), got {actions.shape}")
    if actions.shape[0] < clip_length:
        raise ValueError(
            f"Need at least {clip_length} actions, but only have {actions.shape[0]}"
        )

    clip_stride = clip_length if clip_stride is None else clip_stride
    clips: list[np.ndarray] = []
    for start in range(0, actions.shape[0] - clip_length + 1, clip_stride):
        clips.append(actions[start : start + clip_length])
        if max_clips is not None and len(clips) >= max_clips:
            break

    return mx.array(np.stack(clips, axis=0))


def make_minigrid_dataset(
    env: gym.Env,
    *,
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    clip_length: int = 4,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> tuple[mx.array, mx.array]:
    frames, actions = rollout_minigrid_frames(
        env=env,
        num_steps=num_steps,
        tile_size=tile_size,
        seed=seed,
    )
    frame_clips = frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    action_clips = actions_to_clips(
        actions,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    return frame_clips, action_clips


def generate_minigrid_video(
    model: UNet3D,
    *,
    initial_clip: mx.array,
    initial_actions: mx.array,
    num_actions: int,
    num_new_frames: int,
    num_steps: int = 32,
    sample_fps: float = 2.0,
    save_dir: str | Path,
    seed: int = 0,
) -> mx.array:
    """Autoregressively extend `initial_clip` using random actions for new frames."""
    rng = np.random.default_rng(seed)
    initial_actions_np = np.asarray(initial_actions)
    batch_size = int(initial_clip.shape[0])
    extra_actions_np = rng.integers(
        0, num_actions, size=(batch_size, num_new_frames)
    ).astype(np.int32)
    full_actions = mx.array(
        np.concatenate([initial_actions_np, extra_actions_np], axis=1)
    )
    generated = generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=full_actions,
        num_steps=num_steps,
    )
    save_clip_previews(generated, save_dir, max_clips=batch_size, fps=sample_fps)
    print(f"saved generated video to: {save_dir}")
    return generated


@click.group()
def cli() -> None:
    """MiniGrid world-model pretraining and generation."""


def _env_options(func):
    func = click.option("--env-id", default="MiniGrid-Empty-8x8-v0")(func)
    func = click.option("--num-recording-steps", default=32, type=int)(func)
    func = click.option("--tile-size", default=8, type=int)(func)
    func = click.option("--seed", default=0, type=int)(func)
    func = click.option("--clip-length", default=4, type=int)(func)
    func = click.option("--clip-stride", default=None, type=int)(func)
    func = click.option("--max-clips", default=None, type=int)(func)
    return func


@cli.command(name="train")
@_env_options
@click.option("--save-dir", default="logs/minigrid-v1")
@click.option(
    "--load-dir",
    default=None,
    help="Resume training from a checkpoint saved in this directory.",
)
@click.option("--base-channels", default=16, type=int)
@click.option("--train-steps", default=10_000, type=int)
@click.option("--batch-size", default=8, type=int)
@click.option("--log-every", default=100, type=int)
@click.option("--preview-dir", default=None)
@click.option("--preview-clips", default=4, type=int)
@click.option("--preview-fps", default=2.0, type=float)
def train_cmd(
    env_id: str,
    num_recording_steps: int,
    tile_size: int,
    seed: int,
    clip_length: int,
    clip_stride: int | None,
    max_clips: int | None,
    save_dir: str,
    load_dir: str | None,
    base_channels: int,
    train_steps: int,
    batch_size: int,
    log_every: int,
    preview_dir: str | None,
    preview_clips: int,
    preview_fps: float,
) -> None:
    """Train the diffusion world model on MiniGrid rollouts."""
    env = gym.make(env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=num_recording_steps,
        tile_size=tile_size,
        seed=seed,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    print(f"env: {env_id}")
    print(f"clips shape: {tuple(clips.shape)}")
    print(f"action clips shape: {tuple(action_clips.shape)}")

    if preview_dir is not None:
        save_clip_previews(
            clips, preview_dir, max_clips=preview_clips, fps=preview_fps
        )
        print(f"saved previews to: {preview_dir}")

    save_path = Path(save_dir)
    num_actions = int(env.action_space.n)

    initial_model = None
    if load_dir is not None:
        initial_model = load_model(load_dir)
        print(f"resuming training from: {load_dir}")

    model, _ = train_on_dataset(
        clips,
        actions=action_clips,
        action_dim=num_actions,
        base_channels=base_channels,
        batch_size=batch_size,
        steps=train_steps,
        sample_fps=preview_fps,
        sample_dir=str(save_path),
        log_every=log_every,
        model=initial_model,
    )

    save_model(
        model,
        save_path,
        config={
            "in_channels": int(clips.shape[-1]),
            "out_channels": int(clips.shape[-1]),
            "base_channels": base_channels,
            "action_dim": num_actions,
        },
    )
    print(f"saved model to: {save_path}")


@cli.command(name="generate")
@_env_options
@click.option(
    "--load-dir",
    required=True,
    help="Directory containing the saved model to load.",
)
@click.option(
    "--save-dir",
    default=None,
    help="Output directory for generated previews. Defaults to <load-dir>/generated.",
)
@click.option("--generate-new-frames", default=32, type=int)
@click.option("--generate-num-steps", default=32, type=int)
@click.option("--num-samples", default=1, type=int)
@click.option("--preview-fps", default=2.0, type=float)
def generate_cmd(
    env_id: str,
    num_recording_steps: int,
    tile_size: int,
    seed: int,
    clip_length: int,
    clip_stride: int | None,
    max_clips: int | None,
    load_dir: str,
    save_dir: str | None,
    generate_new_frames: int,
    generate_num_steps: int,
    num_samples: int,
    preview_fps: float,
) -> None:
    """Load a pretrained model and autoregressively generate a video."""
    env = gym.make(env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=num_recording_steps,
        tile_size=tile_size,
        seed=seed,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    print(f"env: {env_id}")
    print(f"clips shape: {tuple(clips.shape)}")

    num_actions = int(env.action_space.n)
    model = load_model(load_dir)
    print(f"loaded model from: {load_dir}")

    out_dir = Path(save_dir) if save_dir is not None else Path(load_dir) / "generated"
    sample_count = min(num_samples, int(clips.shape[0]))
    generate_minigrid_video(
        model,
        initial_clip=clips[:sample_count],
        initial_actions=action_clips[:sample_count],
        num_actions=num_actions,
        num_new_frames=generate_new_frames,
        num_steps=generate_num_steps,
        sample_fps=preview_fps,
        save_dir=out_dir,
        seed=seed + 1,
    )


if __name__ == "__main__":
    cli()
