from __future__ import annotations

import argparse

import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np

from diffusion import train_on_dataset
from video_utils import frames_to_clips, save_clip_previews


def rollout_minigrid_frames(
    *,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    highlight: bool = True,
) -> np.ndarray:
    """Roll out random actions in a MiniGrid env and capture tile-rendered RGB frames.

    Returns an array of shape (num_steps, H, W, 3) of float32 in [-1, 1].
    Episodes auto-reset on terminate/truncate so the frame stream is contiguous.
    """
    env = gym.make(env_id)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    for _ in range(num_steps):
        frame = env.unwrapped.get_frame(tile_size=tile_size, highlight=highlight)
        frames.append(np.asarray(frame))
        action = int(rng.integers(0, env.action_space.n))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    env.close()

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    return stacked


def make_minigrid_dataset(
    *,
    env_id: str = "MiniGrid-Empty-8x8-v0",
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    clip_length: int = 4,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> mx.array:
    frames = rollout_minigrid_frames(
        env_id=env_id,
        num_steps=num_steps,
        tile_size=tile_size,
        seed=seed,
    )
    return frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Roll out a MiniGrid env with random actions and build a clip dataset."
    )
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--num-recording-steps", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--clip-stride", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--preview-dir", type=str, default=None)
    parser.add_argument("--preview-clips", type=int, default=4)
    parser.add_argument("--preview-fps", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    clips = make_minigrid_dataset(
        env_id=args.env_id,
        num_steps=args.num_recording_steps,
        tile_size=args.tile_size,
        seed=args.seed,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        max_clips=args.max_clips,
    )
    print(f"env: {args.env_id}")
    print(f"clips shape: {tuple(clips.shape)}")

    if args.preview_dir is not None:
        save_clip_previews(
            clips,
            args.preview_dir,
            max_clips=args.preview_clips,
            fps=args.preview_fps,
        )
        print(f"saved previews to: {args.preview_dir}")

    train_on_dataset(
        clips,
        batch_size=8,
        steps=1000,
        sample_fps=2.0,
        sample_dir="logs/minigrid-v0",
    )


if __name__ == "__main__":
    main()
