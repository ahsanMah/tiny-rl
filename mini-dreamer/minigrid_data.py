from __future__ import annotations

import argparse

import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np

from diffusion import train_on_dataset
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
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    action = 0
    for _ in range(num_steps):
        frame = env.unwrapped.get_frame(tile_size=tile_size, highlight=highlight)
        frames.append(np.asarray(frame))
        # action = int(rng.integers(0, env.action_space.n))
        action = (action + 1) % 3
        actions.append(action)
        _, _, terminated, truncated, _ = env.step(action)
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
    """Slice a 1-D action stream into (num_clips, clip_length) windows.

    Uses the same start indices as `frames_to_clips` so clip i's actions are
    aligned to clip i's frames.
    """
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

    env = gym.make(args.env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=args.num_recording_steps,
        tile_size=args.tile_size,
        seed=args.seed,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        max_clips=args.max_clips,
    )
    print(f"env: {args.env_id}")
    print(f"clips shape: {tuple(clips.shape)}")
    print(f"action clips shape: {tuple(action_clips.shape)}")

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
        actions=action_clips,
        action_dim=env.action_space.n,
        batch_size=4,
        steps=10_000,
        sample_fps=2.0,
        sample_dir="logs/minigrid-v0",
        log_every=100,
    )


if __name__ == "__main__":
    main()
