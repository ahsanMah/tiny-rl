from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mlx.core as mx
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from diffusion import generate_video, sample_euler_to_mp4
from unet import UNet3D
from video_utils import frames_to_clips, save_clip_previews


def rollout_minigrid_frames(
    env: gym.Env,
    *,
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    max_action_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Roll out random actions in a MiniGrid env and capture tile-rendered RGB frames.

    Returns:
        frames: array of shape (num_steps, H, W, 3) of float32 in [-1, 1].
        actions: array of shape (num_steps,) of int32. `actions[i]` is the
            action taken at `frames[i]` (which produced the next frame).
        episode_ends: exclusive end indices of each episode into frames/actions.
    """
    env = RGBImgObsWrapper(env, tile_size=tile_size)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    episode_ends: list[int] = []
    max_action_idx = env.action_space.n if max_action_idx == -1 else max_action_idx

    for _ in range(num_steps):
        action = env.action_space.sample() % max_action_idx
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)

        frame = obs["image"]
        frames.append(np.asarray(frame))
        if terminated or truncated:
            episode_ends.append(len(frames))
            env.reset()

    env.close()

    if not episode_ends or episode_ends[-1] < len(frames):
        episode_ends.append(len(frames))

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    print(f"Collected {len(frames)} frames over {len(episode_ends)} episodes")
    return stacked, action_array, episode_ends


def rollout_box2d_frames(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 0,
    warmup_steps: int = 50,
    frame_skip: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Roll out random discrete actions in a Box2D env and capture RGB frames.

    Returns:
        frames: (num_steps, H, W, 3) float32 in [-1, 1].
        actions: (num_steps,) int32.
        episode_ends: exclusive end indices of each episode into frames/actions.
    """

    def _warmup():
        for _ in range(warmup_steps):
            env.step(env.action_space.sample())

    env.reset(seed=seed)
    _warmup()

    frames: list[np.ndarray] = []
    actions: list[int] = []
    episode_ends: list[int] = []
    action = int(env.action_space.sample())

    for _ in range(num_steps):
        reset_happened = False
        for _ in range(frame_skip):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
                _warmup()
                reset_happened = True
                break

        actions.append(action)
        frames.append(np.asarray(obs))

        if reset_happened:
            # frames[-1] is the terminal frame - episode ends here
            episode_ends.append(len(frames))

        action = int(env.action_space.sample())

    env.close()

    if not episode_ends or episode_ends[-1] < len(frames):
        episode_ends.append(len(frames))

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    print(f"Collected {len(frames)} frames over {len(episode_ends)} episodes")
    return stacked, action_array, episode_ends


def actions_to_clips(
    actions: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> mx.array:
    """Slice a 1-D action stream into (num_clips, clip_length) windows."""
    if actions.ndim != 1:
        raise ValueError(f"Expected actions with shape (T,), got {actions.shape}")
    if actions.shape[0] < clip_length:
        raise ValueError(
            f"Need at least {clip_length} actions, but only have {actions.shape[0]}"
        )

    clip_stride = 1 if clip_stride is None else clip_stride
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be > 0, got {clip_stride}")
    clips: list[np.ndarray] = []
    for start in range(0, actions.shape[0] - clip_length + 1, clip_stride):
        clips.append(actions[start : start + clip_length])

    return mx.array(np.stack(clips, axis=0))


def clips_from_episodes(
    frames: np.ndarray,
    actions: np.ndarray,
    episode_ends: list[int],
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> tuple[mx.array, mx.array]:
    """Slice frames and actions into clips that never cross episode boundaries."""
    clip_stride = 1 if clip_stride is None else clip_stride
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be > 0, got {clip_stride}")

    frame_clips: list[np.ndarray] = []
    action_clips: list[np.ndarray] = []
    ep_start = 0
    for ep_end in episode_ends:
        ep_len = ep_end - ep_start
        if ep_len >= clip_length:
            for s in range(0, ep_len - clip_length + 1, clip_stride):
                frame_clips.append(frames[ep_start + s : ep_start + s + clip_length])
                action_clips.append(actions[ep_start + s : ep_start + s + clip_length])
        ep_start = ep_end

    if not frame_clips:
        raise ValueError(
            f"No clips formed: all episodes shorter than clip_length={clip_length}"
        )

    return mx.array(np.stack(frame_clips)), mx.array(np.stack(action_clips))


def make_dataset(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 42,
    clip_length: int = 4,
    clip_stride: int | None = None,
    tile_size: int = 8,
    max_action_idx: int = -1,
) -> tuple[mx.array, mx.array]:
    if "MiniGrid" in (env.spec.id if env.spec else ""):
        frames, actions, episode_ends = rollout_minigrid_frames(
            env=env,
            num_steps=num_steps,
            tile_size=tile_size,
            seed=seed,
            max_action_idx=max_action_idx,
        )
    else:
        frames, actions, episode_ends = rollout_box2d_frames(
            env=env, num_steps=num_steps, seed=seed
        )
    return clips_from_episodes(
        frames, actions, episode_ends, clip_length=clip_length, clip_stride=clip_stride
    )


def generate_env_video(
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
    actions_pool: list[int] | None = None,
) -> mx.array:
    """Autoregressively extend `initial_clip` using random actions for new frames."""
    rng = np.random.default_rng(seed)
    initial_actions_np = np.asarray(initial_actions)
    batch_size = int(initial_clip.shape[0])
    if actions_pool is not None:
        extra_actions_np = rng.choice(actions_pool, size=(batch_size, num_new_frames))
    else:
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
    save_clip_previews(
        generated,
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
    )
    print(f"saved generated video to: {save_dir}")
    return generated
