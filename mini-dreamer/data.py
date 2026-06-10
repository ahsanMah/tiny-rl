from pathlib import Path
from pprint import pprint
from typing import Callable

import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper
from vizdoom import gymnasium_wrapper

# def _is_minigrid(env_id: str) -> bool:
#     return "MiniGrid" in env_id

# Files written by `save_rollout`. `frames` is stored as a plain `.npy` (not
# bundled into a single `.npz`) because a zip archive cannot be memory-mapped —
# keeping it standalone lets `load_rollout(mmap=True)` page clips off disk.
FRAMES_FILE = "frames.npy"
ACTIONS_FILE = "actions.npy"


def make_env(env_id: str) -> gym.Env:
    if "MiniGrid" in env_id:
        return gym.make(env_id)

    if "doom" in env_id:
        return gym.make(env_id, continuous=False, frame_skip=4)

    return gym.make(env_id, continuous=False)


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


def rollout_doom(
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
    print("Using Doom Env")
    env.reset(seed=seed)
    env.action_space.sample()
    print(env.metadata)
    pprint(env.spec)
    warmup_steps = 5

    def _warmup():
        for _ in range(warmup_steps):
            env.step(env.action_space.sample())

    # _warmup()
    # print(f"warmup_steps={warmup_steps}")

    frames: list[np.ndarray] = []
    actions: list[int] = []
    episode_ends: list[int] = []
    action = int(env.action_space.sample())

    for _ in range(num_steps):
        reset_happened = False
        for _ in range(frame_skip):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=seed)
                # print(f"reset happened at {len(frames)}")
                # _warmup()
                reset_happened = True
                break

        actions.append(action)
        frames.append(np.asarray(obs["screen"]))

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


def pad_frames_to_multiple(frames: np.ndarray, *, multiple: int = 16) -> np.ndarray:
    """Pad the spatial dims of `frames` (T, H, W, C) up to the next multiple.

    Padding is centered (split between both sides) and filled with -1.0,
    i.e. black for frames normalized to [-1, 1].
    """
    _, height, width, _ = frames.shape
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return frames

    top, left = pad_h // 2, pad_w // 2
    pad_widths = ((0, 0), (top, pad_h - top), (left, pad_w - left), (0, 0))
    padded = np.pad(frames, pad_widths, mode="constant", constant_values=-1.0)
    print(f"padded frames from {(height, width)} to {padded.shape[1:3]}")
    return padded


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


def clip_starts_from_episodes(
    episode_ends: list[int],
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> list[int]:
    """Absolute start indices of clips that never cross an episode boundary.

    Shared by the in-memory (`clips_from_episodes`) and on-disk/memmap
    (`MemmapClipDataset`) paths so the boundary logic lives in one place.
    """
    clip_stride = 1 if clip_stride is None else clip_stride
    if clip_stride <= 0:
        raise ValueError(f"clip_stride must be > 0, got {clip_stride}")

    starts: list[int] = []
    ep_start = 0
    for ep_end in episode_ends:
        ep_len = ep_end - ep_start
        if ep_len >= clip_length:
            for s in range(0, ep_len - clip_length + 1, clip_stride):
                starts.append(ep_start + s)
        ep_start = ep_end

    if not starts:
        raise ValueError(
            f"No clips formed: all episodes shorter than clip_length={clip_length}"
        )

    return starts


def clips_from_episodes(
    frames: np.ndarray,
    actions: np.ndarray,
    episode_ends: list[int],
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice frames and actions into clips that never cross episode boundaries."""
    starts = clip_starts_from_episodes(
        episode_ends, clip_length=clip_length, clip_stride=clip_stride
    )
    frame_clips = [frames[s : s + clip_length] for s in starts]
    action_clips = [actions[s : s + clip_length] for s in starts]
    return np.stack(frame_clips), np.stack(action_clips)


def rollout_env(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 42,
    tile_size: int = 8,
    max_action_idx: int = -1,
    warmup_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Dispatch to the right rollout fn for `env` and return the raw stream.

    Returns ``(frames, actions, episode_ends)`` — frames are float32 in
    [-1, 1] with shape ``(T, H, W, C)``, actions are int32 ``(T,)``.
    """
    env_id = env.spec.id if env.spec else ""
    if "MiniGrid" in env_id:
        return rollout_minigrid_frames(
            env=env,
            num_steps=num_steps,
            tile_size=tile_size,
            seed=seed,
            max_action_idx=max_action_idx,
        )
    if "Vizdoom" in env_id:
        return rollout_doom(
            env=env,
            num_steps=num_steps,
            seed=seed,
            warmup_steps=warmup_steps,
        )
    return rollout_box2d_frames(
        env=env,
        num_steps=num_steps,
        seed=seed,
        warmup_steps=warmup_steps,
    )


def record_rollouts(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 42,
    clip_length: int = 4,
    clip_stride: int | None = None,
    tile_size: int = 8,
    max_action_idx: int = -1,
    warmup_steps: int = 50,
    save_to_disk: bool = False,
    save_dir: str | Path | None = None,
    pad_multiple: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str | Path | None]:
    frames, actions, episode_ends = rollout_env(
        env,
        num_steps=num_steps,
        seed=seed,
        tile_size=tile_size,
        max_action_idx=max_action_idx,
        warmup_steps=warmup_steps,
    )

    if pad_multiple is not None:
        frames = pad_frames_to_multiple(frames, multiple=pad_multiple)

    frames, actions = clips_from_episodes(
        frames, actions, episode_ends, clip_length=clip_length, clip_stride=clip_stride
    )

    if save_to_disk:
        assert save_dir is not None
        save_rollouts(save_dir, frames, actions)

    return frames, actions, save_dir


def save_rollouts(
    save_dir: str | Path,
    frames: np.ndarray,
    actions: np.ndarray,
) -> Path:
    """Persist a recorded rollout to `save_dir` as memmap-friendly `.npy` files.

    Writes `frames.npy` (T, H, W, C) float32, `actions.npy` (T,) int32, and
    `episode_ends.npy` (num_episodes,) int32. Reload with `load_rollout`.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / FRAMES_FILE, np.ascontiguousarray(frames, dtype=np.float32))
    np.save(save_dir / ACTIONS_FILE, np.asarray(actions, dtype=np.int32))
    print(f"saved rollout ({frames.shape[0]} frames) to: {save_dir}")
    return save_dir


def load_rollouts(
    data_dir: str | Path,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a rollout saved by `save_rollout`.

    When `mmap` is True the large `frames` array is returned as a read-only
    `np.memmap` so individual clips are paged off disk on access rather than
    loaded into RAM up front. `actions`/`episode_ends` are small and always
    read fully.
    """
    data_dir = Path(data_dir)
    frames = np.load(data_dir / FRAMES_FILE, mmap_mode="r" if mmap else None)
    actions = np.load(data_dir / ACTIONS_FILE)
    return frames, actions


def _split_size(dataset_size: int, val_fraction: float) -> int:
    """Number of held-out validation items, leaving at least one for training."""
    if dataset_size < 2:
        raise ValueError(
            f"Need at least 2 clips to make a train/val split, got {dataset_size}"
        )
    val_size = max(1, int(round(dataset_size * val_fraction)))
    return min(val_size, dataset_size - 1)


def sample_batch(
    videos: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size >= videos.shape[0]:
        return videos[:batch_size], actions[:batch_size]

    indices = np.random.randint(0, videos.shape[0], size=(batch_size,))
    return (videos[indices], actions[indices])


class Dataset:
    """Train/val view over clip tensors already materialised in memory or on disk."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        val_fraction: float = 0.05,
        encoder: Callable | None = None,
        memory_map: bool = False,
    ):
        videos, actions = load_rollouts(data_dir, mmap=memory_map)

        dataset_size = int(videos.shape[0])
        val_size = _split_size(dataset_size, val_fraction)
        train_size = dataset_size - val_size
        self.encoder = encoder

        self.train_videos = videos[:train_size]
        self.train_actions = actions[:train_size]
        self.val_videos = videos[train_size:]
        self.val_actions = actions[train_size:]

        self.dataset_size = dataset_size
        self.train_size = train_size
        self.val_size = int(self.val_videos.shape[0])
        self.num_channels = int(videos.shape[-1])

    def _build_tensor(
        self,
        videos: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[mx.array, mx.array]:

        video_batch = mx.array(videos)
        action_batch = mx.array(actions)

        if self.encoder is not None:
            video_batch = self.encoder(video_batch)
        return video_batch, action_batch

    def sample_train_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        videos, actions = sample_batch(
            self.train_videos, self.train_actions, batch_size
        )
        return self._build_tensor(videos, actions)

    def sample_val_batch(self, batch_size: int) -> tuple[mx.array, mx.array]:
        videos, actions = sample_batch(self.val_videos, self.val_actions, batch_size)
        return self._build_tensor(videos, actions)

    def val_clips(self, num_clips: int) -> tuple[mx.array, mx.array]:
        return self._build_tensor(
            self.val_videos[:num_clips], self.val_actions[:num_clips]
        )
