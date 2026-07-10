from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
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
REWARDS_FILE = "rewards.npy"


@dataclass(frozen=True)
class DatasetConfig:
    tile_size: int = 8
    seed: int = 0
    clip_length: int = 4
    clip_stride: int | None = None
    max_clips: int | None = None
    preview_fps: float = 2.0
    rollout_steps: int = 32
    preview_dir: str | None = None
    preview_clips: int = 4
    frame_skip: int = 1
    save_to_disk: bool = False
    save_dir: str | None = None
    warmup_steps: int = 50
    pad_multiple: int | None = None
    recompute: bool = False


def make_env(env_id: str, frame_skip: int = 1) -> gym.Env:
    if "MiniGrid" in env_id:
        return gym.make(env_id)

    if "doom" in env_id:
        return gym.make(env_id, continuous=False, frame_skip=frame_skip)

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
        rewards: array of shape (num_steps,) of float32. `rewards[i]` is the
            reward received for the step that produced `frames[i]`.
        episode_ends: exclusive end indices of each episode into frames/actions.
    """
    env = RGBImgObsWrapper(env, tile_size=tile_size)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    max_action_idx = env.action_space.n if max_action_idx == -1 else max_action_idx

    for _ in range(num_steps):
        action = env.action_space.sample() % max_action_idx
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = obs["image"]
        frames.append(np.asarray(frame))
        rewards.append(float(reward))
        if terminated or truncated:
            episode_ends.append(len(frames))
            env.reset()

    env.close()

    if not episode_ends or episode_ends[-1] < len(frames):
        episode_ends.append(len(frames))

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    print(f"Collected {len(frames)} frames over {len(episode_ends)} episodes")
    return stacked, action_array, reward_array, episode_ends


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
        rewards: (num_steps,) float32, summed over frame_skip.
        episode_ends: exclusive end indices of each episode into frames/actions.
    """

    def _warmup():
        for _ in range(warmup_steps):
            env.step(env.action_space.sample())

    env.reset(seed=seed)
    _warmup()

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    action = int(env.action_space.sample())

    for _ in range(num_steps):
        reset_happened = False
        step_reward = 0.0
        for _ in range(frame_skip):
            obs, reward, terminated, truncated, _ = env.step(action)
            step_reward += float(reward)
            if terminated or truncated:
                env.reset()
                _warmup()
                reset_happened = True
                break

        actions.append(action)
        frames.append(np.asarray(obs))
        rewards.append(step_reward)

        if reset_happened:
            # frames[-1] is the terminal frame - episode ends here
            episode_ends.append(len(frames))

        action = int(env.action_space.sample())

    env.close()

    if not episode_ends or episode_ends[-1] < len(frames):
        episode_ends.append(len(frames))

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    print(f"Collected {len(frames)} frames over {len(episode_ends)} episodes")
    return stacked, action_array, reward_array, episode_ends


def rollout_doom(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 0,
    warmup_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Roll out random discrete actions in a Box2D env and capture RGB frames.

    Returns:
        frames: (num_steps, H, W, 3) float32 in [-1, 1].
        actions: (num_steps,) int32.
        rewards: (num_steps,) float32, summed over frame_skip.
        episode_ends: exclusive end indices of each episode into frames/actions.
    """
    # print("Using Doom Env")
    env.reset(seed=seed)
    # env.action_space.sample()
    # print(env.metadata)
    # pprint(env.spec)
    warmup_steps = 5

    def _warmup():
        for _ in range(warmup_steps):
            env.step(env.action_space.sample())

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    episode_ends: list[int] = []
    action = int(env.action_space.sample())

    for i in range(num_steps):
        reset_happened = False
        step_reward = 0.0
        obs, reward, terminated, truncated, _ = env.step(action)
        step_reward += float(reward)
        if terminated or truncated:
            env.reset(seed=seed + i)
            reset_happened = True

        actions.append(action)
        frames.append(np.asarray(obs["screen"]))
        rewards.append(step_reward)

        if reset_happened:
            # frames[-1] is the terminal frame - episode ends here
            episode_ends.append(len(frames))

        action = int(env.action_space.sample())

    env.close()

    if not episode_ends or episode_ends[-1] < len(frames):
        episode_ends.append(len(frames))

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    print(f"Collected {len(frames)} frames over {len(episode_ends)} episodes")
    return stacked, action_array, reward_array, episode_ends


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
    return padded


def actions_to_clips(
    actions: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> np.ndarray:
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

    return np.stack(clips, axis=0)


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
    rewards: np.ndarray,
    episode_ends: list[int],
    *,
    clip_length: int,
    clip_stride: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice frames, actions, rewards and dones into clips that never cross
    episode boundaries.

    ``dones`` is a per-frame terminal mask derived from ``episode_ends`` (True on
    each episode's final frame), sliced alongside the other streams so callers
    can read a clip's terminal flag as ``done_clips[:, -1]`` — symmetric with the
    aligned action/reward at ``[:, -1]``.
    """
    starts = clip_starts_from_episodes(
        episode_ends, clip_length=clip_length, clip_stride=clip_stride
    )
    frame_dones = np.zeros(len(frames), dtype=bool)
    frame_dones[np.asarray(episode_ends) - 1] = True

    frame_clips = [frames[s : s + clip_length] for s in starts]
    action_clips = [actions[s : s + clip_length] for s in starts]
    reward_clips = [rewards[s : s + clip_length] for s in starts]
    done_clips = [frame_dones[s : s + clip_length] for s in starts]
    return (
        np.stack(frame_clips),
        np.stack(action_clips),
        np.stack(reward_clips),
        np.stack(done_clips),
    )


def rollout_env(
    env: gym.Env,
    *,
    num_steps: int = 256,
    seed: int = 42,
    tile_size: int = 8,
    max_action_idx: int = -1,
    warmup_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Dispatch to the right rollout fn for `env` and return the raw stream.

    Returns ``(frames, actions, rewards, episode_ends)`` — frames are float32
    in [-1, 1] with shape ``(T, H, W, C)``, actions are int32 ``(T,)``,
    rewards are float32 ``(T,)``.
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
    recompute: bool = False,
    return_dones: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | Path | None]:
    """Record clips from ``env``.

    If ``return_dones`` is True, the per-clip terminal mask ``done_sequence``
    (``(num_clips, clip_length)`` bool, terminal at ``[:, -1]``) is appended to
    the return tuple. Dones are not persisted, so a cached disk load returns
    ``None`` for them.
    """

    if save_to_disk:
        assert save_dir is not None
        if not recompute and Path(save_dir).exists():
            print(f"Loading precomputed rollouts from {save_dir}")
            result = (*load_rollouts(save_dir), save_dir)
            return (*result, None) if return_dones else result

    frames, actions, rewards, episode_ends = rollout_env(
        env,
        num_steps=num_steps,
        seed=seed,
        tile_size=tile_size,
        max_action_idx=max_action_idx,
        warmup_steps=warmup_steps,
    )

    if pad_multiple is not None:
        height, width = frames.shape[1:3]
        frames = pad_frames_to_multiple(frames, multiple=pad_multiple)
        print(f"padded frames from {(height, width)} to {frames.shape[1:3]}")

    frames, actions, rewards, dones = clips_from_episodes(
        frames,
        actions,
        rewards,
        episode_ends,
        clip_length=clip_length,
        clip_stride=clip_stride,
    )

    if save_to_disk:
        assert save_dir is not None
        save_rollouts(save_dir, frames, actions, rewards)

    result = (frames, actions, rewards, save_dir)
    return (*result, dones) if return_dones else result


def save_rollouts(
    save_dir: str | Path,
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray | None = None,
) -> Path:
    """Persist a recorded rollout to `save_dir` as memmap-friendly `.npy` files.

    Writes `frames.npy` (T, H, W, C) float32, `actions.npy` (T,) int32, and
    `rewards.npy` (T,) float32 when provided. Reload with `load_rollouts`.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / FRAMES_FILE, np.ascontiguousarray(frames, dtype=np.float32))
    np.save(save_dir / ACTIONS_FILE, np.asarray(actions, dtype=np.int32))
    if rewards is not None:
        np.save(save_dir / REWARDS_FILE, np.asarray(rewards, dtype=np.float32))
    print(f"saved rollout ({frames.shape[0]} frames) to: {save_dir}")
    return save_dir


def load_rollouts(
    data_dir: str | Path,
    *,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a rollout saved by `save_rollouts`.

    When `mmap` is True the large `frames` array is returned as a read-only
    `np.memmap` so individual clips are paged off disk on access rather than
    loaded into RAM up front. `actions`/`rewards` are small and always read
    fully. `rewards` is None for datasets recorded before rewards were saved.
    """
    data_dir = Path(data_dir)
    frames = np.load(data_dir / FRAMES_FILE, mmap_mode="r" if mmap else None)
    actions = np.load(data_dir / ACTIONS_FILE)
    rewards_path = data_dir / REWARDS_FILE
    rewards = np.load(rewards_path) if rewards_path.exists() else None
    return frames, actions, rewards


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
    rewards: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Sample a random batch. Returns 2 arrays, or 3 when `rewards` is given."""
    if batch_size >= videos.shape[0]:
        indices = slice(None, batch_size)
    else:
        indices = np.random.randint(0, videos.shape[0], size=(batch_size,))

    if rewards is None:
        return videos[indices], actions[indices]
    return videos[indices], actions[indices], rewards[indices]
