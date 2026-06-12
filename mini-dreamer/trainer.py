from __future__ import annotations

import gymnasium as gym
import mlx.core as mx
import numpy as np

from data import clip_starts_from_episodes, clips_from_episodes, make_env, rollout_env
from diffusion import FlowMatchingTrainer, load_model, save_model
from unet import UNet3D
from vae import WaveletVAE, encode_clips, load_vae


class WorldModel:
    def __init__(self, model: UNet3D, vae: WaveletVAE | None = None):
        self.model = model
        self.vae = vae

    @property
    def null_action(self) -> int:
        return self.model.null_action

    @property
    def max_context_size(self) -> int:
        return self.model.max_context_size

    def decode(self, x: mx.array, t: mx.array, context: mx.array) -> mx.array:
        pass

    def encode(self, x: mx.array, t: mx.array, context: mx.array) -> mx.array:
        if self.vae is not None:
            mean, _ = self.vae.encode(x)
            x = mean / self.vae.latent_scale
        xmid, _, _ = self.model.encode(x, t, context=context)
        return xmid


def collect_and_encode_rollout(
    env: gym.Env,
    model: WorldModel,
    *,
    num_steps: int = 16,
    seed: int = 0,
    encode_batch_size: int = 64,
) -> dict[str, tuple[mx.array, np.ndarray, np.ndarray]]:
    """Roll out ``env`` and encode it into per-step world-model state embeddings.

    Collects a raw ``(frames, actions, rewards)`` stream via ``rollout_env`` and
    slides a ``model.max_context_size``-frame window over it (respecting episode
    boundaries) to produce a bottleneck embedding per step. Each window is fed to
    ``model.encode`` with ``t=1`` and its context's last action set to the NULL
    action, yielding the deterministic "state embedding for acting" described in
    ``UNet3D.encode``. The embedding is the spatio-temporal mean-pool of ``xmid``.

    Window ``[s, s+L)`` represents the state at frame ``s+L-1``; its aligned
    ``action``/``reward`` are the ones recorded at that same frame.

    Args:
        vae: if set, frames are latent-encoded with ``encode_clips`` before being
            passed to the world model (latent-space models).
        encode_batch_size: number of windows encoded per ``model.encode`` call.

    Returns:
        embeddings: ``(num_windows, 4 * base_channels)`` float32 state embeddings.
        actions: ``(num_windows,)`` int32 — action taken from each state.
        rewards: ``(num_windows,)`` float32 — reward received at each state.
    """
    rollouts = {}
    frames, actions, rewards, episode_ends = rollout_env(
        env, num_steps=num_steps, seed=seed
    )

    context_size = model.max_context_size + 1
    videos, action_sequence, reward_sequence = clips_from_episodes(
        frames, actions, rewards, episode_ends, clip_length=context_size, clip_stride=1
    )

    rollouts["world_model"] = (videos, action_sequence, reward_sequence)

    _embeddings: list[mx.array] = []
    null_action = model.null_action
    for batch_start in range(0, len(videos), encode_batch_size):
        video_batch = videos[batch_start : batch_start + encode_batch_size]
        video_batch = mx.array(video_batch)

        action_batch = action_sequence[batch_start : batch_start + encode_batch_size]
        # Last action is the one we are about to choose -> NULL it out so the
        # embedding is the pre-action state.
        action_batch = mx.array(action_batch)
        action_batch[:, -1] = null_action

        t = mx.ones((video_batch.shape[0],))
        embs = model.encode(video_batch, t, context=mx.array(action_batch))
        _embeddings.append(embs)

    embeddings = mx.concatenate(_embeddings, axis=0)

    # Align actions/rewards to the frame each window ends on (s + L - 1).
    _actions, _rewards = [], []
    for act, rew in zip(action_sequence, reward_sequence):
        _actions.append(act[-1])
        _rewards.append(rew[-1])

    aligned_actions = np.array(_actions).astype(np.int32)
    aligned_rewards = np.array(_rewards).astype(np.float32)

    rollouts["policy"] = (embeddings, aligned_actions, aligned_rewards)

    return rollouts


def train():

    gym_env = make_env("VizdoomBasic-v1")
    model = load_model("logs/vizdoom-latent-rewards")
    vae = load_vae("logs/vizdoom-vae")
    world_model = WorldModel(model, vae)

    rollouts = collect_and_encode_rollout(gym_env, world_model)

    print(rollouts)


if __name__ == "__main__":
    train()
