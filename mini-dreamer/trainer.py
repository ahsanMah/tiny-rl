import gymnasium as gym
import mlx.core as mx
import numpy as np
from tqdm import tqdm

from data import Dataset, make_env, record_rollouts
from diffusion import FlowMatchingTrainer, load_model, save_model
from unet import UNet3D
from vae import WaveletVAE, decode_latents, encode_clips, load_vae


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

    def decoder(self, latents: mx.array) -> mx.array:
        return decode_latents(self.vae, latents)

    def image_encoder(self, x: mx.array) -> mx.array:
        if self.vae is not None:
            return encode_clips(self.vae, x)
        return x

    def encode(self, x: mx.array, t: mx.array, context: mx.array) -> mx.array:
        x = self.image_encoder(x)
        xmid, _, _ = self.model.encode(x, t, context=context)
        return xmid


def collect_and_encode_rollout(
    env: gym.Env,
    model: WorldModel,
    *,
    num_steps: int = 256,
    seed: int = 0,
    encode_batch_size: int = 64,
) -> tuple[Dataset, tuple[mx.array, np.ndarray, np.ndarray]]:
    """Roll out ``env`` and encode it into per-step world-model state embeddings.

    Collects a raw ``(frames, actions, rewards)`` stream via ``rollout_env`` and
    slides a ``model.max_context_size``-frame window over it (respecting episode
    boundaries) to produce a bottleneck embedding per step. Each window is fed to
    ``model.encode`` with ``t=1`` and its context's last action set to the NULL
    action, yielding the deterministic "state embedding for acting" described in
    ``UNet3D.encode``.

    Window ``[s, s+L)`` represents the state at frame ``s+L-1``; its aligned
    ``action``/``reward`` are the ones recorded at that same frame.

    If ``model.vae`` is set, frames are latent-encoded with ``encode_clips``
    before being passed to the world model (latent-space models). The returned
    ``Dataset`` carries the same encoder so its raw-pixel clips are encoded to
    latents on the fly when sampled for world-model training.

    The embedding is the raw ``xmid`` bottleneck, intentionally left spatial
    (not pooled/flattened) so a downstream policy can consume it convolutionally.

    Args:
        encode_batch_size: number of windows encoded per ``model.encode`` call.

    Returns:
        rollout_dataset: ``Dataset`` over the raw ``(frames, actions, rewards)``
            clips, for training the world model on real-env transitions.
        policy_rollouts: ``(embeddings, actions, rewards)`` where
            - embeddings: ``(num_windows, S/4, H/4, W/4, 4 * base_channels)``
              float32 state embeddings,
            - actions: ``(num_windows,)`` int32 — action taken from each state,
            - rewards: ``(num_windows,)`` float32 — reward received at each state.
    """
    context_size = model.max_context_size + 1

    videos, action_sequence, reward_sequence, save_dir = record_rollouts(
        env=env,
        num_steps=num_steps,
        seed=seed,
        clip_length=context_size,
        clip_stride=1,
        save_to_disk=True,
        save_dir="/tmp/dreamer/test",
        recompute=True,
        pad_multiple=32,
    )

    rollout_dataset = Dataset(save_dir, encoder=model.image_encoder)

    _embeddings: list[mx.array] = []
    null_action = model.null_action
    for batch_start in tqdm(
        range(0, len(videos), encode_batch_size), desc="encoding rollout"
    ):
        video_batch = videos[batch_start : batch_start + encode_batch_size]
        video_batch = mx.array(video_batch)

        action_batch = action_sequence[batch_start : batch_start + encode_batch_size]
        # Last action is the one we are about to choose -> NULL it out so the
        # embedding is the pre-action state.
        action_batch = mx.array(action_batch)
        action_batch[:, -1] = null_action

        t = mx.ones((video_batch.shape[0],))
        embs = model.encode(video_batch, t, context=action_batch)
        _embeddings.append(embs)

    embeddings = mx.concatenate(_embeddings, axis=0)

    # Align actions/rewards to the frame each window ends on (s + L - 1).
    aligned_actions = action_sequence[:, -1].astype(np.int32)
    aligned_rewards = reward_sequence[:, -1].astype(np.float32)

    policy_rollouts = (embeddings, aligned_actions, aligned_rewards)

    return rollout_dataset, policy_rollouts


def update_world_model(
    trainer: FlowMatchingTrainer,
    dataset: Dataset,
    *,
    num_steps: int = 10,
    batch_size: int = 8,
) -> float:
    """Run ``num_steps`` flow-matching updates on real-env clips.

    Samples latent-encoded ``(clips, actions, rewards)`` batches from ``dataset``
    and steps the (compiled) flow-matching trainer, mutating the world model and
    its EMA shadow in place. Returns the mean flow loss over the updates.
    """
    total_loss = mx.array(0.0)
    pbar = tqdm(range(num_steps), desc="world model update")
    for step in pbar:
        batch, actions, rewards = dataset.sample_train_batch(batch_size)
        loss, _reward_loss = trainer.compiled_train_step(batch, actions, rewards)
        total_loss = total_loss + loss
        mx.async_eval(total_loss)
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")

    return float(total_loss) / num_steps


def train():

    load_dir = "logs/vizdoom-latent-rewards"
    gym_env = make_env("VizdoomBasic-v1")
    # Online + EMA copies start from the same pretrained weights.
    model = load_model(load_dir)
    ema_model = load_model(load_dir)
    vae = load_vae("logs/vizdoom-vae")
    world_model = WorldModel(ema_model, vae)

    world_model_trainer = FlowMatchingTrainer(
        model,
        ema_model,
        learning_rate=3e-4,
        reward_loss_weight=1e-1,
    )

    rollout_dataset, policy_rollouts = collect_and_encode_rollout(gym_env, world_model)

    # train world (diffusion) model using real frames
    wm_loss = update_world_model(world_model_trainer, rollout_dataset)
    print(f"world model loss: {wm_loss:.4f}")

    # train policy using world model embeddings
    # i.e fill buffer with observations, advantages etc
    # and do ppo update


if __name__ == "__main__":
    train()
