from __future__ import annotations

import functools
import json
from pprint import pprint
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import lax, random
from safetensors.flax import save_file

from logger_utils import RLLogger
from jax_utils import (
    ema_update,
    flat_params,
    linear_warmup_decay_schedule,
    load_flat_params,
)
from unet_jax import UNet3D
from video_utils import save_clip_previews, save_diffusion_mp4
from data import _split_size, load_rollouts, sample_batch

NoiseDistribution = Literal["uniform", "logitnorm", "normal"]

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
        videos, actions, rewards = load_rollouts(data_dir, mmap=memory_map)
        self.has_rewards = rewards is not None
        if rewards is None:
            rewards = np.zeros(actions.shape, dtype=np.float32)

        dataset_size = int(videos.shape[0])
        val_size = _split_size(dataset_size, val_fraction)
        train_size = dataset_size - val_size
        self.encoder = encoder

        self.train_videos = videos[:train_size]
        self.train_actions = actions[:train_size]
        self.train_rewards = rewards[:train_size]
        self.val_videos = videos[train_size:]
        self.val_actions = actions[train_size:]
        self.val_rewards = rewards[train_size:]

        self.dataset_size = dataset_size
        self.train_size = train_size
        self.val_size = int(self.val_videos.shape[0])
        self.num_channels = int(videos.shape[-1])
        # Env action-space size; an action id >= num_actions would index out
        # of the model's embedding table (which NaNs silently under jit).
        self.num_actions = int(actions.max()) + 1

        print(f"setup dataset from {data_dir}")
        print(f"{self.train_size = } - {self.val_size = }")

    def _build_tensor(
        self,
        videos: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        video_batch = jnp.array(videos)
        action_batch = jnp.array(actions)
        reward_batch = jnp.array(rewards)

        if self.encoder is not None:
            video_batch = self.encoder(video_batch)
        return video_batch, action_batch, reward_batch

    def sample_train_batch(
        self, batch_size: int
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        videos, actions, rewards = sample_batch(
            self.train_videos, self.train_actions, batch_size, self.train_rewards
        )
        return self._build_tensor(videos, actions, rewards)

    def sample_val_batch(self, batch_size: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        videos, actions, rewards = sample_batch(
            self.val_videos, self.val_actions, batch_size, self.val_rewards
        )
        return self._build_tensor(videos, actions, rewards)

    def val_clips(self, num_clips: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._build_tensor(
            self.val_videos[:num_clips],
            self.val_actions[:num_clips],
            self.val_rewards[:num_clips],
        )

@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    base_channels: int = 16
    max_context_size: int = 3
    num_transformer_blocks: int = 2
    use_wavelet: bool = False
    predict_reward: bool = False


@dataclass(frozen=True)
class TrainConfig:
    train_steps: int = 1_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    lr_final: float | None = None
    lr_warmup_steps: int = 500
    lr_hold_steps: int = 3000
    ema_decay: float = 0.99
    action_dropout: float = 0.1
    reward_loss_weight: float = 0.0
    reward_t_threshold: float = 0.6
    # Context-noise augmentation: conditioning frames are corrupted to a flow
    # level sampled uniformly in [min_context_t, 1.0] (1.0 = clean). Lower =
    # stronger augmentation against autoregressive drift; 1.0 disables it.
    min_context_t: float = 0.5
    log_every: int = 50
    save_dir: str | Path | None = None
    load_dir: str | None = None
    num_gen_samples: int = 4
    sample_steps: int = 32
    sampling_distribution: str = "uniform"
    log_tensorboard: bool = False


def make_final_frame_mask(x: jnp.ndarray) -> jnp.ndarray:
    frame_mask = (jnp.arange(x.shape[1]) == (x.shape[1] - 1)).astype(x.dtype)
    return jnp.reshape(frame_mask, (1, x.shape[1], 1, 1, 1))


def _loss_at_t(
    model: UNet3D,
    x1: jnp.ndarray,
    actions: jnp.ndarray,
    t: jnp.ndarray,
    *,
    noise: jnp.ndarray,
    t_ctx: jnp.ndarray | None = None,
    rewards: jnp.ndarray | None = None,
    reward_loss_weight: float = 0.0,
    reward_t_threshold: float = 0.0,
    return_eval_aux: bool = False,
) -> (
    tuple[jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Flow-matching loss at the given t.

    Unlike the MLX version, ``noise`` is an explicit argument (same shape as
    ``x1``) rather than sampled inside: this keeps the function pure
    (jit-friendly) and lets parity tests inject identical noise into both
    frameworks. The trainer's ``reward_loss_weight``/``reward_t_threshold``
    come in as kwargs since this is a free function.

    By default returns ``(total_loss, reward_loss)`` where ``total_loss``
    is the flow loss plus ``reward_loss_weight * reward_loss``. The reward
    loss is an MSE on the final frame's reward, computed only when
    ``rewards`` (B,) is given and ``reward_loss_weight > 0``, and only over
    batch elements with ``t >= reward_t_threshold`` (near-clean inputs —
    heavily noised frames carry too little state to predict reward from).

    If ``return_eval_aux`` is True, returns ``(loss, recon_mse, x1_pred,
    r2)`` for eval/logging instead (flow loss only):

    - ``recon_mse``: MSE of the one-step x1 reconstruction
        ``x1_pred = xt + (1 - t) * v_pred``.
    - ``x1_pred``: the prediction itself, shape ``(B, 1, H, W, C)``.
    - ``r2``: ``1 - loss / E[target_velocity^2]`` — fraction of target
        variance explained, scale-free baseline.
    """
    mask = make_final_frame_mask(x1)
    if t_ctx is None:
        t_ctx = jnp.ones_like(t)

    # Diffusion-forcing style corruption: the target (final) frame is noised
    # to level ``t`` while the conditioning frames are noised to level
    # ``t_ctx`` (1.0 = clean). Training on imperfect history teaches the
    # model to denoise from its own (error-carrying) rollout frames instead
    # of always-clean ground truth, which is what curbs autoregressive
    # drift. ``level`` is ``t`` on the final frame and ``t_ctx`` elsewhere.
    t_view = jnp.reshape(t, (x1.shape[0], 1, 1, 1, 1)) * mask
    t_ctx_view = jnp.reshape(t_ctx, (x1.shape[0], 1, 1, 1, 1)) * (1 - mask)
    level = t_view + t_ctx_view
    xt = (1.0 - level) * noise + level * x1
    target_velocity = mask * (x1 - noise)
    xmid, skips, time_context = model.encode(xt, t, context=actions, t_ctx=t_ctx)
    pred_velocity = model.decode(xmid, skips, time_context)
    loss = jnp.mean((pred_velocity[:, -1:] - target_velocity[:, -1:]) ** 2)
    if not return_eval_aux:
        reward_loss = jnp.array(0.0, dtype=loss.dtype)
        if rewards is not None and reward_loss_weight > 0.0:
            pred_reward = model.predict_reward(xmid)
            keep = (t >= reward_t_threshold).astype(loss.dtype)
            reward_loss = jnp.mean(keep * (pred_reward - rewards) ** 2)
        return loss + reward_loss_weight * reward_loss, reward_loss

    x1_pred = xt[:, -1:] + (1.0 - level[:, -1:]) * pred_velocity[:, -1:]
    recon_mse = jnp.mean((x1_pred - x1[:, -1:]) ** 2)
    target_var = jnp.mean(target_velocity[:, -1:] ** 2)
    r2 = 1.0 - loss / jnp.maximum(target_var, 1e-12)
    return loss, recon_mse, x1_pred, r2


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

    ts = jnp.linspace(0, 1, num_steps)
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
    intermediates = sample_euler(
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

        sample = sample_euler(
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


class FlowMatchingTrainer(nnx.Module):
    def __init__(
        self,
        model,
        ema_model,
        *,
        learning_rate: float | optax.Schedule = 3e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 2.0,
        ema_decay: float = 0.999,
        action_dropout: float = 0.1,
        reward_loss_weight: float = 0.0,
        reward_t_threshold: float = 0.6,
        min_context_t: float = 1.0,
        sampling_distribution: NoiseDistribution = "logitnorm",
        logit_norm_mu: float = 0.0,
        logit_norm_scale: float = 1.0,
        seed: int = 42,
    ):
        if reward_loss_weight > 0.0 and not getattr(model, "has_reward_head", False):
            raise ValueError(
                "reward_loss_weight > 0 requires a model built with predict_reward=True"
            )
        self.model = model
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.action_dropout = action_dropout
        self.reward_loss_weight = reward_loss_weight
        self.reward_t_threshold = reward_t_threshold
        self.min_context_t = min_context_t
        self.rng_key = nnx.Rngs(seed)

        # Plain floats (not a jnp array) so they can key the eval_step dict
        self.eval_timesteps = (0.01, 1.0 / 3.0, 2.0 / 3.0, 0.9)

        # Grad-clip + AdamW chain
        self.optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
            ),
            wrt=nnx.Param,
        )

        self.sampling_fn_for_t = functools.partial(
            sample_noise,
            noise_distribution=sampling_distribution,
            mu=logit_norm_mu,
            s=logit_norm_scale,
        )
        self.world_model_loss = functools.partial(
            _loss_at_t,
            reward_loss_weight=reward_loss_weight,
            reward_t_threshold=reward_t_threshold,
        )

    # todo look into jax.monitoring
    @nnx.jit
    def train_step(
        self,
        batch: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        if rewards is not None and rewards.ndim == 2:
            rewards = rewards[:, -1]  # reward of the final (denoised) frame

        # This is where the logit norm sampling may be used
        rng_key = self.rng_key.reparam()
        rng_key, key = random.split(rng_key)
        t = self.sampling_fn_for_t(key, batch.shape[:1])

        # Uniformyl select noise timepoint for context frames
        rng_key, key = random.split(rng_key)
        t_ctx = random.uniform(
            key=key, shape=batch.shape[:1], minval=self.min_context_t, maxval=1.0
        )

        # Random dropout for actions
        rng_key, key = random.split(rng_key)
        drop = random.uniform(key, shape=actions.shape) < self.action_dropout
        null = jnp.full(
            shape=actions.shape, fill_value=self.model.null_action, dtype=actions.dtype
        )
        actions = jnp.where(drop, null, actions)

        rng_key, key = random.split(rng_key)
        noise = random.normal(key, shape=batch.shape, dtype=batch.dtype)

        loss, grads = nnx.value_and_grad(
            lambda model: self.world_model_loss(
                model, batch, actions, t, noise=noise, t_ctx=t_ctx, rewards=rewards
            ),
            has_aux=True,
        )(self.model)

        self.optimizer.update(self.model, grads)
        ema_update(self.ema_model, self.model, self.ema_decay)
        return loss


    def eval_step(
        self,
        batch: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> dict[float, tuple]:
        """Per-timestep validation metrics: ``(losses, psnrs, r2s, preds)``.

        PSNR uses data range 2 since frames live in [-1, 1]. R² is
        ``1 - loss / E[target_velocity^2]`` — fraction of target variance
        explained. ``preds`` maps each timestep to the one-step x1 prediction
        ``(B, 1, H, W, C)`` for that batch — useful for image logging.
        """
        losses: dict[float, tuple] = {}

        log10_max = 20.0 * float(jnp.log10(jnp.array(2.0)))

        rng_key = self.rng_key.reparam()
        rng_key, key = random.split(rng_key)
        noise = random.normal(key, shape=batch.shape, dtype=batch.dtype)

        for timestep in self.eval_timesteps:
            t = jnp.full((batch.shape[0],), timestep, dtype=batch.dtype)
            loss, recon_mse, x1_pred, r2 = _loss_at_t(
                self.model, batch, actions, t, noise=noise, return_eval_aux=True
            )
            psnr = log10_max - 10.0 * jnp.log10(jnp.maximum(recon_mse, 1e-12))
            losses[timestep] = (loss, psnr, r2, x1_pred)

        return losses


def train_on_dataset(
    dataset: Dataset,
    num_env_actions: int = 1,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    sample_fps: float = 8.0,
    decode_fn: Callable | None = None,
):
    model_config = ModelConfig() if model_config is None else model_config
    train_config = TrainConfig() if train_config is None else train_config
    train_logger = None
    if train_config.log_tensorboard:
        save_path = Path(train_config.save_dir)
        train_logger = RLLogger(log_dir=str(save_path.parent), exp_name=save_path.name)

    decoder = decode_fn if decode_fn is not None else (lambda x: x)

    print("Using train config:")
    pprint(train_config)
    print("Using model config:")
    pprint(model_config)

    input_config = {
        "in_channels": dataset.num_channels,
        "out_channels": dataset.num_channels,
        "num_actions": num_env_actions,
    }
    full_model_config = {**input_config, **asdict(model_config)}

    if train_config.load_dir is not None:
        model = load_model(train_config.load_dir, prefer_ema=False)
        print(f"resuming training from: {train_config.load_dir}")
    else:
        model = UNet3D(**full_model_config, rngs=nnx.Rngs(0))

    ema_model = nnx.clone(model)

    lr_schedule: optax.Schedule = linear_warmup_decay_schedule(
        train_config.learning_rate,
        total_steps=train_config.train_steps,
        warmup_steps=train_config.lr_warmup_steps,
        hold_steps=train_config.lr_hold_steps,
        final_lr=train_config.lr_final,
    )

    if train_config.reward_loss_weight > 0.0 and not dataset.has_rewards:
        raise ValueError(
            "reward_loss_weight > 0 but the dataset has no rewards.npy — "
            "re-record the rollouts to capture rewards"
        )

    trainer = FlowMatchingTrainer(
        model,
        ema_model,
        learning_rate=lr_schedule,
        action_dropout=train_config.action_dropout,
        reward_loss_weight=train_config.reward_loss_weight,
        reward_t_threshold=train_config.reward_t_threshold,
        min_context_t=train_config.min_context_t,
    )

    checkpoint_interval = 1000
    if train_config.save_dir is not None:
        save_path = Path(train_config.save_dir) / "resume-ckpt"
        print(f"periodically saving checkpoints to: {save_path}")

    print("dataset clips:", dataset.dataset_size)
    print("train split:", dataset.train_size)
    print("val split:", dataset.val_size)
    # print(nnx.tabulate(model, x, t, a))

    sample_count = min(train_config.num_gen_samples, dataset.val_size)
    if train_config.save_dir is not None:
        val_conditioning_clips, val_conditioning_actions, _ = dataset.val_clips(
            sample_count
        )

    avg_loss = 0.0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    batch_size = train_config.batch_size

    for step in range(1, train_config.train_steps + 1):
        batch, batch_actions, batch_rewards = dataset.sample_train_batch(batch_size)
        loss, reward_loss = trainer.train_step(batch, batch_actions, batch_rewards)
        avg_loss += loss

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.train_steps
        ):
            val_batch, val_batch_actions, _ = dataset.sample_val_batch(
                min(train_config.batch_size, dataset.val_size)
            )

            val_metrics = trainer.eval_step(val_batch, val_batch_actions)

            window_steps = step - last_log_step
            avg_loss_f = float(avg_loss) / max(window_steps, 1)
            now = time.time()
            samples_per_sec = window_steps * batch_size / max(now - last_log_time, 1e-8)

            # Recall that jax.tree.map aligns corresponding leaves across the input PyTrees
            #  and passes them together to your function
            val_losses = {t: float(m[0]) for t, m in val_metrics.items()}
            val_psnrs = {t: float(m[1]) for t, m in val_metrics.items()}
            val_r2s = {t: float(m[2]) for t, m in val_metrics.items()}
            val_preds = {t: m[3] for t, m in val_metrics.items()}
            reduced_metrics = jax.tree.map(
                   # Mapped over leaf values e.g [loss_t1, loss_t2, loss_t3]
                   lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
                   # This is where each timepoint tuple is passed as a separate arg
                   # e.g. [loss_t1, psnr_t1,...], [loss_t2, psnr_t2, ...] ...
                   *((m[:3] for m in val_metrics.values())),
               )
            avg_val_loss, avg_val_psnr, avg_val_r2 = reduced_metrics
            val_report = f"val=[loss={float(avg_val_loss):.4f} psnr={float(avg_val_psnr):.2f} r2={float(avg_val_r2):.3f}]"
            reward_report = ""

            train_metrics = {
                "samples_per_second": samples_per_sec,
                "loss": float(loss),
                "avg_loss": avg_loss_f,
            }

            if train_config.reward_loss_weight > 0.0:
                reward_loss_f = float(reward_loss)
                reward_report = f" reward_loss={reward_loss_f:.4f}"
                train_metrics["reward_loss"] = reward_loss_f

            print(
                f"step={step:5d} sample/s={samples_per_sec:.2f} "
                f"loss={train_metrics['loss']:.4f} avg={avg_loss_f:.4f}"
                f"{reward_report} {val_report}"
            )

            if train_logger is not None:
                train_logger.log_train_metrics(step, train_metrics)
                train_logger.log_train_metrics(
                    step,
                    {
                        "avg_loss_t": avg_val_loss,
                        "avg_psnr": avg_val_psnr,
                        "avg_r2": avg_val_r2,
                    },
                    val=True,
                )
                train_logger.log_validation_steps(step, val_losses)
                train_logger.log_validation_psnrs(step, val_psnrs)
                train_logger.log_validation_r2s(step, val_r2s)
                _viz_samples = min(16, val_batch.shape[0])

                train_logger.log_reconstructions(
                    step,
                    decoder(val_batch[:_viz_samples, -1:])[:, 0],
                    {t: decoder(p[:_viz_samples])[:, 0] for t, p in val_preds.items()},
                )

            if train_config.save_dir is not None and step % checkpoint_interval == 0:
                save_model(
                    model,
                    save_path,
                    config={"step": step, **asdict(train_config), **full_model_config},
                )

            avg_loss = 0.0
            last_log_step = step
            last_log_time = time.time()

    # if train_config.save_dir is not None:
    #     samples = sample_euler(
    #         trainer.model,
    #         conditioning_clips=val_conditioning_clips[:, :-1],
    #         actions=val_conditioning_actions,
    #         num_steps=train_config.sample_steps,
    #     )

    #     save_clip_previews(
    #         decoder(samples),
    #         train_config.save_dir,
    #         max_clips=sample_count,
    #         fps=sample_fps,
    #     )
    #     print(f"saved samples to: {train_config.save_dir}")

    return trainer.model, trainer.ema_model, full_model_config

def train_overfit_random(
    *,
    steps: int = 1_000,
    batch_size: int = 1,
    base_channels: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 5,
    num_samples: int = 4,
    sample_steps: int = 32,
    sample_fps: float = 8.0,
    # The U-Net halves the time axis twice, so clips must be >= 4 frames;
    # the vae dirs hold 2-frame VAE pairs and won't fit the world model.
    data_dir: str | Path = "logs/port-utils/vizdoom-vae/test-set",
):
    dataset = Dataset(data_dir=data_dir, memory_map=True)
    return train_on_dataset(
        dataset,
        num_env_actions=dataset.num_actions,
        model_config=ModelConfig(base_channels=base_channels),
        train_config=TrainConfig(
            train_steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_every=log_every,
            num_gen_samples=num_samples,
            sample_steps=sample_steps,
            save_dir="logs/port-utils/vizdoom-diffusion/overfit/",
            log_tensorboard=True,
        ),
    )

if __name__ == "__main__":
    train_overfit_random()
