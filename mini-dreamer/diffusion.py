from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pp, pprint
from typing import Callable, Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from data import _split_size, load_rollouts, sample_batch
from logger_utils import RLLogger
from unet import UNet3D, print_param_table
from video_utils import (
    load_video_dataset,
    make_random_video_dataset,
    save_clip_previews,
    save_diffusion_mp4,
)

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

        print(f"setup dataset from {data_dir}")
        print(f"{self.train_size = } - {self.val_size = }")

    def _build_tensor(
        self,
        videos: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> tuple[mx.array, mx.array, mx.array]:
        video_batch = mx.array(videos)
        action_batch = mx.array(actions)
        reward_batch = mx.array(rewards)

        if self.encoder is not None:
            video_batch = self.encoder(video_batch)
        return video_batch, action_batch, reward_batch

    def sample_train_batch(
        self, batch_size: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        videos, actions, rewards = sample_batch(
            self.train_videos, self.train_actions, batch_size, self.train_rewards
        )
        return self._build_tensor(videos, actions, rewards)

    def sample_val_batch(self, batch_size: int) -> tuple[mx.array, mx.array, mx.array]:
        videos, actions, rewards = sample_batch(
            self.val_videos, self.val_actions, batch_size, self.val_rewards
        )
        return self._build_tensor(videos, actions, rewards)

    def val_clips(self, num_clips: int) -> tuple[mx.array, mx.array, mx.array]:
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


def make_final_frame_mask(x: mx.array) -> mx.array:
    frame_mask = (mx.arange(x.shape[1]) == (x.shape[1] - 1)).astype(x.dtype)
    return mx.reshape(frame_mask, (1, x.shape[1], 1, 1, 1))


def clone_model(model_config_dict, original_parameters) -> UNet3D:
    clone = UNet3D(**model_config_dict)
    clone.update(tree_map(lambda x: x * 1.0, original_parameters))
    return clone


def ema_update(ema_model: UNet3D, model: UNet3D, decay: float) -> None:
    ema_params = ema_model.parameters()
    model_params = model.parameters()
    ema_model.update(
        tree_map(
            lambda ema, current: decay * ema + (1.0 - decay) * current,
            ema_params,
            model_params,
        )
    )


def sample_t_logit_normal(
    shape: tuple[int, ...], mu: float = 0.0, s: float = 1.0, eps: float = 1e-6
):
    # logit(t) ~ N(mu, s^2)  =>  t = sigmoid(N(...))
    z = mu + s * mx.random.normal(shape)
    t = mx.sigmoid(z)  # t in (0, 1)
    return mx.clip(t, eps, 1.0 - eps)  # avoid exact 0/1


def sample_noise(
    shape: tuple[int, ...],
    noise_distribution: NoiseDistribution,
    mu: float = 0.0,
    s: float = 1.0,
    dtype: mx.Dtype = mx.float32,
):
    match noise_distribution:
        case "logitnorm":
            return sample_t_logit_normal(shape, mu, s)

        case "normal":
            return mx.random.normal(shape=shape, dtype=dtype)

        case "uniform":
            return mx.random.uniform(shape=shape, dtype=dtype)

        case _:
            raise ValueError(f"Unknown noise distribution: {noise_distribution}")


def linear_warmup_decay_schedule(
    peak_lr: float,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    hold_steps: int = 0,
    final_lr: float | None = None,
) -> float | Callable[[mx.array], mx.array]:
    """Linear warmup from ``peak_lr / 10`` to ``peak_lr`` over ``warmup_steps``,
    hold at ``peak_lr`` for ``hold_steps``, then linear decay to ``final_lr``
    over the remaining steps.

    Every part is optional: ``warmup_steps=0`` skips the warmup, ``hold_steps=0``
    starts the decay right after warmup, and ``final_lr=None`` holds the LR
    constant after warmup. With everything disabled this returns the plain
    ``peak_lr`` float (a fixed learning rate).
    """
    if warmup_steps <= 0 and final_lr is None:
        return peak_lr

    constant = optim.linear_schedule(peak_lr, peak_lr, 1)
    tail = (
        optim.linear_schedule(
            peak_lr, final_lr, max(total_steps - warmup_steps - hold_steps, 1)
        )
        if final_lr is not None
        else constant
    )

    schedules = [tail]
    boundaries: list[int] = []
    if hold_steps > 0 and final_lr is not None:
        schedules.insert(0, constant)
        boundaries.insert(0, warmup_steps + hold_steps)
    if warmup_steps > 0:
        schedules.insert(
            0, optim.linear_schedule(peak_lr / 10.0, peak_lr, warmup_steps)
        )
        boundaries.insert(0, warmup_steps)

    if not boundaries:
        return tail
    return optim.join_schedules(schedules, boundaries)


class FlowMatchingTrainer:
    def __init__(
        self,
        model,
        ema_model,
        *,
        learning_rate: float | Callable[[mx.array], mx.array] = 3e-4,
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
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.max_grad_norm = max_grad_norm
        # self.loss_and_grad = nn.value_and_grad(self.model, self.loss)
        self.sampling_fn_for_t = lambda x: sample_noise(
            x, sampling_distribution, logit_norm_mu, logit_norm_scale
        )
        self.logit_norm_mu = logit_norm_mu
        self.logit_norm_scale = logit_norm_scale

        # State captured as input and output of the compiled step. Must list
        # every piece of mutable state the step reads or writes, or compile
        # freezes it to its first-traced value: model + optimizer (updated by
        # the step), the EMA shadow weights (updated in-step so their graph
        # stays evaluated rather than accumulating across steps), and the RNG
        # state (otherwise `t`/noise are identical every step).
        train_state = [
            self.model.state,
            self.optimizer.state,
            self.ema_model.state,
            mx.random.state,
        ]
        self.compiled_train_step = mx.compile(
            lambda batch, actions, rewards: self.train_step(batch, actions, rewards),
            inputs=train_state,
            outputs=train_state,
        )
        # Eval only reads the online model and advances the RNG (for the noise
        # draw); it must not capture optimizer/EMA state.
        # eval_state = [self.model.state, mx.random.state]
        # self.eval_loss_by_timestep = mx.compile(
        #     lambda batch, actions, timesteps: self._eval_loss_by_timestep(
        #         batch, actions, timesteps
        #     ),
        #     inputs=eval_state,
        #     outputs=eval_state,
        # )
        self.eval_loss_by_timestep = self._eval_loss_by_timestep

    def _loss_at_t(
        self,
        model: UNet3D,
        x1: mx.array,
        actions: mx.array,
        t: mx.array,
        *,
        t_ctx: mx.array | None = None,
        rewards: mx.array | None = None,
        return_eval_aux: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array, mx.array]:
        """Flow-matching loss at the given t.

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
            t_ctx = mx.ones_like(t)

        # Diffusion-forcing style corruption: the target (final) frame is noised
        # to level ``t`` while the conditioning frames are noised to level
        # ``t_ctx`` (1.0 = clean). Training on imperfect history teaches the
        # model to denoise from its own (error-carrying) rollout frames instead
        # of always-clean ground truth, which is what curbs autoregressive
        # drift. ``level`` is ``t`` on the final frame and ``t_ctx`` elsewhere.
        noise = sample_noise(x1.shape, noise_distribution="normal")
        t_view = mx.reshape(t, (x1.shape[0], 1, 1, 1, 1)) * mask
        t_ctx_view = mx.reshape(t_ctx, (x1.shape[0], 1, 1, 1, 1)) * (1 - mask)
        level = t_view + t_ctx_view
        xt = (1.0 - level) * noise + level * x1
        target_velocity = mask * (x1 - noise)
        xmid, skips, time_context = model.encode(xt, t, context=actions, t_ctx=t_ctx)
        pred_velocity = model.decode(xmid, skips, time_context)
        loss = mx.mean((pred_velocity[:, -1:] - target_velocity[:, -1:]) ** 2)
        if not return_eval_aux:
            reward_loss = mx.array(0.0)
            if rewards is not None and self.reward_loss_weight > 0.0:
                pred_reward = model.predict_reward(xmid)
                keep = (t >= self.reward_t_threshold).astype(loss.dtype)
                reward_loss = mx.mean(keep * (pred_reward - rewards) ** 2)
            return loss + self.reward_loss_weight * reward_loss, reward_loss

        x1_pred = xt[:, -1:] + (1.0 - level[:, -1:]) * pred_velocity[:, -1:]
        recon_mse = mx.mean((x1_pred - x1[:, -1:]) ** 2)
        target_var = mx.mean(target_velocity[:, -1:] ** 2)
        r2 = 1.0 - loss / mx.maximum(target_var, 1e-12)
        return loss, recon_mse, x1_pred, r2

    def _eval_loss_by_timestep(
        self,
        batch: mx.array,
        actions: mx.array,
        timesteps: tuple[float, ...],
    ) -> tuple[
        dict[float, float],
        dict[float, float],
        dict[float, float],
        dict[float, mx.array],
    ]:
        """Per-timestep validation metrics: ``(losses, psnrs, r2s, preds)``.

        PSNR uses data range 2 since frames live in [-1, 1]. R² is
        ``1 - loss / E[target_velocity^2]`` — fraction of target variance
        explained. ``preds`` maps each timestep to the one-step x1 prediction
        ``(B, 1, H, W, C)`` for that batch — useful for image logging.
        """
        losses: dict[float, float] = {}
        psnrs: dict[float, float] = {}
        r2s: dict[float, float] = {}
        preds: dict[float, mx.array] = {}
        log10_max = 20.0 * float(mx.log10(mx.array(2.0)))
        for timestep in timesteps[::-1]:
            t = mx.full((batch.shape[0],), timestep, dtype=batch.dtype)
            loss, recon_mse, x1_pred, r2 = self._loss_at_t(
                self.model, batch, actions, t, return_eval_aux=True
            )
            psnr = log10_max - 10.0 * mx.log10(mx.maximum(recon_mse, 1e-12))
            # mx.eval(loss, psnr, r2, x1_pred)
            losses[timestep] = loss
            psnrs[timestep] = psnr
            r2s[timestep] = r2
            preds[timestep] = x1_pred
        return losses, psnrs, r2s, preds

    def _sample_context_t(self, shape: tuple[int, ...]) -> mx.array:
        """Sample the conditioning-frame noise level, uniform in
        ``[min_context_t, 1.0]`` (1.0 = clean). ``min_context_t == 1.0``
        disables context-noise augmentation (always-clean history)."""
        lo = self.min_context_t
        return lo + (1.0 - lo) * mx.random.uniform(shape=shape)

    def _dropout_actions(self, actions: mx.array) -> mx.array:
        """Replace each action slot with the NULL action independently with
        probability ``action_dropout``. Full-NULL contexts train the
        unconditional path (CFG); partial-NULL contexts make "real prefix +
        NULL last action" in-distribution for policy/value embedding
        extraction."""
        if self.action_dropout <= 0.0:
            return actions
        drop = mx.random.uniform(shape=actions.shape) < self.action_dropout
        null = mx.array(self.model.null_action, dtype=actions.dtype)
        return mx.where(drop, null, actions)

    def train_step(
        self, batch: mx.array, actions: mx.array, rewards: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        if rewards is not None and rewards.ndim == 2:
            rewards = rewards[:, -1]  # reward of the final (denoised) frame

        def _loss(batch, actions, rewards):
            # This is where the logit norm sampling may be used
            t = self.sampling_fn_for_t(batch.shape[:1])
            t_ctx = self._sample_context_t(batch.shape[:1])
            actions = self._dropout_actions(actions)
            return self._loss_at_t(
                self.model, batch, actions, t, t_ctx=t_ctx, rewards=rewards
            )

        loss_and_grad_fn = nn.value_and_grad(self.model, _loss)
        (loss, reward_loss), grads = loss_and_grad_fn(batch, actions, rewards)

        clipped_grads, total_norm = optim.clip_grad_norm(
            grads, max_norm=self.max_grad_norm
        )
        self.optimizer.update(self.model, clipped_grads)
        ema_update(self.ema_model, self.model, self.ema_decay)
        return loss, reward_loss


def sample_euler(
    model: UNet3D,
    *,
    conditioning_clips: mx.array,
    actions: mx.array,
    num_steps: int = 32,
    return_intermediates: bool = False,
) -> mx.array | list[mx.array]:
    """Run Euler integration to generate the next frame.

    Args:
        return_intermediates: if True, return a list of ``(B, 1, H, W, C)``
            arrays — one snapshot of the noisy frame ``x`` after each Euler
            step — instead of the final concatenated clip.  Useful for
            visualising the denoising trajectory.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    batch_size, history_size = conditioning_clips.shape[0], conditioning_clips.shape[1]
    assert actions.shape == (batch_size, history_size + 1), (
        f"Actions should include history and the next action"
    )
    x_shape = (
        batch_size,
        1,
        int(conditioning_clips.shape[2]),
        int(conditioning_clips.shape[3]),
        int(conditioning_clips.shape[4]),
    )
    x = mx.random.normal(x_shape, dtype=conditioning_clips.dtype)
    dt = 1.0 / num_steps
    intermediates: list[mx.array] = []
    for step in range(num_steps):
        t = mx.full((batch_size,), step / num_steps, dtype=conditioning_clips.dtype)
        xt = mx.concatenate([conditioning_clips, x], axis=1)
        v = model(xt, t, context=actions)
        x = x + dt * v[:, -1:]
        if return_intermediates:
            mx.eval(x)
            intermediates.append(x)

    if return_intermediates:
        return intermediates

    samples = mx.concatenate([conditioning_clips, x], axis=1)
    mx.eval(samples)
    return samples


def sample_euler_to_mp4(
    model: UNet3D,
    *,
    conditioning_clips: mx.array,
    actions: mx.array,
    output_path: str | Path,
    num_steps: int = 32,
    fps: float = 8.0,
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
        conditioning_clips=conditioning_clips,
        actions=actions,
        num_steps=num_steps,
        return_intermediates=True,
    )
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
    model.save_weights(str(save_dir / "model.safetensors"))
    if ema_model is not None:
        ema_model.save_weights(str(save_dir / "ema_model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_model(save_dir: str | Path, *, prefer_ema: bool = True) -> UNet3D:
    """Load a `UNet3D` previously saved via `save_model`.

    If `prefer_ema` is True and `ema_model.safetensors` exists, EMA weights are
    loaded. Otherwise `model.safetensors` is loaded.
    """
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    model = UNet3D(**config)

    model_path = save_dir / "model.safetensors"
    ema_path = save_dir / "ema_model.safetensors"
    weights_path = ema_path if prefer_ema and ema_path.exists() else model_path
    model.load_weights(str(weights_path), strict=False)
    return model


def generate_video(
    model: UNet3D,
    *,
    initial_clip: mx.array,
    num_new_frames: int,
    actions: mx.array,
    num_steps: int = 32,
) -> mx.array:
    """Autoregressively extend `initial_clip` by `num_new_frames` new frames.

    Args:
        initial_clip: (B, L, H, W, C) seed frames.
        num_new_frames: number of frames to generate after the seed.
        actions: optional (B, L + num_new_frames) action stream. Each Euler
            call receives an (L - 1)-frame conditioning window and an L-action
            window aligned to the generated clip.
        num_steps: Euler integration steps per generated frame.

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
    print(f"Using actions={actions}")
    frames = initial_clip
    max_context_size = model.max_context_size
    for step in range(num_new_frames):
        window = frames[:, -max_context_size:]
        print(
            f"generating frame {step + 1}/{num_new_frames} with conditioning window shape {window.shape}..."
        )
        end = frames.shape[1] + 1  # frame index being generated, + 1 inclusive
        action_window = actions[:, max(0, end - max_context_size - 1) : end]
        # print(f"Using action_window={action_window}")

        sample = sample_euler(
            model,
            conditioning_clips=window,
            actions=action_window,
            num_steps=num_steps,
        )
        frames = mx.concatenate([frames, sample[:, -1:]], axis=1)

    mx.eval(frames)
    return frames


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
    decode_fn: Callable | None = None,
) -> mx.array:
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
    full_actions = mx.array(
        np.concatenate([initial_actions_np, extra_actions_np], axis=1)
    )
    print(
        f"num_new_frames in genenv={num_new_frames}, initial_clip={initial_clip.shape}"
    )
    generated = generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=full_actions,
        num_steps=num_steps,
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
        decode_fn=decode_fn,
    )
    print(f"saved generated video to: {save_dir}")
    return generated


def decoder(x: mx.array) -> mx.array:
    return x


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

    if decode_fn is not None:
        decoder = decode_fn

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
        model = UNet3D(**full_model_config)

    ema_model = clone_model(full_model_config, model.parameters())

    lr_schedule = linear_warmup_decay_schedule(
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
    save_path = Path(train_config.save_dir) / "resume-ckpt"
    print(f"periodically saving checkpoints to: {save_path}")

    print("dataset clips:", dataset.dataset_size)
    print("train split:", dataset.train_size)
    print("val split:", dataset.val_size)
    print_param_table(model)

    sample_count = min(train_config.num_gen_samples, dataset.val_size)
    if train_config.save_dir is not None:
        val_conditioning_clips, val_conditioning_actions, _ = dataset.val_clips(
            sample_count
        )

    val_timesteps = (0.0, 1.0 / 3.0, 2.0 / 3.0, 0.9)
    avg_loss = 0.0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    batch_size = train_config.batch_size

    for step in range(1, train_config.train_steps + 1):
        batch, batch_actions, batch_rewards = dataset.sample_train_batch(batch_size)
        loss, reward_loss = trainer.compiled_train_step(
            batch, batch_actions, batch_rewards
        )
        avg_loss += loss
        mx.async_eval(loss, avg_loss, reward_loss)

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.train_steps
        ):
            val_batch, val_batch_actions, _ = dataset.sample_val_batch(
                min(train_config.batch_size, dataset.val_size)
            )

            val_losses, val_psnrs, val_r2s, val_preds = trainer.eval_loss_by_timestep(
                val_batch, val_batch_actions, val_timesteps
            )
            mx.async_eval(*val_losses.values(), *val_psnrs.values(), *val_r2s.values())

            window_steps = step - last_log_step
            loss_f = float(loss)
            avg_loss_f = float(avg_loss) / window_steps
            val_losses = {t: float(v) for t, v in val_losses.items()}
            val_psnrs = {t: float(v) for t, v in val_psnrs.items()}
            val_r2s = {t: float(v) for t, v in val_r2s.items()}
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            avg_val_psnr = sum(val_psnrs.values()) / len(val_psnrs)
            avg_val_r2 = sum(val_r2s.values()) / len(val_r2s)
            val_report = (
                f"avg_val_t={avg_val_loss:.4f} avg_val_psnr={avg_val_psnr:.2f}dB "
                f"avg_val_r2={avg_val_r2:.3f}"
            )

            now = time.time()
            samples = window_steps * batch_size
            samples_per_sec = samples / max(now - last_log_time, 1e-8)

            reward_report = ""
            train_metrics = {
                "samples_per_second": samples_per_sec,
                "loss": loss_f,
                "avg_loss": avg_loss_f,
            }
            if train_config.reward_loss_weight > 0.0:
                reward_loss_f = float(reward_loss)
                reward_report = f" reward_loss={reward_loss_f:.4f}"
                train_metrics["reward_loss"] = reward_loss_f

            print(
                f"step={step:5d} sample/s={samples_per_sec:.2f} "
                f"loss={loss_f:.4f} avg={avg_loss_f:.4f}{reward_report} {val_report}"
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

            if step % checkpoint_interval == 0:
                save_model(
                    model,
                    save_path,
                    config={"step": step, **asdict(train_config), **full_model_config},
                )

            avg_loss = 0.0
            mx.clear_cache()
            last_log_step = step
            last_log_time = time.time()

    if train_config.save_dir is not None:
        samples = sample_euler(
            trainer.model,
            conditioning_clips=val_conditioning_clips[:, :-1],
            actions=val_conditioning_actions,
            num_steps=train_config.sample_steps,
        )

        save_clip_previews(
            decoder(samples),
            train_config.save_dir,
            max_clips=sample_count,
            fps=sample_fps,
        )
        print(f"saved samples to: {train_config.save_dir}")

    return trainer.model, trainer.ema_model, full_model_config


def train_overfit_random_noise(
    *,
    steps: int = 1_000,
    batch_size: int = 1,
    num_videos: int = 1,
    frames: int = 4,
    height: int = 32,
    width: int = 32,
    channels: int = 3,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 50,
    sample_dir: str | Path | None = None,
    num_samples: int = 4,
    sample_steps: int = 32,
    sample_fps: float = 8.0,
):
    videos = make_random_video_dataset(
        num_videos=num_videos,
        frames=frames,
        height=height,
        width=width,
        channels=channels,
        seed=seed,
    )
    return train_on_dataset(
        videos,
        model_config=ModelConfig(base_channels=base_channels),
        train_config=TrainConfig(
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_every=log_every,
            sample_dir=sample_dir,
            num_gen_samples=num_samples,
            sample_steps=sample_steps,
            sample_fps=sample_fps,
        ),
    )


def train_on_video(
    video_path: str | Path,
    *,
    steps: int = 1_000,
    batch_size: int = 1,
    frames: int = 4,
    target_fps: float = 8.0,
    spatial_downsample: int = 2,
    clip_stride: int | None = None,
    max_clips: int | None = None,
    preview_dir: str | Path | None = None,
    preview_clips: int = 4,
    preview_only: bool = False,
    sample_dir: str | Path | None = None,
    num_samples: int = 4,
    sample_steps: int = 32,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    log_every: int = 50,
):
    videos, info = load_video_dataset(
        video_path,
        clip_length=frames,
        target_fps=target_fps,
        spatial_downsample=spatial_downsample,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    print(f"video: {video_path}")
    print(
        "video info:",
        {
            "source_fps": round(float(info["source_fps"]), 3),
            "actual_fps": round(float(info["actual_fps"]), 3),
            "frame_step": int(info["frame_step"]),
            "source_size": info["source_size"],
            "processed_size": info["processed_size"],
            "spatial_downsample": int(info["spatial_downsample"]),
            "num_frames": int(info["num_frames"]),
            "num_clips": int(info["num_clips"]),
            "clip_length": int(info["clip_length"]),
        },
    )

    if preview_dir is not None:
        save_clip_previews(
            videos,
            preview_dir,
            max_clips=preview_clips,
            fps=float(info["actual_fps"]),
        )
        print(f"saved previews to: {preview_dir}")

    if preview_only:
        return None, []

    return train_on_dataset(
        videos,
        model_config=ModelConfig(base_channels=base_channels),
        train_config=TrainConfig(
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_every=log_every,
            sample_dir=sample_dir,
            num_gen_samples=num_samples,
            sample_steps=sample_steps,
            sample_fps=float(info["actual_fps"]),
        ),
    )


def generate_from_pretrained(
    *,
    load_dir: str | Path,
    sample_dir: str | Path,
    initial_clip: mx.array,
    num_new_frames: int,
    num_steps: int = 32,
    sample_fps: float = 8.0,
    seed: int = 0,
) -> mx.array:
    """Load a saved model and write a generated video preview to `sample_dir`.

    Uses zero-action context for the generated frames.
    """
    model = load_model(load_dir)
    print(f"loaded model from: {load_dir}")
    print_param_table(model)

    clip_length = int(initial_clip.shape[1])
    batch_size = int(initial_clip.shape[0])
    actions = mx.zeros((batch_size, clip_length + num_new_frames), dtype=mx.int32)

    generated = generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=actions,
        num_steps=num_steps,
    )
    save_clip_previews(generated, sample_dir, max_clips=batch_size, fps=sample_fps)
    print(f"saved generated video to: {sample_dir}")
    return generated


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal flow-matching trainer for UNet3D on random-noise or MP4 videos."
    )
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--video-target-fps", type=float, default=8.0)
    parser.add_argument("--video-downsample", type=int, default=2)
    parser.add_argument(
        "--clip-stride",
        type=int,
        default=None,
        help="Stride between clips; defaults to 1 for a rolling window.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--preview-dir", type=str, default=None)
    parser.add_argument("--preview-clips", type=int, default=4)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--sample-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--full-resolution", action="store_true")
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help="If set, load a pretrained model from this directory and run generation "
        "(skips training).",
    )
    parser.add_argument(
        "--generate-new-frames",
        type=int,
        default=32,
        help="Number of frames to autoregressively generate after the initial clip.",
    )
    parser.add_argument(
        "--generate-num-steps",
        type=int,
        default=32,
        help="Euler integration steps per generated frame.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.load_dir is not None:
        if args.video is not None:
            videos, info = load_video_dataset(
                args.video,
                clip_length=args.frames,
                target_fps=args.video_target_fps,
                spatial_downsample=1 if args.full_resolution else args.video_downsample,
                clip_stride=args.clip_stride,
                max_clips=args.max_clips,
            )
            initial_clip = videos[: args.num_samples]
            generation_fps = float(info["actual_fps"])
        else:
            initial_clip = make_random_video_dataset(
                num_videos=args.num_samples,
                frames=args.frames,
                height=args.height,
                width=args.width,
                channels=args.channels,
                seed=args.seed,
            )
            generation_fps = args.sample_fps

        sample_dir = args.sample_dir or str(Path(args.load_dir) / "generated")
        generate_from_pretrained(
            load_dir=args.load_dir,
            sample_dir=sample_dir,
            initial_clip=initial_clip,
            num_new_frames=args.generate_new_frames,
            num_steps=args.generate_num_steps,
            sample_fps=generation_fps,
            seed=args.seed,
        )
        return

    if args.video is not None:
        train_on_video(
            args.video,
            steps=args.steps,
            batch_size=args.batch_size,
            frames=args.frames,
            target_fps=args.video_target_fps,
            spatial_downsample=1 if args.full_resolution else args.video_downsample,
            clip_stride=args.clip_stride,
            max_clips=args.max_clips,
            preview_dir=args.preview_dir,
            preview_clips=args.preview_clips,
            preview_only=args.preview_only,
            sample_dir=args.sample_dir,
            num_samples=args.num_samples,
            sample_steps=args.sample_steps,
            base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            log_every=args.log_every,
        )
        return

    train_overfit_random_noise(
        steps=args.steps,
        batch_size=args.batch_size,
        num_videos=1,
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
        sample_dir=args.sample_dir,
        num_samples=args.num_samples,
        sample_steps=args.sample_steps,
        sample_fps=args.sample_fps,
    )


if __name__ == "__main__":
    main()
