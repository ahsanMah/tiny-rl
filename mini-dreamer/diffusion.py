from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pp, pprint
from typing import Literal, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from logger_utils import RLLogger
from unet import UNet3D, print_param_table
from video_utils import (
    load_video_dataset,
    make_random_video_dataset,
    save_clip_previews,
    save_diffusion_mp4,
)

NoiseDistribution = Literal["uniform", "logitnorm"]


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 16
    max_context_size: int = 3
    num_transformer_blocks: int = 2


@dataclass(frozen=True)
class TrainConfig:
    train_steps: int = 1_000
    batch_size: int = 1
    learning_rate: float = 1e-3
    ema_decay: float = 0.999
    log_every: int = 50
    save_dir: str | Path | None = None
    load_dir: str | None = None
    num_gen_samples: int = 4
    sample_steps: int = 32
    preview_fps: float = 8.0
    sampling_distribution: str = "uniform"
    log_tensorboard: bool = False


def make_final_frame_mask(x: mx.array) -> mx.array:
    frame_mask = (mx.arange(x.shape[1]) == (x.shape[1] - 1)).astype(x.dtype)
    return mx.reshape(frame_mask, (1, x.shape[1], 1, 1, 1))


def infer_model_config(model: UNet3D) -> dict[str, int]:
    return {
        "in_channels": int(model.res1.shortcut.weight.shape[-1]),
        "out_channels": int(model.out_conv.weight.shape[0]),
        "base_channels": int(model.out_conv.weight.shape[-1]),
        "num_actions": int(model.context_embed.embed.weight.shape[0]),
        "max_context_size": int(model.max_context_size),
    }


def clone_model(model: UNet3D) -> UNet3D:
    clone = UNet3D(**infer_model_config(model))
    clone.update(tree_map(lambda x: x * 1.0, model.parameters()))
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
    x: mx.array,
    noise_distribution: NoiseDistribution,
    mu: float = 0.0,
    s: float = 1.0,
):
    if noise_distribution == "logitnorm":
        return sample_t_logit_normal(x.shape, mu, s).astype(x.dtype)

    return mx.random.normal(x.shape, dtype=x.dtype)


class FlowMatchingTrainer:
    def __init__(
        self,
        model: UNet3D,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 2.0,
        ema_decay: float = 0.999,
        sampling_distribution: NoiseDistribution = "logitnorm",
        logit_norm_mu: float = 0.0,
        logit_norm_scale: float = 1.0,
    ):
        self.model = model
        self.ema_model = clone_model(model)
        self.ema_decay = ema_decay
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.max_grad_norm = max_grad_norm
        self.loss_and_grad = nn.value_and_grad(model, self.loss)

        self.sampling_fn = lambda x: sample_noise(
            x, sampling_distribution, logit_norm_mu, logit_norm_scale
        )
        self.logit_norm_mu = logit_norm_mu
        self.logit_norm_scale = logit_norm_scale

    def _loss_at_t(
        self,
        model: UNet3D,
        x1: mx.array,
        actions: mx.array,
        t: mx.array,
        *,
        return_recon_mse: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Flow-matching loss at the given t. If ``return_recon_mse`` is True,
        also returns the one-step x1-reconstruction MSE on the target frame
        (``x1_pred = xt + (1 - t) * v_pred``). Skipped by default to keep
        training fast.
        """
        mask = make_final_frame_mask(x1)
        noise = self.sampling_fn(x1) * mask
        t_view = mx.reshape(t, (x1.shape[0], 1, 1, 1, 1)) * mask
        xt = (1.0 - t_view) * noise + t_view * x1 + (1 - mask) * x1
        target_velocity = mask * (x1 - noise)
        pred_velocity = model(xt, t, context=actions)
        loss = mx.mean((pred_velocity[:, -1:] - target_velocity[:, -1:]) ** 2)
        if not return_recon_mse:
            return loss

        x1_pred = xt[:, -1:] + (1.0 - t_view[:, -1:]) * pred_velocity[:, -1:]
        recon_mse = mx.mean((x1_pred - x1[:, -1:]) ** 2)
        return loss, recon_mse

    def loss(self, model: UNet3D, x1: mx.array, actions: mx.array) -> mx.array:
        t = mx.random.uniform(shape=(x1.shape[0],), low=0.0, high=1.0)
        return self._loss_at_t(model, x1, actions, t)

    def eval_loss_by_timestep(
        self,
        batch: mx.array,
        actions: mx.array,
        timesteps: tuple[float, ...],
    ) -> tuple[dict[float, float], dict[float, float]]:
        """Per-timestep validation metrics: ``(losses, psnrs)``.

        PSNR uses data range 2 since frames live in [-1, 1].
        """
        losses: dict[float, float] = {}
        psnrs: dict[float, float] = {}
        log10_max = 20.0 * float(mx.log10(mx.array(2.0)))
        for timestep in timesteps:
            t = mx.full((batch.shape[0],), timestep, dtype=batch.dtype)
            loss, recon_mse = self._loss_at_t(
                self.model, batch, actions, t, return_recon_mse=True
            )
            psnr = log10_max - 10.0 * mx.log10(mx.maximum(recon_mse, 1e-12))
            mx.eval(loss, psnr)
            losses[timestep] = float(loss)
            psnrs[timestep] = float(psnr)
        return losses, psnrs

    def train_step(self, batch: mx.array, actions: mx.array) -> float:
        loss, grads = self.loss_and_grad(self.model, batch, actions)

        clipped_grads, total_norm = optim.clip_grad_norm(
            grads, max_norm=self.max_grad_norm
        )
        self.optimizer.update(self.model, clipped_grads)
        ema_update(self.ema_model, self.model, self.ema_decay)

        mx.eval(
            loss,
            self.model.parameters(),
            self.ema_model.parameters(),
            self.optimizer.state,
        )
        return float(loss)


def sample_batch(
    videos: mx.array,
    actions: mx.array,
    batch_size: int,
) -> tuple[mx.array, mx.array]:
    if batch_size >= videos.shape[0]:
        return videos[:batch_size], actions[:batch_size]

    indices = mx.random.randint(0, videos.shape[0], shape=(batch_size,))
    return (videos[indices], actions[indices])


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
) -> None:
    """Generate one new frame and save the full denoising trajectory as an MP4.

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
    save_diffusion_mp4(conditioning_clips, intermediates, output_path, fps=fps)


def save_model(model: UNet3D, save_dir: str | Path, *, config: dict) -> None:
    """Save model weights (`model.safetensors`) and constructor config (`config.json`).

    `config` must contain exactly the kwargs needed to re-instantiate `UNet3D`.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(save_dir / "model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_model(save_dir: str | Path) -> UNet3D:
    """Load a `UNet3D` previously saved via `save_model`."""
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    model = UNet3D(**config)
    model.load_weights(str(save_dir / "model.safetensors"))
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
    if int(actions.shape[1]) != expected:
        raise ValueError(
            f"actions must have shape (B, {expected}), got {tuple(actions.shape)}"
        )

    if clip_length < 2:
        raise ValueError(
            f"initial_clip must contain at least 2 frames, got {clip_length}"
        )
    print(f"Generating {num_new_frames} frames from {clip_length} initial frames...")
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
        print(f"Using action_window={action_window}")

        sample = sample_euler(
            model,
            conditioning_clips=window,
            actions=action_window,
            num_steps=num_steps,
        )
        frames = mx.concatenate([frames, sample[:, -1:]], axis=1)

    mx.eval(frames)
    return frames


def train_on_dataset(
    videos: mx.array,
    actions: mx.array | None = None,
    num_env_actions: int = 1,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    model: UNet3D | None = None,
):
    model_config = ModelConfig() if model_config is None else model_config
    train_config = TrainConfig() if train_config is None else train_config
    train_logger = None
    if train_config.log_tensorboard:
        save_path = Path(train_config.save_dir)
        train_logger = RLLogger(log_dir=str(save_path.parent), exp_name=save_path.name)

    print("Using train config:")
    pprint(train_config)
    print("Using model config:")
    pprint(model_config)

    if actions is None:
        actions = mx.zeros((videos.shape[0], 1), dtype=mx.int8)

    dataset_size = int(videos.shape[0])
    if dataset_size < 2:
        raise ValueError(
            f"Need at least 2 clips to make a train/val split, got {dataset_size}"
        )

    val_size = max(1, int(round(dataset_size * 0.05)))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size

    train_videos = videos[:train_size]
    train_actions = actions[:train_size]
    val_videos = videos[train_size:]
    val_actions = actions[train_size:]

    if model is None:
        model = UNet3D(
            in_channels=int(videos.shape[-1]),
            out_channels=int(videos.shape[-1]),
            num_actions=num_env_actions,
            **asdict(model_config),
        )
    trainer = FlowMatchingTrainer(model, learning_rate=train_config.learning_rate)

    print("dataset:", tuple(videos.shape))
    print("train split:", tuple(train_videos.shape))
    print("val split:", tuple(val_videos.shape))
    print_param_table(model)

    sample_count = min(train_config.num_gen_samples, int(val_videos.shape[0]))
    if train_config.save_dir is not None:
        val_conditioning_clips = val_videos[:sample_count]
        val_conditioning_actions = val_actions[:sample_count]

    start = time.time()
    losses: list[float] = []
    val_timesteps = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

    for step in range(1, train_config.train_steps + 1):
        batch, batch_actions = sample_batch(
            train_videos, train_actions, train_config.batch_size
        )
        loss = trainer.train_step(batch, batch_actions)
        losses.append(loss)

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.train_steps
        ):
            elapsed = time.time() - start
            avg_loss = sum(losses[-train_config.log_every :]) / min(
                train_config.log_every, len(losses)
            )
            steps_per_sec = step / max(elapsed, 1e-8)
            val_batch, val_batch_actions = sample_batch(
                val_videos,
                val_actions,
                min(train_config.batch_size * 4, int(val_videos.shape[0])),
            )
            val_losses, val_psnrs = trainer.eval_loss_by_timestep(
                val_batch, val_batch_actions, val_timesteps
            )
            avg_val_loss = sum(val_losses.values()) / len(val_losses)
            avg_val_psnr = sum(val_psnrs.values()) / len(val_psnrs)
            val_report = (
                f"avg_val_t={avg_val_loss:.4f} avg_val_psnr={avg_val_psnr:.2f}dB"
            )

            print(
                f"step={step:5d} steps/s={steps_per_sec:.2f} "
                f"loss={loss:.4f} avg={avg_loss:.4f} {val_report}"
            )
            if train_logger is not None:
                train_logger.log_train_metrics(
                    step,
                    {
                        "loss": loss,
                        "avg_loss": avg_loss,
                        "steps_per_second": steps_per_sec,
                    },
                )
                train_logger.log_validation_steps(step, val_losses)
                train_logger.log_validation_psnrs(step, val_psnrs)

    if train_config.save_dir is not None:
        samples = sample_euler(
            trainer.model,
            conditioning_clips=val_conditioning_clips[:, :-1],
            actions=val_conditioning_actions,
            num_steps=train_config.sample_steps,
        )
        save_clip_previews(
            samples,
            train_config.save_dir,
            max_clips=sample_count,
            fps=train_config.preview_fps,
        )
        print(f"saved samples to: {train_config.save_dir}")

    return trainer.ema_model, losses


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
