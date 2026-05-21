from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from unet import UNet3D, print_param_table
from video_utils import (
    load_video_dataset,
    make_random_video_dataset,
    save_clip_previews,
)


def make_final_frame_mask(x: mx.array) -> mx.array:
    frame_mask = (mx.arange(x.shape[1]) == (x.shape[1] - 1)).astype(x.dtype)
    return mx.reshape(frame_mask, (1, x.shape[1], 1, 1, 1))


class FlowMatchingTrainer:
    def __init__(
        self,
        model: UNet3D,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.loss_and_grad = nn.value_and_grad(model, self.loss)

    def loss(
        self, model: UNet3D, x1: mx.array, actions: mx.array | None = None
    ) -> mx.array:
        mask = make_final_frame_mask(x1)
        noise = mx.random.normal(x1.shape, dtype=x1.dtype) * mask
        t = mx.random.uniform(shape=(x1.shape[0],), low=0.0, high=1.0)
        t_view = mx.reshape(t, (x1.shape[0], 1, 1, 1, 1)) * mask
        xt = (1.0 - t_view) * noise + t_view * x1 + (1 - mask) * x1
        target_velocity = mask * (x1 - noise)
        pred_velocity = model(xt, t, context=actions)
        return mx.mean((pred_velocity[:, -1:] - target_velocity[:, -1:]) ** 2)

    def train_step(self, batch: mx.array, actions: mx.array) -> float:
        loss, grads = self.loss_and_grad(self.model, batch, actions)
        self.optimizer.update(self.model, grads)
        mx.eval(loss, self.model.parameters(), self.optimizer.state)
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
    actions: mx.array | None = None,
    num_steps: int = 32,
) -> mx.array:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be > 0, got {num_steps}")

    batch_size = int(conditioning_clips.shape[0])
    context = conditioning_clips[:, :-1]
    target = conditioning_clips[:, -1:]
    x = mx.random.normal(target.shape, dtype=conditioning_clips.dtype)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = mx.full((batch_size,), step / num_steps, dtype=conditioning_clips.dtype)
        xt = mx.concatenate([context, x], axis=1)
        v = model(xt, t, context=actions)
        x = x + dt * v[:, -1:]

    samples = mx.concatenate([context, x], axis=1)
    mx.eval(samples)
    return samples


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


def load_and_generate(
    save_dir: str | Path,
    *,
    initial_clip: mx.array,
    num_new_frames: int,
    actions: mx.array | None = None,
    num_steps: int = 32,
) -> mx.array:
    """Load a saved model and autoregressively generate a video."""
    model = load_model(save_dir)
    return generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=actions,
        num_steps=num_steps,
    )


def generate_video(
    model: UNet3D,
    *,
    initial_clip: mx.array,
    num_new_frames: int,
    actions: mx.array | None = None,
    num_steps: int = 32,
) -> mx.array:
    """Autoregressively extend `initial_clip` by `num_new_frames` new frames.

    Args:
        initial_clip: (B, L, H, W, C) seed frames.
        num_new_frames: number of frames to generate after the seed.
        actions: optional (B, L + num_new_frames) action stream. The window
            of actions tracks the window of frames as we slide forward.
        num_steps: Euler integration steps per generated frame.

    Returns:
        (B, L + num_new_frames, H, W, C) array of frames.
    """
    if num_new_frames <= 0:
        raise ValueError(f"num_new_frames must be > 0, got {num_new_frames}")

    clip_length = int(initial_clip.shape[1])
    if actions is not None:
        expected = clip_length + num_new_frames
        if int(actions.shape[1]) != expected:
            raise ValueError(
                f"actions must have shape (B, {expected}), got {tuple(actions.shape)}"
            )

    frames = initial_clip
    for step in range(num_new_frames):
        window = frames[:, -clip_length:]
        action_window = (
            actions[:, step : step + clip_length] if actions is not None else None
        )
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
    action_dim: int = 1,
    steps: int = 1_000,
    batch_size: int = 1,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    log_every: int = 50,
    sample_dir: str | Path | None = None,
    num_gen_samples: int = 4,
    sample_steps: int = 32,
    sample_fps: float = 8.0,
    model: UNet3D | None = None,
):

    if actions is None:
        actions = mx.zeros((videos.shape[0], action_dim), dtype=mx.int8)

    if model is None:
        model = UNet3D(
            in_channels=int(videos.shape[-1]),
            out_channels=int(videos.shape[-1]),
            base_channels=base_channels,
            action_dim=action_dim,
        )
    trainer = FlowMatchingTrainer(model, learning_rate=learning_rate)

    print("dataset:", tuple(videos.shape))
    print_param_table(model)

    sample_count = min(num_gen_samples, int(videos.shape[0]))
    if sample_dir is not None:
        val_conditioning_clips = videos[:sample_count]
        val_conditioning_actions = actions[:sample_count]

        videos = videos[sample_count:]
        actions = actions[sample_count:]

    start = time.time()
    losses: list[float] = []

    for step in range(1, steps + 1):
        batch, batch_actions = sample_batch(videos, actions, batch_size)
        loss = trainer.train_step(batch, batch_actions)
        losses.append(loss)

        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.time() - start
            avg_loss = sum(losses[-log_every:]) / min(log_every, len(losses))
            steps_per_sec = step / max(elapsed, 1e-8)
            print(
                f"step={step:5d} loss={loss:.6f} avg={avg_loss:.6f} "
                f"steps/s={steps_per_sec:.2f}"
            )

    if sample_dir is not None:
        samples = sample_euler(
            model,
            conditioning_clips=val_conditioning_clips,
            actions=val_conditioning_actions,
            num_steps=sample_steps,
        )
        save_clip_previews(samples, sample_dir, max_clips=sample_count, fps=sample_fps)
        print(f"saved samples to: {sample_dir}")

    return model, losses


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
        steps=steps,
        batch_size=batch_size,
        base_channels=base_channels,
        learning_rate=learning_rate,
        log_every=log_every,
        sample_dir=sample_dir,
        num_gen_samples=num_samples,
        sample_steps=sample_steps,
        sample_fps=sample_fps,
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
        steps=steps,
        batch_size=batch_size,
        base_channels=base_channels,
        learning_rate=learning_rate,
        log_every=log_every,
        sample_dir=sample_dir,
        num_gen_samples=num_samples,
        sample_steps=sample_steps,
        sample_fps=float(info["actual_fps"]),
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
    parser.add_argument("--clip-stride", type=int, default=None)
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
