from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as iio
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from unet import UNet3D, print_param_table


class FlowMatchingTrainer:
    def __init__(
        self,
        model: UNet3D,
        *,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.loss_and_grad = nn.value_and_grad(model, self.loss)

    def loss(self, model: UNet3D, x1: mx.array) -> mx.array:
        x0 = mx.random.normal(x1.shape, dtype=x1.dtype)
        t = mx.random.uniform(shape=(x1.shape[0],), low=0.0, high=1.0)
        t_view = mx.reshape(t, (x1.shape[0], 1, 1, 1, 1))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        pred_velocity = model(xt, t)
        return mx.mean((pred_velocity - target_velocity) ** 2)

    def train_step(self, batch: mx.array) -> float:
        loss, grads = self.loss_and_grad(self.model, batch)
        self.optimizer.update(self.model, grads)
        mx.eval(loss, self.model.parameters(), self.optimizer.state)
        return float(loss)


def require_valid_unet_shape(frames: int, height: int, width: int) -> None:
    dims = {"frames": frames, "height": height, "width": width}
    for name, value in dims.items():
        if value < 4:
            raise ValueError(f"{name} must be >= 4, got {value}")
        if value % 4 != 0:
            raise ValueError(f"{name} must be divisible by 4, got {value}")


def make_random_video_dataset(
    *,
    num_videos: int,
    frames: int,
    height: int,
    width: int,
    channels: int,
    seed: int = 0,
) -> mx.array:
    require_valid_unet_shape(frames, height, width)
    mx.random.seed(seed)
    return mx.random.normal((num_videos, frames, height, width, channels))


def load_video_frames(
    path: str | Path,
    *,
    target_fps: float = 8.0,
    half_resolution: bool = True,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, float | tuple[int, int] | int]]:
    path = Path(path)
    reader = iio.get_reader(path, format="ffmpeg")
    meta = reader.get_meta_data()

    source_fps = float(meta.get("fps", target_fps))
    frame_step = max(int(round(source_fps / target_fps)), 1)
    actual_fps = source_fps / frame_step

    frames: list[np.ndarray] = []
    for index, frame in enumerate(reader):
        if index % frame_step != 0:
            continue
        frame = np.asarray(frame)
        if half_resolution:
            frame = frame[::2, ::2]
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    reader.close()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    frame_array = np.stack(frames, axis=0)
    source_size = meta.get("size")
    if source_size is None:
        source_size = (int(frame_array.shape[2]), int(frame_array.shape[1]))

    info: dict[str, float | tuple[int, int] | int] = {
        "source_fps": source_fps,
        "actual_fps": actual_fps,
        "frame_step": frame_step,
        "source_size": tuple(source_size),
        "processed_size": (int(frame_array.shape[2]), int(frame_array.shape[1])),
        "num_frames": int(frame_array.shape[0]),
    }
    return frame_array, info


def frames_to_clips(
    frames: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> mx.array:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T, H, W, C), got {frames.shape}")
    if frames.shape[0] < clip_length:
        raise ValueError(
            f"Need at least {clip_length} frames, but only loaded {frames.shape[0]}"
        )

    clip_stride = clip_length if clip_stride is None else clip_stride
    require_valid_unet_shape(clip_length, int(frames.shape[1]), int(frames.shape[2]))

    clips = []
    for start in range(0, frames.shape[0] - clip_length + 1, clip_stride):
        clips.append(frames[start : start + clip_length])
        if max_clips is not None and len(clips) >= max_clips:
            break

    if not clips:
        raise ValueError("No clips could be formed from the loaded video")

    return mx.array(np.stack(clips, axis=0))


def load_video_dataset(
    path: str | Path,
    *,
    clip_length: int,
    target_fps: float = 8.0,
    half_resolution: bool = True,
    clip_stride: int | None = None,
    max_clips: int | None = None,
) -> tuple[mx.array, dict[str, float | tuple[int, int] | int]]:
    frames, info = load_video_frames(
        path,
        target_fps=target_fps,
        half_resolution=half_resolution,
    )
    clips = frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    info["num_clips"] = int(clips.shape[0])
    info["clip_length"] = int(clips.shape[1])
    return clips, info


def sample_batch(videos: mx.array, batch_size: int) -> mx.array:
    if batch_size >= videos.shape[0]:
        return videos[:batch_size]
    indices = mx.random.randint(0, videos.shape[0], shape=(batch_size,))
    return videos[indices]


def train_on_dataset(
    videos: mx.array,
    *,
    steps: int = 1_000,
    batch_size: int = 1,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    log_every: int = 50,
):
    model = UNet3D(
        in_channels=int(videos.shape[-1]),
        out_channels=int(videos.shape[-1]),
        base_channels=base_channels,
    )
    trainer = FlowMatchingTrainer(model, learning_rate=learning_rate)

    print("dataset:", tuple(videos.shape))
    print_param_table(model)

    start = time.time()
    losses: list[float] = []

    for step in range(1, steps + 1):
        batch = sample_batch(videos, batch_size)
        loss = trainer.train_step(batch)
        losses.append(loss)

        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.time() - start
            avg_loss = sum(losses[-log_every:]) / min(log_every, len(losses))
            steps_per_sec = step / max(elapsed, 1e-8)
            print(
                f"step={step:5d} loss={loss:.6f} avg={avg_loss:.6f} "
                f"steps/s={steps_per_sec:.2f}"
            )

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
    )


def train_overfit_video(
    video_path: str | Path,
    *,
    steps: int = 1_000,
    batch_size: int = 1,
    frames: int = 4,
    target_fps: float = 8.0,
    half_resolution: bool = True,
    clip_stride: int | None = None,
    max_clips: int | None = None,
    base_channels: int = 16,
    learning_rate: float = 1e-3,
    log_every: int = 50,
):
    videos, info = load_video_dataset(
        video_path,
        clip_length=frames,
        target_fps=target_fps,
        half_resolution=half_resolution,
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
            "num_frames": int(info["num_frames"]),
            "num_clips": int(info["num_clips"]),
            "clip_length": int(info["clip_length"]),
        },
    )
    return train_on_dataset(
        videos,
        steps=steps,
        batch_size=batch_size,
        base_channels=base_channels,
        learning_rate=learning_rate,
        log_every=log_every,
    )


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
    parser.add_argument("--clip-stride", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--full-resolution", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.video is not None:
        train_overfit_video(
            args.video,
            steps=args.steps,
            batch_size=args.batch_size,
            frames=args.frames,
            target_fps=args.video_target_fps,
            half_resolution=not args.full_resolution,
            clip_stride=args.clip_stride,
            max_clips=args.max_clips,
            base_channels=args.base_channels,
            learning_rate=args.learning_rate,
            log_every=args.log_every,
        )
        return

    train_overfit_random_noise(
        steps=args.steps,
        batch_size=args.batch_size,
        num_videos=args.num_videos,
        frames=args.frames,
        height=args.height,
        width=args.width,
        channels=args.channels,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
