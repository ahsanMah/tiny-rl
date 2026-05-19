from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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


def make_random_video_dataset(
    *,
    num_videos: int,
    frames: int,
    height: int,
    width: int,
    channels: int,
    seed: int = 0,
) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((num_videos, frames, height, width, channels))


def sample_batch(videos: mx.array, batch_size: int) -> mx.array:
    if batch_size >= videos.shape[0]:
        return videos[:batch_size]
    indices = mx.random.randint(0, videos.shape[0], shape=(batch_size,))
    return videos[indices]


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

    model = UNet3D(
        in_channels=channels,
        out_channels=channels,
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal flow-matching trainer for UNet3D on random-noise videos."
    )
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
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if min(args.frames, args.height, args.width) < 4:
        raise ValueError("frames, height, and width must all be >= 4")

    downsample_levels = 2
    min_spatial = min(args.frames, args.height, args.width)
    required = 2**downsample_levels
    if min_spatial < required:
        raise ValueError(
            f"Input sizes must be >= {required} for the current UNet depth; got "
            f"frames={args.frames}, height={args.height}, width={args.width}"
        )

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
