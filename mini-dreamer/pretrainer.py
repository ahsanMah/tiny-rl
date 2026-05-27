from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import click
import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from diffusion import (
    ModelConfig,
    TrainConfig,
    generate_video,
    infer_model_config,
    load_model,
    save_model,
    train_on_dataset,
)
from logger_utils import RLLogger
from unet import UNet3D
from video_utils import frames_to_clips, save_clip_previews


def rollout_minigrid_frames(
    env: gym.Env,
    *,
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 0,
    max_action_idx: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out random actions in a MiniGrid env and capture tile-rendered RGB frames.

    Returns:
        frames: array of shape (num_steps, H, W, 3) of float32 in [-1, 1].
        actions: array of shape (num_steps,) of int32. `actions[i]` is the
            action taken at `frames[i]` (which produced the next frame).
    Episodes auto-reset on terminate/truncate so the frame stream is contiguous.
    """

    env = RGBImgObsWrapper(env, tile_size=tile_size)
    env.reset(seed=seed)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    max_action_idx = env.action_space.n if max_action_idx == -1 else max_action_idx

    for _ in range(num_steps):
        action = env.action_space.sample() % max_action_idx
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)

        frame = obs["image"]
        frames.append(np.asarray(frame))
        if terminated or truncated:
            env.reset()

    env.close()

    stacked = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    action_array = np.asarray(actions, dtype=np.int32)
    return stacked, action_array


def actions_to_clips(
    actions: np.ndarray,
    *,
    clip_length: int,
    clip_stride: int | None = None,
    max_clips: int | None = None,
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
        if max_clips is not None and len(clips) >= max_clips:
            break

    return mx.array(np.stack(clips, axis=0))


def make_minigrid_dataset(
    env: gym.Env,
    *,
    num_steps: int = 256,
    tile_size: int = 8,
    seed: int = 42,
    clip_length: int = 4,
    clip_stride: int | None = None,
    max_clips: int | None = None,
    max_action_idx: int = -1,
) -> tuple[mx.array, mx.array]:
    frames, actions = rollout_minigrid_frames(
        env=env,
        num_steps=num_steps,
        tile_size=tile_size,
        seed=seed,
        max_action_idx=max_action_idx,
    )
    frame_clips = frames_to_clips(
        frames,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    action_clips = actions_to_clips(
        actions,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    return frame_clips, action_clips


def generate_minigrid_video(
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
) -> mx.array:
    """Autoregressively extend `initial_clip` using random actions for new frames."""
    rng = np.random.default_rng(seed)
    initial_actions_np = np.asarray(initial_actions)
    batch_size = int(initial_clip.shape[0])
    extra_actions_np = rng.integers(
        0, num_actions, size=(batch_size, num_new_frames)
    ).astype(np.int32)
    full_actions = mx.array(
        np.concatenate([initial_actions_np, extra_actions_np], axis=1)
    )
    generated = generate_video(
        model,
        initial_clip=initial_clip,
        num_new_frames=num_new_frames,
        actions=full_actions,
        num_steps=num_steps,
    )
    save_clip_previews(
        generated,
        save_dir,
        max_clips=batch_size,
        fps=sample_fps,
        actions=full_actions,
    )
    print(f"saved generated video to: {save_dir}")
    return generated


@click.group()
def cli() -> None:
    """MiniGrid world-model pretraining and generation."""


@dataclass(frozen=True)
class EnvConfig:
    env_id: str


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


@dataclass(frozen=True)
class GenerateConfig:
    load_dir: str | None
    save_dir: str | None
    generate_new_frames: int
    generate_num_steps: int
    num_samples: int


_SECTION_DATACLASSES = {
    "env": EnvConfig,
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "generate": GenerateConfig,
}
CONFIG_SCHEMA = {
    section: {f.name for f in fields(cls)}
    for section, cls in _SECTION_DATACLASSES.items()
}


def _load_experiment_config(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    unknown_sections = set(config) - set(CONFIG_SCHEMA)
    if unknown_sections:
        section_list = ", ".join(sorted(unknown_sections))
        raise click.ClickException(f"Unknown config sections: {section_list}")

    for section, values in config.items():
        if not isinstance(values, dict):
            raise click.ClickException(
                f"Config section [{section}] must be a table, got {type(values).__name__}."
            )
        unknown_options = set(values) - CONFIG_SCHEMA[section]
        if unknown_options:
            option_list = ", ".join(sorted(unknown_options))
            raise click.ClickException(
                f"Unknown options in config section [{section}]: {option_list}"
            )

    return config


def _from_resolved(cls, resolved: dict[str, Any]):
    """Construct a dataclass from a resolved params dict, ignoring unrelated keys."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in resolved.items() if k in valid})


def _build_train_configs(
    resolved: dict[str, Any],
) -> tuple[EnvConfig, DatasetConfig, ModelConfig, TrainConfig]:
    return (
        _from_resolved(EnvConfig, resolved),
        _from_resolved(DatasetConfig, resolved),
        _from_resolved(ModelConfig, resolved),
        _from_resolved(TrainConfig, resolved),
    )


def _build_generate_configs(
    resolved: dict[str, Any],
) -> tuple[EnvConfig, DatasetConfig, GenerateConfig]:
    return (
        _from_resolved(EnvConfig, resolved),
        _from_resolved(DatasetConfig, resolved),
        _from_resolved(GenerateConfig, resolved),
    )


def _config_option(sections: list[str]):
    """Return a decorator that adds an eager --config option for the given TOML sections.

    When --config is provided, the named sections are merged into ctx.default_map so
    Click applies them as fallback defaults before the command function is called.
    Explicit CLI flags always take precedence.
    """

    def _callback(ctx: click.Context, _param: click.Parameter, value: Path | None) -> None:
        if value is None:
            return
        config = _load_experiment_config(value)
        ctx.default_map = {}
        for section in sections:
            ctx.default_map.update(config.get(section, {}))

    def decorator(func):
        return click.option(
            "--config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            is_eager=True,
            expose_value=False,
            callback=_callback,
            help="Path to an experiment config TOML file.",
        )(func)

    return decorator


def _dataset_options(func):
    func = click.option("--env-id", default="MiniGrid-Empty-8x8-v0")(func)
    func = click.option("--tile-size", default=8, type=int)(func)
    func = click.option("--seed", default=0, type=int)(func)
    func = click.option("--clip-length", default=4, type=int)(func)
    func = click.option(
        "--clip-stride",
        default=None,
        type=int,
        help="Stride between clips; defaults to 1 for a rolling window.",
    )(func)
    func = click.option("--max-clips", default=None, type=int)(func)
    return func


def _train_dataset_options(func):
    func = _dataset_options(func)
    func = click.option("--rollout-steps", default=32, type=int)(func)
    return func


@cli.command(name="train")
@_config_option(["env", "dataset", "model", "train"])
@_train_dataset_options
@click.option("--save-dir", default="logs/minigrid-v1")
@click.option(
    "--load-dir",
    default=None,
    help="Resume training from a checkpoint saved in this directory.",
)
@click.option("--base-channels", default=16, type=int)
@click.option("--train-steps", default=10_000, type=int)
@click.option("--batch-size", default=8, type=int)
@click.option("--learning-rate", default=3e-4, type=float)
@click.option("--log-every", default=100, type=int)
@click.option("--preview-dir", default=None)
@click.option("--preview-clips", default=4, type=int)
@click.option("--preview-fps", default=2.0, type=float)
@click.pass_context
def train_cmd(
    ctx: click.Context,
    env_id: str,
    rollout_steps: int,
    tile_size: int,
    seed: int,
    clip_length: int,
    clip_stride: int | None,
    max_clips: int | None,
    save_dir: str,
    load_dir: str | None,
    base_channels: int,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    log_every: int,
    preview_dir: str | None,
    preview_clips: int,
    preview_fps: float,
) -> None:
    """Train the diffusion world model on MiniGrid rollouts."""
    env_config, dataset_config, model_config, train_config = _build_train_configs(
        ctx.params
    )

    env = gym.make(env_config.env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=dataset_config.rollout_steps,
        tile_size=dataset_config.tile_size,
        seed=dataset_config.seed,
        clip_length=dataset_config.clip_length,
        clip_stride=dataset_config.clip_stride,
        max_clips=dataset_config.max_clips,
    )
    print(f"env: {env_config.env_id}")
    print(f"clips shape: {tuple(clips.shape)}")
    print(f"action clips shape: {tuple(action_clips.shape)}")

    if dataset_config.preview_dir is not None:
        save_clip_previews(
            clips,
            dataset_config.preview_dir,
            max_clips=dataset_config.preview_clips,
            fps=dataset_config.preview_fps,
        )
        print(f"saved previews to: {dataset_config.preview_dir}")

    save_path = Path(train_config.save_dir)
    num_actions = int(env.action_space.n)
    train_logger = RLLogger(log_dir=str(save_path.parent), exp_name=save_path.name)

    initial_model = None
    if train_config.load_dir is not None:
        initial_model = load_model(train_config.load_dir)
        print(f"resuming training from: {train_config.load_dir}")

    try:
        model, _ = train_on_dataset(
            clips,
            actions=action_clips,
            num_env_actions=num_actions,
            model_config=model_config,
            train_config=train_config,
            model=initial_model,
            train_logger=train_logger,
        )

        save_model(model, save_path, config=infer_model_config(model))
        print(f"saved model to: {save_path}")
    finally:
        train_logger.close()


@cli.command(name="generate")
@_config_option(["env", "dataset", "generate"])
@_dataset_options
@click.option(
    "--load-dir",
    default=None,
    help="Directory containing the saved model to load.",
)
@click.option(
    "--save-dir",
    default=None,
    help="Output directory for generated previews. Defaults to <load-dir>/generated.",
)
@click.option("--generate-new-frames", default=32, type=int)
@click.option("--generate-num-steps", default=32, type=int)
@click.option("--num-samples", default=1, type=int)
@click.option("--preview-fps", default=2.0, type=float)
@click.pass_context
def generate_cmd(
    ctx: click.Context,
    env_id: str,
    tile_size: int,
    seed: int,
    clip_length: int,
    clip_stride: int | None,
    max_clips: int | None,
    load_dir: str | None,
    save_dir: str | None,
    generate_new_frames: int,
    generate_num_steps: int,
    num_samples: int,
    preview_fps: float,
) -> None:
    """Load a pretrained model and autoregressively generate a video."""
    env_config, dataset_config, generate_config = _build_generate_configs(ctx.params)

    if generate_config.load_dir is None:
        raise click.UsageError("Missing option '--load-dir'.")

    env = gym.make(env_config.env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=generate_config.num_samples * dataset_config.clip_length,
        tile_size=dataset_config.tile_size,
        seed=dataset_config.seed,
        clip_length=dataset_config.clip_length - 1,  # only grab context clips
        clip_stride=dataset_config.clip_stride,
        max_clips=dataset_config.max_clips,
        max_action_idx=3,
    )
    print(f"env: {env_config.env_id}")
    print(f"clips shape: {tuple(clips.shape)}")

    num_actions = 3  # int(env.action_space.n)
    model = load_model(generate_config.load_dir)
    print(f"loaded model from: {generate_config.load_dir}")

    out_dir = (
        Path(generate_config.save_dir)
        if generate_config.save_dir is not None
        else Path(generate_config.load_dir)
        / f"generated-{generate_config.generate_new_frames}f-{generate_config.generate_num_steps}s"
    )
    sample_count = min(generate_config.num_samples, int(clips.shape[0]))
    generate_minigrid_video(
        model,
        initial_clip=clips[:sample_count, : model.max_context_size + 1],
        initial_actions=action_clips[:sample_count],
        num_actions=num_actions,
        num_new_frames=generate_config.generate_new_frames,
        num_steps=generate_config.generate_num_steps,
        sample_fps=dataset_config.preview_fps,
        save_dir=out_dir,
        seed=dataset_config.seed + 1,
    )


if __name__ == "__main__":
    cli()
