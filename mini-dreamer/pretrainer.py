from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import click
import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs in gymnasium
import mlx.core as mx
import numpy as np
from click.core import ParameterSource
from minigrid.wrappers import RGBImgObsWrapper

from diffusion import (
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


CONFIG_SCHEMA: dict[str, set[str]] = {
    "env": {
        "env_id",
        "rollout_steps",
        "tile_size",
        "seed",
        "clip_length",
        "clip_stride",
        "max_clips",
    },
    "train": {
        "save_dir",
        "load_dir",
        "base_channels",
        "train_steps",
        "batch_size",
        "learning_rate",
        "log_every",
    },
    "preview": {"preview_dir", "preview_clips", "preview_fps"},
    "generate": {
        "load_dir",
        "save_dir",
        "generate_new_frames",
        "generate_num_steps",
        "num_samples",
    },
}
TRAIN_CONFIG_OPTIONS = {
    "env_id": "env",
    "rollout_steps": "env",
    "tile_size": "env",
    "seed": "env",
    "clip_length": "env",
    "clip_stride": "env",
    "max_clips": "env",
    "save_dir": "train",
    "load_dir": "train",
    "base_channels": "train",
    "train_steps": "train",
    "batch_size": "train",
    "learning_rate": "train",
    "log_every": "train",
    "preview_dir": "preview",
    "preview_clips": "preview",
    "preview_fps": "preview",
}
GENERATE_CONFIG_OPTIONS = {
    "env_id": "env",
    "rollout_steps": "env",
    "tile_size": "env",
    "seed": "env",
    "clip_length": "env",
    "clip_stride": "env",
    "max_clips": "env",
    "load_dir": "generate",
    "save_dir": "generate",
    "generate_new_frames": "generate",
    "generate_num_steps": "generate",
    "num_samples": "generate",
    "preview_fps": "preview",
}


def _load_experiment_config(config_path: Path | None) -> dict[str, dict[str, Any]]:
    if config_path is None:
        return {}

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


def _resolve_command_options(
    ctx: click.Context,
    *,
    config_path: Path | None,
    option_sections: dict[str, str],
) -> dict[str, Any]:
    resolved = dict(ctx.params)
    config = _load_experiment_config(config_path)
    option_lookup = {
        param.name: param
        for param in ctx.command.params
        if isinstance(param, click.Option) and param.name is not None
    }

    for option_name, section in option_sections.items():
        if ctx.get_parameter_source(option_name) is not ParameterSource.DEFAULT:
            continue
        if option_name not in config.get(section, {}):
            continue
        raw_value = config[section][option_name]
        try:
            resolved[option_name] = option_lookup[option_name].type_cast_value(
                ctx, raw_value
            )
        except click.ClickException as exc:
            raise click.ClickException(
                f"Invalid value for [{section}].{option_name}: {raw_value!r}"
            ) from exc

    return resolved


def _config_option(func):
    return click.option(
        "--config",
        "config_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Path to an experiment config TOML file.",
    )(func)


def _env_options(func):
    func = click.option("--env-id", default="MiniGrid-Empty-8x8-v0")(func)
    func = click.option("--rollout-steps", default=32, type=int)(func)
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


@cli.command(name="train")
@_config_option
@_env_options
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
    config_path: Path | None,
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
    resolved = _resolve_command_options(
        ctx, config_path=config_path, option_sections=TRAIN_CONFIG_OPTIONS
    )
    env_id = resolved["env_id"]
    rollout_steps = resolved["rollout_steps"]
    tile_size = resolved["tile_size"]
    seed = resolved["seed"]
    clip_length = resolved["clip_length"]
    clip_stride = resolved["clip_stride"]
    max_clips = resolved["max_clips"]
    save_dir = resolved["save_dir"]
    load_dir = resolved["load_dir"]
    base_channels = resolved["base_channels"]
    train_steps = resolved["train_steps"]
    batch_size = resolved["batch_size"]
    learning_rate = resolved["learning_rate"]
    log_every = resolved["log_every"]
    preview_dir = resolved["preview_dir"]
    preview_clips = resolved["preview_clips"]
    preview_fps = resolved["preview_fps"]

    env = gym.make(env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=rollout_steps,
        tile_size=tile_size,
        seed=seed,
        clip_length=clip_length,
        clip_stride=clip_stride,
        max_clips=max_clips,
    )
    print(f"env: {env_id}")
    print(f"clips shape: {tuple(clips.shape)}")
    print(f"action clips shape: {tuple(action_clips.shape)}")

    if preview_dir is not None:
        save_clip_previews(clips, preview_dir, max_clips=preview_clips, fps=preview_fps)
        print(f"saved previews to: {preview_dir}")

    save_path = Path(save_dir)
    num_actions = int(env.action_space.n)
    train_logger = RLLogger(log_dir=str(save_path.parent), exp_name=save_path.name)

    initial_model = None
    if load_dir is not None:
        initial_model = load_model(load_dir)
        print(f"resuming training from: {load_dir}")

    try:
        model, _ = train_on_dataset(
            clips,
            actions=action_clips,
            num_env_actions=num_actions,
            base_channels=base_channels,
            batch_size=batch_size,
            steps=train_steps,
            sample_fps=preview_fps,
            sample_dir=str(save_path),
            log_every=log_every,
            model=initial_model,
            learning_rate=learning_rate,
            train_logger=train_logger,
        )

        save_model(model, save_path, config=infer_model_config(model))
        print(f"saved model to: {save_path}")
    finally:
        train_logger.close()


@cli.command(name="generate")
@_config_option
@_env_options
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
    config_path: Path | None,
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
    **_ignored,
) -> None:
    """Load a pretrained model and autoregressively generate a video."""
    resolved = _resolve_command_options(
        ctx, config_path=config_path, option_sections=GENERATE_CONFIG_OPTIONS
    )
    env_id = resolved["env_id"]
    tile_size = resolved["tile_size"]
    seed = resolved["seed"]
    clip_length = resolved["clip_length"]
    clip_stride = resolved["clip_stride"]
    max_clips = resolved["max_clips"]
    load_dir = resolved["load_dir"]
    save_dir = resolved["save_dir"]
    generate_new_frames = resolved["generate_new_frames"]
    generate_num_steps = resolved["generate_num_steps"]
    num_samples = resolved["num_samples"]
    preview_fps = resolved["preview_fps"]

    if load_dir is None:
        raise click.UsageError("Missing option '--load-dir'.")

    env = gym.make(env_id)
    clips, action_clips = make_minigrid_dataset(
        env=env,
        num_steps=num_samples * clip_length,
        tile_size=tile_size,
        seed=seed,
        clip_length=clip_length - 1,  # only grab context clips
        clip_stride=clip_stride,
        max_clips=max_clips,
        max_action_idx=3,
    )
    print(f"env: {env_id}")
    print(f"clips shape: {tuple(clips.shape)}")

    num_actions = 3  # int(env.action_space.n)
    model = load_model(load_dir)
    print(f"loaded model from: {load_dir}")

    out_dir = (
        Path(save_dir)
        if save_dir is not None
        else Path(load_dir) / f"generated-{generate_new_frames}f-{generate_num_steps}s"
    )
    sample_count = min(num_samples, int(clips.shape[0]))
    generate_minigrid_video(
        model,
        initial_clip=clips[:sample_count:, : model.max_context_size + 1],
        initial_actions=action_clips[:sample_count],
        num_actions=num_actions,
        num_new_frames=generate_new_frames,
        num_steps=generate_num_steps,
        sample_fps=preview_fps,
        save_dir=out_dir,
        seed=seed + 1,
    )


if __name__ == "__main__":
    cli()
