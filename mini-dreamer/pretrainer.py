import tomllib
import types
import typing
import wave
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from pprint import pprint
from typing import Any

import click

from data import make_dataset, make_env
from diffusion import (
    ModelConfig,
    TrainConfig,
    generate_env_video,
    load_model,
    sample_batch,
    save_model,
    train_on_dataset,
)
from logger_utils import RLLogger
from video_utils import save_clip_previews


@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "MiniGrid-Empty-8x8-v0"


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
    frame_skip: int = 1


@dataclass(frozen=True)
class GenerateConfig:
    load_dir: str | None = None
    save_dir: str | None = None
    generate_new_frames: int = 32
    generate_num_steps: int = 32
    num_samples: int = 1
    not_use_ema: bool = False


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


@click.group()
def cli() -> None:
    """MiniGrid world-model pretraining and generation."""

    """MiniGrid world-model pretraining and generation."""


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

    def _callback(
        ctx: click.Context, _param: click.Parameter, value: Path | None
    ) -> None:
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


def _dataclass_options(cls, *, exclude: frozenset[str] = frozenset()):
    """Decorator factory: emit one ``@click.option`` per field of ``cls``.

    Fields in ``exclude`` are skipped (use when another decorator already
    covers the same option name).  The options are inserted in the same
    order as the dataclass fields so ``--help`` output is readable.
    """
    hints = typing.get_type_hints(cls)

    def _click_kwargs(hint, default) -> dict:
        # Unwrap `X | None`
        if isinstance(hint, types.UnionType):
            inner = [a for a in hint.__args__ if a is not type(None)]
            if len(inner) == 1:
                return _click_kwargs(inner[0], default)
        if hint is bool:
            return {"is_flag": True, "default": bool(default)}
        if hint in (int, float, str):
            return {"type": hint, "default": default}
        # Path / other types: pass as string
        return {"type": str, "default": str(default) if default is not None else None}

    def decorator(func):
        for f in reversed(fields(cls)):
            if f.name in exclude:
                continue
            default = f.default if f.default is not MISSING else None
            kwargs = _click_kwargs(hints[f.name], default)
            func = click.option(f"--{f.name.replace('_', '-')}", **kwargs)(func)
        return func

    return decorator


@cli.command(name="train")
@_config_option(["env", "dataset", "model", "train"])
@_dataclass_options(EnvConfig)
@_dataclass_options(DatasetConfig)
@_dataclass_options(ModelConfig)
@_dataclass_options(TrainConfig)
@click.pass_context
def train_cmd(ctx: click.Context, **kwargs) -> None:
    """Train the diffusion world model on MiniGrid rollouts."""
    env_config, dataset_config, model_config, train_config = _build_train_configs(
        ctx.params
    )

    print("Using data config:")
    pprint(dataset_config)
    env = make_env(env_config.env_id)
    clips, action_clips = make_dataset(
        env=env,
        num_steps=dataset_config.rollout_steps,
        tile_size=dataset_config.tile_size,
        seed=dataset_config.seed,
        clip_length=dataset_config.clip_length,
        clip_stride=dataset_config.clip_stride,
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

    num_actions = int(env.action_space.n)
    save_path = Path(train_config.save_dir)

    model, ema_model, full_model_config = train_on_dataset(
        clips,
        actions=action_clips,
        num_env_actions=num_actions,
        model_config=model_config,
        train_config=train_config,
        sample_fps=dataset_config.preview_fps,
    )

    save_model(model, save_path, config=full_model_config, ema_model=ema_model)
    print(f"saved models to: {save_path}")


@cli.command(name="generate")
@_config_option(["env", "dataset", "generate"])
@_dataclass_options(EnvConfig)
@_dataclass_options(DatasetConfig)
@_dataclass_options(GenerateConfig)
@click.pass_context
def generate_cmd(ctx: click.Context, **kwargs) -> None:
    """Load a pretrained model and autoregressively generate a video."""
    env_config, dataset_config, generate_config = _build_generate_configs(ctx.params)

    if generate_config.load_dir is None:
        raise click.UsageError("Missing option '--load-dir'.")

    if generate_config.save_dir is None:
        raise click.UsageError("Missing option '--save-dir'.")

    model = load_model(
        generate_config.load_dir, prefer_ema=generate_config.not_use_ema == False
    )
    print(
        f"loaded {'ckpt' if generate_config.not_use_ema else 'ema'} model from: {generate_config.load_dir}"
    )

    env = make_env(env_config.env_id)
    max_action_idx = -1
    clip_length = model.max_context_size
    clips, action_clips = make_dataset(
        env=env,
        num_steps=generate_config.num_samples * dataset_config.clip_length * 4,
        tile_size=dataset_config.tile_size,
        seed=dataset_config.seed,
        clip_length=clip_length,  # only grab context clips
        clip_stride=dataset_config.clip_stride,
        max_action_idx=max_action_idx,
        warmup_steps=20,
    )

    print(f"clips shape: {tuple(clips.shape)}")
    sample_count = generate_config.num_samples
    sample_clips, sample_action_clips = sample_batch(
        videos=clips,
        actions=action_clips,
        batch_size=sample_count,
    )

    num_actions = env.action_space.n
    print(f"{sample_clips.shape = }")
    print(f"{sample_action_clips.shape = }")

    print(f"env: {env_config.env_id}")

    actions_pool = None  # [1, 2]
    out_dir = (
        Path(generate_config.save_dir)
        / f"{generate_config.generate_new_frames}f-{generate_config.generate_num_steps}s"
    )

    debug = True

    if debug:
        # save the wavelets
        wavelets = model.prepool(sample_clips)
        print(f"{wavelets.shape = }")
        B, T, H, W, C4 = wavelets.shape  # 8, 3, 120, 160, 12
        C = C4 // 4
        # split last dim → (4 subbands, 3 channels)
        wavelets = wavelets.reshape(B, T, H, W, 4, C)
        print(f"{wavelets.shape = }")

        # B, T, 4, H, W, C  (move channels next to T)
        wavelets = wavelets.transpose(0, 1, 4, 2, 3, 5)
        print(f"{wavelets.shape = }")

        # B, 4T, H, W, C
        wavelets = wavelets.reshape(B, T * 4, H, W, C)
        print(f"{wavelets.shape = }")

        save_clip_previews(wavelets, out_dir / "wavelets")

    generate_env_video(
        model,
        initial_clip=sample_clips[:, : model.max_context_size + 1],
        initial_actions=sample_action_clips,
        num_actions=num_actions,
        num_new_frames=generate_config.generate_new_frames,
        num_steps=generate_config.generate_num_steps,
        sample_fps=dataset_config.preview_fps,
        save_dir=out_dir,
        seed=dataset_config.seed + 1,
        actions_pool=actions_pool,
    )


if __name__ == "__main__":
    cli()
