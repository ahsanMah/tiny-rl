"""
Generic RL training runner that dispatches to different algorithms.
"""

import inspect

import click
from click.core import ParameterSource
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Default configuration values
DEFAULTS = {
    "env_name": "CartPole-v1",
    "num_parallel_envs": 4,
    "num_timesteps_per_epoch": 2048,
    "max_episode_steps": 500,
    "hidden_dim": 32,
    "init_scale": 0.1,
    "init_scale_final": 0.01,
    "value_init_scale": 0.1,
    "value_init_scale_final": 1.0,
    "policy_lr": 0.01,
    "value_lr": 0.02,
    "grad_clip_value": 2.0,
    "num_epochs": 20,
    "num_trajectories": 128,
    "value_batch_size": 32,
    "discount_factor": 0.95,
    "ema_factor": 0.96,
    "state_normalization": True,
    "record_eval_videos": False,
    "log_dir": "./tb-logs/",
    "eval_log_dir": "./eval-logs",
    "seed": 42,
    "gamma": 0.975,
    "alpha": 0.2,
    "num_updates_per_epoch": 1000,
    "batch_size": 256,
    "num_rollouts_per_epoch": 1000,
}


def _to_option_name(param_name: str) -> str:
    return f"--{param_name.replace('_', '-')}"


@click.command()
@click.option(
    "--algorithm",
    required=True,
    type=click.Choice(["ppo_jax", "ppo", "vectorized_gae", "sac"]),
    help="Algorithm to run",
)
@click.option("--env-name", default=DEFAULTS["env_name"], show_default=True)
@click.option(
    "--num-parallel-envs",
    default=DEFAULTS["num_parallel_envs"],
    show_default=True,
    type=int,
)
@click.option(
    "--num-timesteps-per-epoch",
    default=DEFAULTS["num_timesteps_per_epoch"],
    show_default=True,
    type=int,
)
@click.option(
    "--max-episode-steps",
    default=DEFAULTS["max_episode_steps"],
    show_default=True,
    type=int,
)
@click.option(
    "--hidden-dim", default=DEFAULTS["hidden_dim"], show_default=True, type=int
)
@click.option(
    "--init-scale", default=DEFAULTS["init_scale"], show_default=True, type=float
)
@click.option(
    "--init-scale-final",
    default=DEFAULTS["init_scale_final"],
    show_default=True,
    type=float,
)
@click.option(
    "--value-init-scale",
    default=DEFAULTS["value_init_scale"],
    show_default=True,
    type=float,
)
@click.option(
    "--value-init-scale-final",
    default=DEFAULTS["value_init_scale_final"],
    show_default=True,
    type=float,
)
@click.option(
    "--policy-lr", default=DEFAULTS["policy_lr"], show_default=True, type=float
)
@click.option("--value-lr", default=DEFAULTS["value_lr"], show_default=True, type=float)
@click.option(
    "--grad-clip-value",
    "grad_clip",
    default=DEFAULTS["grad_clip_value"],
    show_default=True,
    type=float,
)
@click.option(
    "--num-epochs", default=DEFAULTS["num_epochs"], show_default=True, type=int
)
@click.option(
    "--num-trajectories",
    default=DEFAULTS["num_trajectories"],
    show_default=True,
    type=int,
)
@click.option(
    "--value-batch-size",
    "value_batch_size",
    default=DEFAULTS["value_batch_size"],
    show_default=True,
    type=int,
)
@click.option(
    "--discount-factor",
    "discount",
    default=DEFAULTS["discount_factor"],
    show_default=True,
    type=float,
)
@click.option(
    "--ema-factor", "ema", default=DEFAULTS["ema_factor"], show_default=True, type=float
)
@click.option(
    "--state-normalization",
    default=DEFAULTS["state_normalization"],
    show_default=True,
    type=bool,
)
@click.option("--seed", default=DEFAULTS["seed"], type=int, show_default=True)
@click.option("--log-dir", default=DEFAULTS["log_dir"], show_default=True)
@click.option("--eval-log-dir", default=DEFAULTS["eval_log_dir"], show_default=True)
@click.option(
    "--record-eval-videos",
    "record_eval_videos",
    default=DEFAULTS["record_eval_videos"],
    show_default=True,
    type=bool,
)
@click.option("--gamma", default=DEFAULTS["gamma"], show_default=True, type=float)
@click.option("--alpha", default=DEFAULTS["alpha"], show_default=True, type=float)
@click.option(
    "--num-updates-per-epoch",
    default=DEFAULTS["num_updates_per_epoch"],
    show_default=True,
    type=int,
)
@click.option("--batch-size", default=DEFAULTS["batch_size"], show_default=True, type=int)
@click.option(
    "--num-rollouts-per-epoch",
    default=DEFAULTS["num_rollouts_per_epoch"],
    show_default=True,
    type=int,
)
@click.pass_context
def main(
    ctx,
    algorithm,
    env_name,
    num_parallel_envs,
    num_timesteps_per_epoch,
    max_episode_steps,
    hidden_dim,
    init_scale,
    init_scale_final,
    value_init_scale,
    value_init_scale_final,
    policy_lr,
    value_lr,
    grad_clip,
    num_epochs,
    num_trajectories,
    value_batch_size,
    state_normalization,
    discount,
    ema,
    seed,
    log_dir,
    eval_log_dir,
    record_eval_videos,
    gamma,
    alpha,
    num_updates_per_epoch,
    batch_size,
    num_rollouts_per_epoch,
):
    # Dynamically import selected algorithm and inspect its run signature.
    algo_module = __import__(f"algorithms.{algorithm}", fromlist=["run"])
    run_signature = inspect.signature(algo_module.run)
    run_parameters = run_signature.parameters

    all_cli_kwargs = {
        "env_name": env_name,
        "num_parallel_envs": num_parallel_envs,
        "max_episode_steps": max_episode_steps,
        "hidden_dim": hidden_dim,
        "init_scale": init_scale,
        "init_scale_final": init_scale_final,
        "value_init_scale": value_init_scale,
        "value_init_scale_final": value_init_scale_final,
        "policy_lr": policy_lr,
        "value_lr": value_lr,
        "grad_clip": grad_clip,
        "num_epochs": num_epochs,
        "num_trajectories": num_trajectories,
        "value_batch_size": value_batch_size,
        "state_normalization": state_normalization,
        "discount": discount,
        "ema": ema,
        "seed": seed,
        "log_dir": log_dir,
        "eval_log_dir": eval_log_dir,
        "record_eval_videos": record_eval_videos,
        "num_timesteps_per_epoch": num_timesteps_per_epoch,
        "gamma": gamma,
        "alpha": alpha,
        "num_updates_per_epoch": num_updates_per_epoch,
        "batch_size": batch_size,
        "num_rollouts_per_epoch": num_rollouts_per_epoch,
    }

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in run_parameters.values()
    )

    if not accepts_var_kwargs:
        unsupported_options = [
            _to_option_name(name)
            for name in all_cli_kwargs
            if name not in run_parameters
            and ctx.get_parameter_source(name) != ParameterSource.DEFAULT
        ]
        if unsupported_options:
            raise click.UsageError(
                f"Unsupported options for algorithm '{algorithm}': "
                f"{', '.join(sorted(unsupported_options))}"
            )

    if accepts_var_kwargs:
        algo_kwargs = all_cli_kwargs
    else:
        algo_kwargs = {}
        for name, parameter in run_parameters.items():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue

            if name not in all_cli_kwargs:
                continue

            parameter_source = ctx.get_parameter_source(name)
            is_required = parameter.default is inspect.Parameter.empty
            is_user_supplied = parameter_source != ParameterSource.DEFAULT
            if is_required or is_user_supplied:
                algo_kwargs[name] = all_cli_kwargs[name]

    missing_required_args = [
        name
        for name, parameter in run_parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Parameter.empty
        and name not in algo_kwargs
    ]
    if missing_required_args:
        missing_options = ", ".join(_to_option_name(name) for name in missing_required_args)
        raise click.UsageError(
            f"Missing required options for algorithm '{algorithm}': {missing_options}"
        )

    # Pretty print the selected run configuration.
    table = Table(title="Run Configuration", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_row("algorithm", algorithm)

    if accepts_var_kwargs:
        display_kwargs = algo_kwargs
    else:
        display_kwargs = {}
        for name, parameter in run_parameters.items():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if name in algo_kwargs:
                display_kwargs[name] = algo_kwargs[name]
            elif parameter.default is not inspect.Parameter.empty:
                display_kwargs[name] = parameter.default

    for name, value in display_kwargs.items():
        table.add_row(name, str(value))

    console.print(
        Panel(
            table,
            title="[bold red]Starting RL Training Run[/bold red]",
            border_style="red",
        )
    )

    algo_module.run(**algo_kwargs)


if __name__ == "__main__":
    main()
