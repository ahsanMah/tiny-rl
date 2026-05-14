"""
Generic RL training runner that dispatches to different algorithms.
"""

import os
import sys

import click
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
}


@click.command()
@click.option(
    "--algorithm",
    required=True,
    type=click.Choice(["ppo_jax", "ppo", "vectorized_gae"]),
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
def main(
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
):
    # Pretty print the run configuration
    table = Table(title="Run Configuration", show_lines=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Algorithm", algorithm)
    table.add_row("Environment", env_name)
    table.add_row("Parallel Envs", str(num_parallel_envs))
    table.add_row("Timesteps/Epoch", str(num_timesteps_per_epoch))
    table.add_row("Max Episode Steps", str(max_episode_steps))
    table.add_row("Hidden Dim", str(hidden_dim))
    table.add_row("Init Scale", str(init_scale))
    table.add_row("Init Scale Final", str(init_scale_final))
    table.add_row("Value Init Scale", str(value_init_scale))
    table.add_row("Value Init Scale Final", str(value_init_scale_final))
    table.add_row("Policy LR", str(policy_lr))
    table.add_row("Value LR", str(value_lr))
    table.add_row("Grad Clip", str(grad_clip))
    table.add_row("Num Epochs", str(num_epochs))
    table.add_row("Num Trajectories", str(num_trajectories))
    table.add_row("Value Batch Size", str(value_batch_size))
    table.add_row("Discount Factor", str(discount))
    table.add_row("EMA Factor", str(ema))
    table.add_row("State Normalization", str(state_normalization))
    table.add_row("Seed", str(seed))
    table.add_row("Log Dir", log_dir)
    table.add_row("Eval Log Dir", eval_log_dir)
    table.add_row("Record Eval Videos", str(record_eval_videos))

    console.print(
        Panel(
            table,
            title="[bold red]Starting RL Training Run[/bold red]",
            border_style="red",
        )
    )

    # Dynamically import and run the selected algorithm
    algo_module = __import__(f"algorithms.{algorithm}", fromlist=["run"])
    algo_kwargs = {
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
    }

    algo_module.run(**algo_kwargs)


if __name__ == "__main__":
    main()
