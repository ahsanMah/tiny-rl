"""Framework-generic JAX/NNX helpers shared by the ported trainers
(vae_jax.py, diffusion_jax.py): EMA, LR schedule, safetensors save/load,
and the param-table printer."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from safetensors.flax import load_file

from unet import format_param_table


def ema_update(ema_model: nnx.Module, model: nnx.Module, decay: float) -> None:
    """JAX port of ``diffusion.ema_update`` (that one operates on MLX modules)."""
    ema_params = nnx.state(ema_model, nnx.Param)
    params = nnx.state(model, nnx.Param)
    nnx.update(
        ema_model,
        jax.tree.map(
            lambda ema, current: decay * ema + (1.0 - decay) * current,
            ema_params,
            params,
        ),
    )


def print_param_table(model: nnx.Module) -> None:
    """Reuse ``unet.format_param_table`` via a ``.parameters()`` shim (it only
    needs a nested dict of arrays with ``shape``/``dtype``)."""
    params = nnx.state(model, nnx.Param).to_pure_dict()
    print(format_param_table(SimpleNamespace(parameters=lambda: params)))


def linear_warmup_decay_schedule(
    peak_lr: float,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    hold_steps: int = 0,
    final_lr: float | None = None,
) -> optax.Schedule:
    """Linear warmup from ``peak_lr / 10`` to ``peak_lr`` over ``warmup_steps``,
    hold at ``peak_lr`` for ``hold_steps``, then linear decay to ``final_lr``
    over the remaining steps.

    Every part is optional: ``warmup_steps=0`` skips the warmup, ``hold_steps=0``
    starts the decay right after warmup, and ``final_lr=None`` holds the LR
    constant after warmup. With everything disabled this returns the plain
    ``peak_lr`` float (a fixed learning rate).
    """
    constant = optax.constant_schedule(peak_lr)
    if warmup_steps <= 0 and final_lr is None:
        return constant
    tail = (
        optax.linear_schedule(
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
            0, optax.linear_schedule(peak_lr / 10.0, peak_lr, warmup_steps)
        )
        boundaries.insert(0, warmup_steps)

    if not boundaries:
        return tail

    return optax.join_schedules(schedules, boundaries)


def flat_params(model: nnx.Module) -> dict[str, jax.Array]:
    """Flatten trainable params to ``{'enc_stem.kernel': array, ...}`` for safetensors."""
    return {
        ".".join(map(str, path)): variable.value
        for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))
    }


def load_flat_params(model: nnx.Module, weights_path: str | Path) -> None:
    tensors = load_file(str(weights_path))
    flat = nnx.to_flat_state(nnx.state(model, nnx.Param))
    nnx.update(
        model,
        nnx.from_flat_state(
            [
                (path, variable.replace(tensors[".".join(map(str, path))]))
                for path, variable in flat
            ]
        ),
    )
