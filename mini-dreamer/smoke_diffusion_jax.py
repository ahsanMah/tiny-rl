"""Smoke tests for diffusion_jax — tiny shapes, cheapest-first ladder.

Run with: uv run smoke_diffusion_jax.py

Covers todo steps 2-5. Tests for not-yet-implemented pieces are skipped, so
this file runs green at every stage of the port. Pinned contract (agreed):

- t/t_ctx sampling and action dropout live *inside* train_step (no free
  functions), drawing keys from the trainer's own `nnx.Rngs`.
- `FlowMatchingTrainer(model, ema_model, *, learning_rate, weight_decay,
  max_grad_norm, ema_decay, action_dropout, reward_loss_weight,
  reward_t_threshold, min_context_t, sampling_distribution, logit_norm_mu,
  logit_norm_scale, seed)` — RNG state is built internally from `seed`;
  training loops never touch keys. (Requires the trainer to be an
  nnx.Module and the internal key to live in an nnx RNG Variable so step
  6's @nnx.jit tracks its advancement — with raw jax.jit it silently
  freezes.)
- `trainer.train_step(batch, actions, rewards=None) -> (loss, reward_loss)`,
  eager (jit equivalence is step 6).
- `trainer.eval_step(batch, actions) -> {t: (loss, psnr, r2, x1_pred)}` for
  each t in `trainer.eval_timesteps`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random

import diffusion_jax
from diffusion_jax import _loss_at_t, sample_noise, sample_t_logit_normal
from unet_jax import UNet3D


def _skip(name: str, *attrs: str) -> bool:
    missing = [a for a in attrs if not hasattr(diffusion_jax, a)]
    if missing:
        print(f"{name} SKIP (not implemented: {', '.join(missing)})")
    return bool(missing)


def _make_model(*, predict_reward: bool = False, use_wavelet: bool = False) -> UNet3D:
    return UNet3D(
        in_channels=3,
        num_actions=4,
        max_context_size=3,
        base_channels=8,
        num_transformer_blocks=1,
        use_wavelet=use_wavelet,
        predict_reward=predict_reward,
        rngs=nnx.Rngs(0),
    )


# --- step 2: _loss_at_t ---


def test_loss_at_t() -> None:
    B, L, H, W, C = 2, 4, 16, 16, 3
    key = random.key(0)
    k_x1, k_noise, k_t, k_tctx, k_actions, k_rewards = random.split(key, 6)
    x1 = random.normal(k_x1, (B, L, H, W, C))
    noise = sample_noise(k_noise, x1.shape, noise_distribution="normal")
    t = sample_t_logit_normal(k_t, (B,))
    t_ctx = random.uniform(k_tctx, (B,), minval=0.5, maxval=1.0)
    actions = random.randint(k_actions, (B, L), 0, 4)
    rewards = random.normal(k_rewards, (B,))

    for use_wavelet in (False, True):
        model = _make_model(use_wavelet=use_wavelet)
        label = "wavelet" if use_wavelet else "no wavelet"

        loss, reward_loss = _loss_at_t(model, x1, actions, t, noise=noise)
        assert loss.shape == (), loss.shape
        assert jnp.isfinite(loss), f"[{label}] loss not finite"
        assert reward_loss == 0.0, f"[{label}] reward_loss should be 0 without rewards"
        print(f"[{label}] loss={float(loss):.4f}")

        # t_ctx path
        loss_ctx, _ = _loss_at_t(model, x1, actions, t, noise=noise, t_ctx=t_ctx)
        assert jnp.isfinite(loss_ctx), f"[{label}] t_ctx loss not finite"
        print(f"[{label}] loss(t_ctx)={float(loss_ctx):.4f}")

        # eval aux path
        loss_e, recon_mse, x1_pred, r2 = _loss_at_t(
            model, x1, actions, t, noise=noise, return_eval_aux=True
        )
        assert jnp.allclose(loss_e, loss), f"[{label}] eval flow loss != train flow loss"
        assert x1_pred.shape == (B, 1, H, W, C), x1_pred.shape
        assert jnp.isfinite(recon_mse) and jnp.isfinite(r2)
        print(f"[{label}] recon_mse={float(recon_mse):.4f} r2={float(r2):.4f}")

    # reward head path
    model = _make_model(predict_reward=True)
    total, reward_loss = _loss_at_t(
        model,
        x1,
        actions,
        t,
        noise=noise,
        rewards=rewards,
        reward_loss_weight=0.5,
        reward_t_threshold=0.0,
    )
    flow_only, _ = _loss_at_t(model, x1, actions, t, noise=noise)
    assert jnp.isfinite(reward_loss)
    assert jnp.allclose(total, flow_only + 0.5 * reward_loss)
    print(f"[reward] total={float(total):.4f} reward_loss={float(reward_loss):.4f}")

    # threshold above every t -> reward loss masked to 0
    _, masked = _loss_at_t(
        model,
        x1,
        actions,
        t,
        noise=noise,
        rewards=rewards,
        reward_loss_weight=0.5,
        reward_t_threshold=2.0,
    )
    assert masked == 0.0, f"reward loss should be masked out, got {float(masked)}"
    print("test_loss_at_t OK")


# --- step 4: FlowMatchingTrainer.__init__ ---


def test_trainer_init() -> None:
    if _skip("test_trainer_init", "FlowMatchingTrainer"):
        return
    trainer = diffusion_jax.FlowMatchingTrainer(
        _make_model(), _make_model(), learning_rate=1e-3, seed=0
    )
    assert trainer is not None

    # reward_loss_weight > 0 without a reward head must raise
    try:
        diffusion_jax.FlowMatchingTrainer(
            _make_model(), _make_model(), reward_loss_weight=0.5, seed=0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("reward_loss_weight>0 without reward head should raise")

    # with a reward head it must construct fine
    diffusion_jax.FlowMatchingTrainer(
        _make_model(predict_reward=True),
        _make_model(predict_reward=True),
        reward_loss_weight=0.5,
        seed=0,
    )
    print("test_trainer_init OK")


# --- step 5: train_step, eager (overfit one repeated batch) ---


def _param_delta_norm(before: dict, after: dict) -> float:
    leaves_b = jax.tree.leaves(before)
    leaves_a = jax.tree.leaves(after)
    return float(
        jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(leaves_a, leaves_b)))
    )


def test_train_step_overfits() -> None:
    if _skip("test_train_step_overfits", "FlowMatchingTrainer"):
        return
    model = _make_model()
    ema_model = _make_model()
    # EMA starts as a copy of the model so its drift is measurable against
    # the same origin
    nnx.update(ema_model, nnx.state(model, nnx.Param))
    trainer = diffusion_jax.FlowMatchingTrainer(
        model,
        ema_model,
        learning_rate=5e-3,
        ema_decay=0.999,
        action_dropout=0.1,
        min_context_t=0.5,
        seed=42,
    )
    if not hasattr(trainer, "train_step"):
        print("test_train_step_overfits SKIP (no train_step yet)")
        return

    B, L, H, W, C = 4, 4, 16, 16, 3
    batch = random.normal(random.key(0), (B, L, H, W, C))
    actions = random.randint(random.key(1), (B, L), 0, 4)

    params_start = jax.tree.map(
        jnp.copy, nnx.to_flat_state(nnx.state(model, nnx.Param))
    )

    losses = []
    for _ in range(50):
        loss, reward_loss = trainer.train_step(batch, actions)
        assert jnp.isfinite(loss), "loss went non-finite"
        losses.append(float(loss))

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f} (avg {first:.4f} -> {last:.4f})")
    assert last < 0.7 * first, (
        f"loss should collapse on a repeated batch: {first:.4f} -> {last:.4f}"
    )

    # params actually moved (grads were nonzero) ...
    params_end = nnx.to_flat_state(nnx.state(model, nnx.Param))
    model_delta = _param_delta_norm(dict(params_start), dict(params_end))
    assert model_delta > 0.0, "model params did not move"
    # ... and the EMA shadow trails the online model
    ema_end = nnx.to_flat_state(nnx.state(ema_model, nnx.Param))
    ema_delta = _param_delta_norm(dict(params_start), dict(ema_end))
    assert 0.0 < ema_delta < model_delta, (
        f"EMA should move slower than model: ema {ema_delta:.4f} vs model {model_delta:.4f}"
    )

    # reward path: one step with rewards on a reward-head model stays finite
    r_trainer = diffusion_jax.FlowMatchingTrainer(
        _make_model(predict_reward=True),
        _make_model(predict_reward=True),
        learning_rate=1e-3,
        reward_loss_weight=0.5,
        reward_t_threshold=0.0,
        seed=43,
    )
    rewards = random.normal(random.key(2), (B,))
    loss, reward_loss = r_trainer.train_step(batch, actions, rewards)
    assert jnp.isfinite(loss) and jnp.isfinite(reward_loss)
    print(
        f"test_train_step_overfits OK (ema/model delta ratio "
        f"{ema_delta / model_delta:.4f})"
    )


# --- step 7: eval_step ---


def test_eval_step() -> None:
    if _skip("test_eval_step", "FlowMatchingTrainer"):
        return
    trainer = diffusion_jax.FlowMatchingTrainer(
        _make_model(), _make_model(), learning_rate=1e-3, seed=0
    )
    if not hasattr(trainer, "eval_step"):
        print("test_eval_step SKIP (no eval_step yet)")
        return

    B, L, H, W, C = 2, 4, 16, 16, 3
    batch = random.normal(random.key(0), (B, L, H, W, C))
    actions = random.randint(random.key(1), (B, L), 0, 4)

    metrics = trainer.eval_step(batch, actions)
    assert len(metrics) == len(trainer.eval_timesteps), (
        f"expected one entry per eval timestep, got {len(metrics)}"
    )

    psnr_by_t: dict[float, float] = {}
    for timestep, (loss, psnr, r2, x1_pred) in metrics.items():
        t = float(timestep)
        assert jnp.isfinite(loss), f"[t={t}] loss not finite"
        assert jnp.isfinite(psnr), f"[t={t}] psnr not finite"
        assert jnp.isfinite(r2), f"[t={t}] r2 not finite"
        assert x1_pred.shape == (B, 1, H, W, C), x1_pred.shape
        psnr_by_t[t] = float(psnr)
        print(f"  t={t:.3f} loss={float(loss):.4f} psnr={float(psnr):.2f} r2={float(r2):.4f}")

    # At t ~ 1 the target frame is nearly clean, so the one-step recon
    # x1_pred = xt + (1 - t) * v is ~x1 even for an untrained model; at
    # t ~ 0 it starts from pure noise. PSNR must reflect that.
    t_lo, t_hi = min(psnr_by_t), max(psnr_by_t)
    assert psnr_by_t[t_hi] > psnr_by_t[t_lo], (
        f"psnr at t={t_hi} ({psnr_by_t[t_hi]:.2f}) should beat "
        f"t={t_lo} ({psnr_by_t[t_lo]:.2f})"
    )
    print("test_eval_step OK")


if __name__ == "__main__":
    test_loss_at_t()
    test_trainer_init()
    test_train_step_overfits()
    test_eval_step()
