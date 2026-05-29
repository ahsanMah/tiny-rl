# gym-practice

RL algorithms implemented from scratch using [Apple MLX](https://github.com/ml-explore/mlx), with a JAX port for GPU/TPU scaling. Each algorithm lives in `algorithms/` and can be run directly or through the unified CLI in `main.py`.

## Algorithms

| File | Algorithm | Notes |
|---|---|---|
| `vanilla_policy_grad.py` | Policy Gradient | Full returns, no baseline |
| `reinforce.py` | REINFORCE | Reward-to-go + optional value baseline |
| `reinforce_gae.py` | REINFORCE + GAE | Generalized Advantage Estimation |
| `vectorized_gae.py` | Vectorized GAE | Parallel envs via Gymnasium's `AsyncVectorEnv` |
| `ppo.py` | PPO | Clipped surrogate objective, continuous action support (MLX) |
| `ppo_jax.py` | PPO (JAX) | JAX/Flax port for GPU/TPU scaling experiments |
| `sac.py` | SAC | Soft Actor-Critic with EMA target network (MLX) |

## Running

Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
# Run a specific algorithm directly
uv run python algorithms/ppo.py

# Or use the unified CLI runner
uv run python main.py --algorithm ppo --env-name BipedalWalker-v3
```

The runner prints the full config and environment spec at startup, then dispatches only the parameters the selected algorithm actually accepts.

## Key Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--env-name` | `CartPole-v1` | Any Gymnasium environment ID |
| `--num-parallel-envs` | `4` | Vectorized environment count |
| `--num-timesteps-per-epoch` | `2048` | Steps collected per training epoch |
| `--hidden-dim` | `32` | Network hidden layer size |
| `--policy-lr` | `0.01` | Policy optimizer learning rate |
| `--value-lr` | `0.02` | Value function learning rate |
| `--grad-clip-value` | `2.0` | Gradient norm clip threshold |
| `--discount-factor` | `0.95` | Discount γ |
| `--ema-factor` | `0.96` | GAE λ (advantage decay) |
| `--state-normalization` | `true` | Welford running mean/var normalization |
| `--seed` | `42` | RNG seed |

## Logging

Training artifacts are written in two formats:
- **TensorBoard** → `./tb-logs/`
- **Dashboard JSON** → `./dashboard_artifacts/runs/<run_id>/`

The dashboard JSON format is documented in [`../dashboard/docs/SCHEMA.md`](../dashboard/docs/SCHEMA.md) and can be visualised with the [`dashboard/`](../dashboard/) app.

## Notes & Findings

See [`logbook.md`](logbook.md) for running notes on implementation decisions and bugs.

Notable findings:
- Gradient norm clipping is critical for PPO stability on continuous control
- State normalization (Welford) fixes cross-environment instabilities but can cause gradient explosions — clip the value function gradients separately
- JAX on Apple Silicon currently underperforms MLX due to CPU↔device sync overhead; `jax-mps` is still experimental
- The `cumsum / powers` trick is a clean way to vectorize discounted return computation
