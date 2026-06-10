# tiny-rl

A personal reinforcement learning playground — algorithms implemented from scratch, a browser-based experiment tracker, and a couple of custom environments. Built on Apple MLX and JAX, targeting Mac-native acceleration.

## Repository Layout

```
tiny-rl/
├── gym-practice/      # Core RL algorithms + unified training CLI
├── dashboard/         # Browser-based experiment tracker
├── market-agent/      # Gymnasium-style market trading environment
└── mini-dreamer/      # Diffusion-based world model (model-based RL)
```

---

## gym-practice

The main sandbox. Each algorithm is self-contained in `algorithms/`, with a shared CLI runner in `main.py`.

| File | Algorithm |
|---|---|
| `vanilla_policy_grad.py` | Policy Gradient (full returns, no baseline) |
| `reinforce.py` | REINFORCE with reward-to-go + value baseline |
| `reinforce_gae.py` | REINFORCE + Generalized Advantage Estimation |
| `vectorized_gae.py` | Vectorized GAE across parallel environments |
| `ppo.py` | PPO with clipped surrogate objective (MLX) |
| `ppo_jax.py` | PPO port for JAX/Flax — GPU/TPU scaling experiments |
| `sac.py` | Soft Actor-Critic with EMA target network (MLX) |

```bash
cd gym-practice
uv run python main.py --algorithm ppo --env-name BipedalWalker-v3
```

See [`gym-practice/README.md`](gym-practice/README.md) for the full hyperparameter reference and [`gym-practice/logbook.md`](gym-practice/logbook.md) for implementation notes.

---

## dashboard

A zero-build browser app for inspecting training runs. Reads structured JSON artifacts written by `gym-practice`.

- Live training curves, episode return strips, and checkpoint browser
- Best / median / worst rollout videos per checkpoint
- Six themes (dark, light, and high/medium contrast variants)

Artifact schema: [`dashboard/docs/SCHEMA.md`](dashboard/docs/SCHEMA.md)

### Installation

```bash
cd dashboard && npm install
```


### Starting the dashboard

```bash
cd dashboard && npm start
# → http://localhost:8080/index.html
```

[browser-sync](https://browsersync.io/) serves the files and hot-reloads `hifi/**/*` on change.

### Run index

The dashboard uses runs via `runs/index.json` containing the run-directory names:

```json
[
    "BipedalWalker-v3-ppo_2026-05-28_16-14",
    "BipedalWalker-v3-sac_1779998141"
 ]
```

`index.json` is **rebuilt automatically** every time a `DashboardRunWriter` is created or closed in `gym-practice/logger_utils.py`. No manual editing needed; new runs appear in the sidebar after a browser refresh.

---

## market-agent

A Gymnasium-compatible environment for trading on time-series price data. Supports buy / hold / sell actions over a sliding observation window, with PnL-based rewards. Ingests any HuggingFace `datasets.Dataset`.
 
---

## mini-dreamer

Experiments toward a diffusion-based world model for model-based RL. Trains a video diffusion U-Net on MiniGrid rollouts, then uses the learned model for imagination-based planning.

> Work in progress available on branch mini-dreamer-workspace
---

## Requirements

- Python ≥ 3.12, managed with [uv](https://github.com/astral-sh/uv)
- Node.js + npm (dashboard only)
- macOS recommended — MLX is Apple Silicon-native
>>>>>>> main
