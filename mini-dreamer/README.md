# mini-dreamer

This is my attempt to build a model-based RL agent the way Ha & Schmidhuber's *World Models* and the *Dreamer* line of work do it — but with the recurrent/VAE world model swapped out for a small **video diffusion model**. The plan is to learn to *dream* an environment frame-by-frame, conditioned on the actions an agent takes, and eventually train a policy inside that dream.

The "mini" is doing a lot of work in that sentence. Everything here is written from scratch in [MLX](https://github.com/ml-explore/mlx), runs on a single M4 Macbook, and is small enough to actually read in an afternoon. The goal is understanding, not a leaderboard.

## Current progress

**Done - the world model:** we can roll out a Gymnasium environment with random actions, train a diffusion model to predict the next frame from a short window of past frames plus the action taken, and then autoregressively *generate* a video by feeding the model its own predictions. That whole loop works and is what you'll find in `diffusion.py`, `unet.py`, and `data.py`.

**Not done yet - the dream:** the actual RL part (rolling out a real policy, updating world model and policy together, then continuing to train *only* inside the generated dream) is still on the [roadmap](#roadmap) below. So today this is really a **conditional video predictor** — a necessary first brick, not the finished house.

## Core idea

A world model has one job: given what just happened and what the agent decides to do, predict what happens next. If it can do that well enough, an agent can plan (or train) against the model's imagination instead of paying for expensive real-environment steps.

So how should we model "predict the next frame"? We treat next-frame prediction as a **conditional generation** problem and let a diffusion model handle it.

### Flow matching

The model learns a *velocity field* that transports a pure-noise frame to a real frame along a straight line. Training is just a regression — sample a time `t ∈ (0, 1)`, form the interpolated frame `xt = (1 - t)·noise + t·x`, and ask the model to predict the velocity `x - noise`. At sampling time we start from noise and integrate that field forward with a plain Euler stepper (`sample_euler` in `diffusion.py`).

A couple of choices worth flagging, since they're easy to miss:

- **We only supervise the last frame.** The conditioning frames are passed in clean and a mask (`make_final_frame_mask`) zeroes out the loss everywhere except the final frame. The model isn't denoising the whole clip — it's denoising *one new frame* while reading the rest as context.

- **Timesteps are sampled logit-normally, not uniformly.** Drawing `t` from a logit-normal distribution concentrates training signal in the middle of the trajectory, where there's the most to learn. (`sample_distribution = "uniform" | "normal" | "logitnorm"` if you want to compare.)

### The model: a 3D U-Net that reads space *and* time

The network (`UNet3D`) is a fairly standard U-Net with 3D convolutions so it can see across the temporal axis of a clip, plus a few things bolted on:

- **Time conditioning via FiLM / AdaLN** The diffusion timestep gets a Gaussian-Fourier embedding and is injected into every residual block as a feature-wise scale and shift.
- **Action conditioning.** This is what makes it a *world* model and not just a video model. Each action is embedded (a categorical lookup for discrete envs, a linear projection for continuous ones) and mixed into the conditioning signal, so the same context frames can produce different futures depending on what the agent does.
- **Transformer bottleneck.** At the lowest spatial resolution - where global context is cheap to aggregate - sit a stack of Transformer blocks (self-attention, then cross-attention onto the action context). Attention is multi-query to keep it light.
- **Haar wavelet (de)pooling.** With `use_wavelet = true`, frames are losslessly split into four wavelet subbands before the U-Net and recombined after. It's a cheap way to halve the spatial resolution the convolutions have to chew through without throwing information away. Note: this is still experimental and I'm not yet certain it's a clear win on every environment.

We also keep an **EMA copy** of the weights, which tends to generalize noticeably better at sampling time, and clip gradients (`max_grad_norm`) because early training was prone to the occasional explosion.

## Setup

You'll need Macs since MLX is the whole backend but it may run on CPU only. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

That pulls in MLX, Gymnasium (with Box2D), MiniGrid, and ViZDoom. ViZDoom is installed straight from the Farama GitHub, so the first sync may take a minute.

## Usage

Everything runs through `pretrainer.py`, which is a small `click` CLI with two subcommands — `train` and `generate` — both driven by structured TOML configs.

### Train a world model

```bash
uv run python pretrainer.py train --config experiments/minigrid.toml
```

Any field in the config can be overridden on the command line, which is handy for quick sweeps:

```bash
uv run python pretrainer.py train --config experiments/minigrid.toml --train-steps 2000
```

Training collects rollouts, slices them into clips that never cross episode boundaries, holds out ~5% for validation, and reports per-timestep **loss, PSNR, and R²** as it goes. If `log_tensorboard = true`, it also logs reconstructions (ground-truth frame vs. one-step prediction at several timesteps) so failure modes are visually obvious — staring at a single loss number rarely tells you *why* something is broken.

### Dream a video from a trained model

```bash
uv run python pretrainer.py generate --config experiments/minigrid.toml
```

This seeds the model with a few real frames, then autoregressively rolls forward — each new frame is fed back in as context for the next. You get two artifacts: an annotated preview (frames labeled with the action taken) and a `denoising.mp4` showing the Euler trajectory of a single generated frame evolving out of noise.

### Config sections

A config is grouped into five tables, each mapping to a dataclass that is the single source of truth for its schema:

| Section | What it controls |
| --- | --- |
| `[env]` | which Gymnasium environment (`env_id`) |
| `[dataset]` | rollout length, clip length/stride, tile size, preview settings |
| `[model]` | `base_channels`, transformer block count, `max_context_size`, `use_wavelet` |
| `[train]` | steps, batch size, learning rate, timestep distribution, logging |
| `[generate]` | which checkpoint to load, how many frames/steps to dream |

`experiments/minigrid.toml` is the minimal example to start from; `carracing-v1.toml`, `lunar.toml`, and `vizdoom.toml` are the other environments I've been poking at.

## Supported environments

The data pipeline (`data.py`) currently knows how to roll out three families:

- **MiniGrid** — tile-rendered RGB grid worlds (`MiniGrid-Empty-8x8-v0` is the default sanity check).
- **Box2D** — `CarRacing-v3`, `LunarLander`, etc., with discrete actions and optional frame-skip.
- **ViZDoom** — first-person Doom frames.

All rollouts use **random actions** for now. That's deliberate at this stage — we want the world model to see a broad slice of the dynamics before any policy starts narrowing the distribution — but it's also a real limitation worth keeping in mind (see below).

## Roadmap

The intended path, roughly following *World Models*:

1. Roll out trajectories from the **true** environment using an actual policy (not random actions).
2. Update **both** the policy and the world model on those real transitions.
3. Continue training the policy **only inside the dream** — the generated rollouts — and check whether it transfers back.

There's also a standing experiment in the backlog to train a wavelet VAE (KL + L1/L2/LPIPS) as a learned latent space, which would let the diffusion model work in a compressed latent rather than pixel/wavelet space.

## Assessment & limitations

A few things I'd want a reader to know before taking any of this too seriously:

- **It's a predictor, not yet an agent.** No policy is being trained against the dream today.
- **Random-action data is a real constraint.** A world model only learns the dynamics it's shown; rare-but-important states (a crash, reaching the goal) are undersampled, which is an inherent limitation of behavior-agnostic data collection, not a bug I can patch away.
- **Autoregressive drift.** Like any model fed its own predictions, errors compound over long rollouts. Short context windows (`max_context_size`) and modest generation lengths keep this manageable but don't solve it. This might be possibly solved via [Diffusion Forcing](https://www.boyuan.space/diffusion-forcing/)
- **Apple Silicon only.** MLX is baked in, it might be possible to run on CUDA but I have not tested yet.

## Acknowledgements

Standing on the shoulders of [World Models (Ha & Schmidhuber)](https://worldmodels.github.io/), the [Dreamer](https://arxiv.org/abs/1912.01603) line of work, and the flow-matching literature. Built on [MLX](https://github.com/ml-explore/mlx) and [Gymnasium](https://gymnasium.farama.org/).
