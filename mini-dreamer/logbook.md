# Logbook

---

## 06/03 - Episode-Safe Rollouts, Doom Support, and New Experiment Configs

Rollout handling now records episode boundaries and slices clips per-episode so training data no longer
crosses reset transitions. This fixes the earlier issue where clips could straddle terminal frames and
mix episodes.

Environment/data utilities were consolidated:

- Added Vizdoom support with a Doom rollout path, plus a `make_env` helper that selects MiniGrid vs.
  Box2D vs. Vizdoom environments.
- Generation can optionally draw new actions from a fixed pool to debug action-conditioning
  behavior.
- The pretrainer now uses the shared dataset helpers for both training and generation.

Experiment configuration was expanded and refreshed:

- New TOML configs were added for CarRacing and Vizdoom runs.
- The MiniGrid config was updated for longer clips, wavelet-enabled models, and new v1 log
  directories with shorter training schedules.
- Minor cleanup removed unused diffusion code and refreshed the UNet benchmarking harness for
  wavelet/non-wavelet comparisons.

## 05/29

- Debugging why thew training is slow led to avery interesting finding
```bash
sampling step 8 took 0.02ms
train step 8 took 0.73ms
loss append 8 took 181.60ms
ema update 8 took 0.91ms
```

The naive loss append is 181ms! Its almost a 100x slower than all functions related to training combined! But thats because the graph execution had not hit until we did a `float(loss)` (which is implcitly an `mx.eval`) so we were in fact measuring the train + sampling execution.


## 05/27 - Dataclass-First CLI, Denoising MP4 Exports, and Better Validation Signals

The 05/27 work focused on making experiment configuration and training diagnostics much harder to
break.

CLI/config handling was refactored so dataclasses are now the source of truth for option names and
TOML schema validation. Instead of manually merging config values and maintaining separate per-command
maps, `--config` now feeds `click` via `ctx.default_map`, and command options are derived directly
from config dataclasses. This removed a lot of duplicate plumbing and fixed an important drift bug:
`num_transformer_blocks` existed in TOML/model config but was missing from the CLI schema path, which
would fail config-driven runs.

Training/eval logging was expanded beyond loss-only reporting:

- Per-timestep validation now reports **loss, PSNR, and R²**.
- TensorBoard logging can be enabled from training config.
- Reconstruction images (`x1` ground truth vs. one-step predictions at each timestep) are logged to
  make failure modes visually obvious.
- Logger run naming and tag conventions were cleaned up for easier comparison across runs.

Sampling and artifact export were also improved:

- Euler sampling can now return intermediate denoising states.
- A new MP4 export path writes denoising trajectories (context frames + generated frame evolution)
  for inspection.
- Sample preview FPS was re-threaded explicitly after the config refactor so preview timing remains
  controllable from CLI/config.

A smaller cleanup removed `max_clips` limits from dataset slicing, simplifying data loading and
avoiding accidental underuse of collected rollouts.

---

## 05/23 - Transformer Bottleneck, Logit-Norm Sampling, and Config Refactor

The UNet3D received a proper bottleneck Transformer block between the encoder and decoder, replacing
the earlier ad-hoc attention experiments. The block is now configurable (number of heads, layers,
hidden dim) and sits cleanly at the lowest spatial resolution, which is the appropriate place for
global context aggregation.

Logit-norm sampling was added to the flow-matching trainer. Rather than sampling diffusion timesteps
uniformly, timesteps are drawn from a logit-normal distribution, which tends to concentrate training
signal near the middle of the flow trajectory where the model has the most to learn.

The experiment configuration system was significantly refactored. Config sections are now grouped
more coherently (model, training, data, generation), the TOML files were updated accordingly, and
the pretrainer CLI was reorganized to load all hyperparameters from config rather than relying on a
mix of CLI flags and hardcoded defaults. This makes experiments more reproducible and easier to
compare.

OpenCV was removed as a dependency for video loading inside the logger; `imageio` was already in use
for the main pipeline and handles the same task with fewer installation complications.

---

## 05/22 - Training Diagnostics, EMA, and Data Pipeline Improvements

Several training diagnostics were added to make it easier to detect overfitting and understand model
behavior during training:

- The dataset is now split into train and validation sets (approximately 95/5).
- Validation loss is reported per diffusion timestep at the end of each epoch, using four evenly
  spaced points across the flow trajectory. This gives a clearer picture of where the model
  struggles (early vs. late denoising).
- EMA (exponential moving average) weights are now tracked during training and saved alongside the
  base checkpoint. EMA weights tend to generalize better at inference time.

Rolling windows were introduced for extracting training clips from longer recordings. Previously
each video contributed a fixed number of non-overlapping clips; rolling windows extract more
training examples per video and expose the model to more temporal context boundaries.

These changes collectively addressed the todo list that had been accumulating: train/val split,
per-timestep loss reporting, and EMA saving were all completed in this session.

---

## 05/21 - Gradient Clipping, Attention Fixes, and Action Conditioning

A significant round of fixes and architectural changes landed together:

- Gradient clipping (`max_grad_norm`) was added to `FlowMatchingTrainer`. Without it, training was
  prone to occasional gradient explosions, particularly early in training when the model is
  producing large prediction errors.
- The attention implementation in the UNet was rewritten: `q`/`k`/`v` projection shapes were
  incorrect, scaling was missing, and certain weight/bias initializations were not zeroed out as
  intended. These were silent bugs - the model was training but the attention layers were not
  functioning correctly.
- RMSNorm replaced the earlier normalization scheme in the attention path.
- Self-attention and cross-attention are now separated, with cross-attention accepting an external
  context tensor. This is the precursor to conditioning on text or other modalities.
- Action conditioning was wired through the UNet. The number of discrete actions (`num_env_actions`)
  is now a required parameter of the trainer and is embedded and injected into the model alongside
  the time embedding.
- Spatial and temporal attention paths were separated experimentally to give the model distinct
  inductive biases for each dimension.
- A misconfiguration bug in `pretrainer.py` (a wrong config key lookup) was patched.

---

## 05/20 - Data Loading, MiniGrid Environment, and Pretrainer Refactor

The project was wired up to the MiniGrid environment. A dedicated `minigrid_data.py` handles
collecting rollouts and storing video clips with associated action sequences.

The pretrainer CLI was refactored from a flat argparse interface to a `click`-based interface with
`train` and `generate` subcommands. Model save/load utilities were added so that checkpoints can be
resumed and generation can be run independently of training.

Saved video previews are now annotated with the action taken at each frame, which makes it much
easier to visually verify that action conditioning is having the expected effect.

The optimizer was switched from Adam to AdamW. AdamW decouples weight decay from the gradient
update, which is generally preferable for training diffusion/flow models.

On the attention side, two experiments were run in quick succession:

- **Multi-Query Attention (MQA)**: keys and values are shared across heads, reducing memory and
  compute. Switched to this from standard multi-head attention.
- **Average pooling**: replaced a learned pooling operation. MQA was kept but the pooling
  mechanism was adjusted.

The flow-matching loss was also masked so that it only supervises the final video frame during
generation rollouts, rather than all frames. This focuses the model on the prediction target.

---

## 05/19 - Initial Mini-Dreamer Implementation

The project was initialized with a 3D UNet designed to operate on video clips (spatial + temporal
dimensions). The initial architecture used residual blocks with FiLM (Feature-wise Linear
Modulation) conditioning for injecting the diffusion timestep into intermediate feature maps.
FiLM time embeddings were added immediately after the skeleton was in place.

A minimal flow-matching trainer was implemented alongside the UNet. Flow matching is used in place
of standard DDPM-style diffusion: the model learns to predict a vector field that transports noise
to data along straight paths. An Euler sampler was implemented for inference.

`imageio` was chosen for video loading from the start, replacing an earlier OpenCV-based prototype.
A clip preview export was added to visually verify that the data pipeline was loading and
preprocessing frames correctly before training began.

Video downsampling is configurable so that resolution can be tuned independently of the model size.
