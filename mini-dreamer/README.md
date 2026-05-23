# mini-dreamer

The goal of this project is to train model-based RL algorithms using a diffusion based world model. We will be using a video diffusion model and train it to learn minigrid environments.

## Experiment configs

`pretrainer.py` supports structured TOML configs for `train` and `generate`.

Example:

```bash
uv run python pretrainer.py train --config experiments/minigrid.toml
uv run python pretrainer.py train --config experiments/minigrid.toml --train-steps 2000
```

Config sections:

- `[env]`
- `[train]`
- `[preview]`
- `[generate]`

See `experiments/minigrid.toml` for a minimal example.
