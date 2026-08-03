
# to train vae
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.98 JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache" uv run \
python pretrainer.py train-vae \
--config experiments/vizdoom-latent-finetune.toml --clip-length 2 \
--load-dir logs/vizdoom-vae-v3
```

XLA_PYTHON_CLIENT_MEM_FRACTION=.98 JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache" uv run \
python pretrainer.py train-vae \
--config experiments/vizdoom-latent.toml --clip-length 2 \
--rollout-steps 500 --batch-size 4


# to run generationss
#  uv run python pretrainer.py generate --config experiments/vizdoom-latent.toml --rollout-steps 500 --generate-new-frames 6 --warmup-steps 50

### Transfer Rclone config
# scp ~/.config/rclone/rclone.conf user@remote:~/.config/rclone/
# scp -P 45906 ~/.config/rclone/rclone.conf root@213.173.104.21:~/.config/rclone/rclone.conf

### to gdrive
#  rclone copy logs/vizdoom-vae-v2 gdrive:/data --progress

## from gdrive
# rclone copy gdrive:/data/vizdoom-vae-v2 ./vizdoom-vae-v2  --progress
# rclone sync vizdoom-vae-jax/ gdrive:/data/logs/vizdoom-vae-v3 --exclude *.npy --progress

# bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh && . \$HOME/.local/bin/env && if [ ! -f '/workspace/.setup_done' ]; then [ ! -d 'tiny-rl' ] && git clone https://github.com/ahsanMah/tiny-rl.git; cd tiny-rl/mini-dreamer && git checkout jax-dreams && DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install cmake git libboost-all-dev libsdl2-dev libopenal-dev && uv add 'jax[cuda]' && uv sync && touch /workspace/.setup_done; else cd tiny-rl/mini-dreamer && git pull; fi"
# python -c "import numpy; print('OK', numpy.__version__)"
#
# import gymnasium as gym
# from vizdoom import gymnasium_wrapper
# env = gym.make("VizdoomBasic-v1", continuous=False, )
