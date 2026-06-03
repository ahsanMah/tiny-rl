from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from diffusion import ema_update, sample_batch
from logger_utils import RLLogger
from unet import WaveletDownsampleConv, WaveletUpsample, print_param_table

ReconLoss = Literal["l1", "l2"]


@dataclass(frozen=True)
class VAEModelConfig:
    latent_channels: int = 4
    base_channels: int = 32
    num_downsamples: int = 2
    use_wavelet: bool = True


@dataclass(frozen=True)
class VAETrainConfig:
    vae_train_steps: int = 1_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    kl_weight: float = 1e-6
    recon_loss: ReconLoss = "l1"
    ema_decay: float = 0.999
    save_dir: str | None = None
    log_every: int = 50
    log_tensorboard: bool = False


class ConvResBlock2D(nn.Module):
    """Per-frame 2D residual block (Conv2d analogue of ConvResBlock3D, no time-FiLM)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.norm1 = nn.RMSNorm(out_channels)
        self.norm2 = nn.RMSNorm(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        zero_init = nn.init.constant(0.0)
        self.conv2.weight = zero_init(self.conv2.weight)
        self.conv2.bias = zero_init(self.conv2.bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.shortcut(x)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + x


class WaveletVAE(nn.Module):
    """Per-frame 2D KL VAE with Haar wavelet pooling as first/last layer.

    Operates on clips ``(B, T, H, W, C)``; collapses ``(B, T) -> B*T`` for the
    2D convs, exactly as ``WaveletDownsampleConv`` does internally.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        num_downsamples: int = 2,
        use_wavelet: bool = True,
        latent_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_downsamples = num_downsamples
        self.use_wavelet = use_wavelet
        self.latent_scale = latent_scale

        zero_init = nn.init.constant(0.0)

        enc_in = in_channels
        if use_wavelet:
            self.prepool = WaveletDownsampleConv(in_channels)
            self.unpool = WaveletUpsample()
            enc_in = in_channels * 4
            dec_out = out_channels * 4
        else:
            dec_out = out_channels

        # Encoder: stem -> [resblock + strided downsample] x num_downsamples -> mean/logvar.
        self.enc_stem = nn.Conv2d(enc_in, base_channels, kernel_size=3, padding=1)
        self.enc_blocks = []
        self.enc_downs = []
        ch = base_channels
        for _ in range(num_downsamples):
            out_ch = ch * 2
            self.enc_blocks.append(ConvResBlock2D(ch, out_ch))
            self.enc_downs.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            ch = out_ch
        self.enc_mid = ConvResBlock2D(ch, ch)
        self.to_moments = nn.Conv2d(ch, latent_channels * 2, kernel_size=1)
        self.to_moments.weight = zero_init(self.to_moments.weight)
        self.to_moments.bias = zero_init(self.to_moments.bias)

        # Decoder: mirror (nearest upsample + conv).
        self.from_latent = nn.Conv2d(latent_channels, ch, kernel_size=3, padding=1)
        self.dec_mid = ConvResBlock2D(ch, ch)
        self.dec_ups = []
        self.dec_blocks = []
        for _ in range(num_downsamples):
            out_ch = ch // 2
            self.dec_ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(ConvResBlock2D(ch, out_ch))
            ch = out_ch
        self.dec_out = nn.Conv2d(ch, dec_out, kernel_size=3, padding=1)
        self.dec_out.weight = zero_init(self.dec_out.weight)
        self.dec_out.bias = zero_init(self.dec_out.bias)

    def encode(self, frames: mx.array) -> tuple[mx.array, mx.array]:
        """``(B, T, H, W, C)`` -> ``(mean, logvar)`` each ``(B, T, h, w, latent_channels)``."""
        B, T = frames.shape[0], frames.shape[1]
        if self.use_wavelet:
            frames = self.prepool(frames)
        x = frames.reshape(B * T, *frames.shape[2:])
        h = self.enc_stem(x)
        for block, down in zip(self.enc_blocks, self.enc_downs):
            h = down(block(h))
        h = self.enc_mid(h)
        moments = self.to_moments(h)
        mean, logvar = moments[..., : self.latent_channels], moments[
            ..., self.latent_channels :
        ]
        logvar = mx.clip(logvar, -30.0, 20.0)
        mean = mean.reshape(B, T, *mean.shape[1:])
        logvar = logvar.reshape(B, T, *logvar.shape[1:])
        return mean, logvar

    def reparameterize(self, mean: mx.array, logvar: mx.array) -> mx.array:
        eps = mx.random.normal(mean.shape)
        return mean + mx.exp(0.5 * logvar) * eps

    def decode(self, z: mx.array) -> mx.array:
        """``(B, T, h, w, latent_channels)`` -> ``(B, T, H, W, C)``."""
        B, T = z.shape[0], z.shape[1]
        x = z.reshape(B * T, *z.shape[2:])
        h = self.from_latent(x)
        h = self.dec_mid(h)
        for up, block in zip(self.dec_ups, self.dec_blocks):
            h = block(up(h))
        out = self.dec_out(h)
        out = out.reshape(B, T, *out.shape[1:])
        if self.use_wavelet:
            out = self.unpool(out)
        return out

    def __call__(self, frames: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        mean, logvar = self.encode(frames)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


def _clone_vae(model_config_dict: dict, original_parameters) -> WaveletVAE:
    clone = WaveletVAE(**model_config_dict)
    clone.update(tree_map(lambda x: x * 1.0, original_parameters))
    return clone


def kl_divergence(mean: mx.array, logvar: mx.array) -> mx.array:
    return -0.5 * mx.mean(1.0 + logvar - mean**2 - mx.exp(logvar))


class VAETrainer:
    """Template: FlowMatchingTrainer. AdamW + grad-clip + EMA + compiled step."""

    def __init__(
        self,
        model: WaveletVAE,
        ema_model: WaveletVAE,
        *,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 2.0,
        ema_decay: float = 0.999,
        kl_weight: float = 1e-6,
        recon_loss: ReconLoss = "l1",
    ):
        self.model = model
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        train_state = [
            self.model.state,
            self.optimizer.state,
            self.ema_model.state,
            mx.random.state,
        ]
        self.compiled_train_step = mx.compile(
            lambda batch: self.train_step(batch),
            inputs=train_state,
            outputs=train_state,
        )

    def _recon(self, recon: mx.array, x: mx.array) -> mx.array:
        if self.recon_loss == "l2":
            return mx.mean((recon - x) ** 2)
        return mx.mean(mx.abs(recon - x))

    def loss(self, batch: mx.array) -> mx.array:
        recon, mean, logvar = self.model(batch)
        recon_loss = self._recon(recon, batch)
        kl = kl_divergence(mean, logvar)
        return recon_loss + self.kl_weight * kl

    def train_step(self, batch: mx.array) -> mx.array:
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss)
        loss, grads = loss_and_grad_fn(batch)
        clipped_grads, _ = optim.clip_grad_norm(grads, max_norm=self.max_grad_norm)
        self.optimizer.update(self.model, clipped_grads)
        ema_update(self.ema_model, self.model, self.ema_decay)
        return loss


def encode_clips(vae: WaveletVAE, clips: mx.array) -> mx.array:
    """Per-frame encode ``(B, T, H, W, C)`` -> ``(B, T, h, w, latent_channels)``.

    Uses the posterior mean, scaled to ~unit variance by ``latent_scale``.
    """
    mean, _ = vae.encode(clips)
    return mean / vae.latent_scale


def decode_latents(vae: WaveletVAE, latents: mx.array) -> mx.array:
    """Per-frame decode ``(B, T, h, w, latent_channels)`` -> ``(B, T, H, W, C)``."""
    return vae.decode(latents * vae.latent_scale)


def save_vae(
    vae: WaveletVAE,
    save_dir: str | Path,
    *,
    config: dict,
    ema_model: WaveletVAE | None = None,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vae.save_weights(str(save_dir / "model.safetensors"))
    if ema_model is not None:
        ema_model.save_weights(str(save_dir / "ema_model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_vae(save_dir: str | Path, *, prefer_ema: bool = True) -> WaveletVAE:
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    config = {k: v for k, v in config.items() if k != "step"}
    vae = WaveletVAE(**config)

    model_path = save_dir / "model.safetensors"
    ema_path = save_dir / "ema_model.safetensors"
    weights_path = ema_path if prefer_ema and ema_path.exists() else model_path
    vae.load_weights(str(weights_path))
    return vae


def _calibrate_latent_scale(vae: WaveletVAE, frames: mx.array) -> float:
    mean, _ = vae.encode(frames)
    return float(mx.std(mean))


def train_vae_on_dataset(
    frames: mx.array,
    model_config: VAEModelConfig | None = None,
    train_config: VAETrainConfig | None = None,
    sample_fps: float = 2.0,
):
    """Train the VAE on clips ``(B, T, H, W, C)``.

    Returns ``(model, ema_model, full_model_config)`` where the config carries
    the calibrated ``latent_scale``.
    """
    model_config = VAEModelConfig() if model_config is None else model_config
    train_config = VAETrainConfig() if train_config is None else train_config

    train_logger = None
    if train_config.log_tensorboard and train_config.save_dir is not None:
        save_path = Path(train_config.save_dir)
        train_logger = RLLogger(log_dir=str(save_path.parent), exp_name=save_path.name)

    print("Using VAE train config:")
    pprint(train_config)
    print("Using VAE model config:")
    pprint(model_config)

    dataset_size = int(frames.shape[0])
    if dataset_size < 2:
        raise ValueError(f"Need at least 2 clips, got {dataset_size}")
    val_size = max(1, int(round(dataset_size * 0.05)))
    val_size = min(val_size, dataset_size - 1)
    train_size = dataset_size - val_size
    train_clips = frames[:train_size]
    val_clips = frames[train_size:]

    input_config = {
        "in_channels": int(frames.shape[-1]),
        "out_channels": int(frames.shape[-1]),
    }
    full_model_config = {**input_config, **asdict(model_config)}

    model = WaveletVAE(**full_model_config)
    ema_model = _clone_vae(full_model_config, model.parameters())
    trainer = VAETrainer(
        model,
        ema_model,
        learning_rate=train_config.learning_rate,
        ema_decay=train_config.ema_decay,
        kl_weight=train_config.kl_weight,
        recon_loss=train_config.recon_loss,
    )

    print("dataset:", tuple(frames.shape))
    print("train split:", tuple(train_clips.shape))
    print("val split:", tuple(val_clips.shape))
    print_param_table(model)

    log10_max = 20.0 * float(mx.log10(mx.array(2.0)))
    avg_loss = 0.0
    start = time.time()
    dummy_actions = mx.zeros((dataset_size, 1), dtype=mx.int8)
    for step in range(1, train_config.vae_train_steps + 1):
        batch, _ = sample_batch(
            train_clips, dummy_actions[:train_size], train_config.batch_size
        )
        loss = float(trainer.compiled_train_step(batch))
        avg_loss += loss

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.vae_train_steps
        ):
            elapsed = time.time() - start
            avg_loss /= train_config.log_every
            steps_per_sec = step / max(elapsed, 1e-8)

            val_batch, _ = sample_batch(
                val_clips,
                dummy_actions[train_size:],
                min(train_config.batch_size, int(val_clips.shape[0])),
            )
            recon, mean, logvar = model(val_batch)
            recon_mse = mx.mean((recon - val_batch) ** 2)
            psnr = log10_max - 10.0 * mx.log10(mx.maximum(recon_mse, 1e-12))
            kl = kl_divergence(mean, logvar)
            psnr_f, kl_f = float(psnr), float(kl)

            print(
                f"step={step:5d} steps/s={steps_per_sec:.2f} "
                f"loss={loss:.4f} avg={avg_loss:.4f} "
                f"val_psnr={psnr_f:.2f}dB val_kl={kl_f:.4f}"
            )
            if train_logger is not None:
                train_logger.log_train_metrics(
                    step,
                    {
                        "loss": loss,
                        "avg_loss": avg_loss,
                        "val_psnr": psnr_f,
                        "val_kl": kl_f,
                        "steps_per_second": steps_per_sec,
                    },
                )
                train_logger.log_reconstructions(
                    step,
                    val_batch[:, -1],
                    {0.0: recon[:, -1]},
                )
            avg_loss = 0.0
            mx.clear_cache()

    # Latent-scale calibration on a sample batch (uses EMA weights).
    calib_batch, _ = sample_batch(
        train_clips, dummy_actions[:train_size], train_config.batch_size
    )
    latent_scale = _calibrate_latent_scale(ema_model, calib_batch)
    latent_scale = latent_scale if latent_scale > 1e-6 else 1.0
    print(f"calibrated latent_scale={latent_scale:.4f}")
    full_model_config["latent_scale"] = latent_scale
    model.latent_scale = latent_scale
    ema_model.latent_scale = latent_scale

    return model, ema_model, full_model_config


def _self_test() -> None:
    mx.random.seed(0)
    frames = mx.random.uniform(-1.0, 1.0, shape=(2, 4, 32, 32, 3))
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=8,
        num_downsamples=1,
        use_wavelet=True,
    )
    model = WaveletVAE(**cfg)
    ema_model = _clone_vae(cfg, model.parameters())
    trainer = VAETrainer(model, ema_model, learning_rate=3e-3, kl_weight=1e-6)

    recon, _, _ = model(frames)
    assert recon.shape == frames.shape, (recon.shape, frames.shape)
    mse_start = float(mx.mean((recon - frames) ** 2))
    for step in range(500):
        loss = float(trainer.compiled_train_step(frames))
    recon, _, _ = model(frames)
    mse_end = float(mx.mean((recon - frames) ** 2))
    print(f"recon MSE: start={mse_start:.5f} end={mse_end:.5f}")
    assert mse_end < mse_start, "recon MSE did not drop"
    print("self-test passed: recon MSE dropped.")


if __name__ == "__main__":
    _self_test()
