import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import lax
from safetensors.flax import load_file, save_file

from data import Dataset
from logger_utils import RLLogger
from unet import format_param_table

ReconLoss = Literal["l1", "l2", "l1+l2"]


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
    lr_warmup_steps: int = 0
    lr_hold_steps: int = 0
    lr_final: float | None = None
    kl_weight: float = 1e-6
    recon_loss: ReconLoss = "l1"
    wavelet_loss: bool = False
    detail_weight: float = 0.0
    ema_decay: float = 0.99
    save_dir: str | None = None
    log_every: int = 50
    log_tensorboard: bool = False


class WaveletDownsampleConv:
    """Conv2d-based 2D Haar DWT. Grouped depthwise 2×2 strided conv variant of WaveletDownsample."""

    def __init__(self, in_channels: int):
        base = jnp.array(
            [
                [[0.25, 0.25], [0.25, 0.25]],  # LL
                [[0.25, -0.25], [0.25, -0.25]],  # LH
                [[0.25, 0.25], [-0.25, -0.25]],  # HL
                [[0.25, -0.25], [-0.25, 0.25]],  # HH
            ]
        )[:, :, :, None]  # (4, 2, 2, 1) — (O, H, W, I/groups)

        # (4*C, 2, 2, 1) — same 4 filters tiled for each input channel
        weight_oihw_like = jnp.concatenate([base] * in_channels, axis=0)

        # JAX 'HWIO' layout expects (H, W, I/groups, O)
        self.weight = jnp.transpose(weight_oihw_like, (1, 2, 3, 0))  # (2, 2, 1, 4*C)
        self._groups = in_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, H, W, C = x.shape
        x_ = x.reshape(B * T, H, W, C)

        out = lax.conv_general_dilated(
            x_,
            self.weight,
            window_strides=(2, 2),
            padding="VALID",
            feature_group_count=self._groups,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # conv outputs (C, 4) interleaved per spatial position; transpose to (4, C) to match WaveletUpsample
        out = (
            out.reshape(B * T, H // 2, W // 2, C, 4)
            .transpose(0, 1, 2, 4, 3)
            .reshape(B * T, H // 2, W // 2, C * 4)
        )
        return out.reshape(B, T, H // 2, W // 2, C * 4)


class WaveletUpsample:
    """Inverse 2D Haar DWT: (B, T, H/2, W/2, 4C) → (B, T, H, W, C)."""

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, Hh, Wh, C4 = x.shape
        C = C4 // 4
        x = x.reshape(B * T, Hh, Wh, C4)
        ll, lh, hl, hh = (
            x[..., :C],
            x[..., C : 2 * C],
            x[..., 2 * C : 3 * C],
            x[..., 3 * C :],
        )

        # Inverse along W: recover lo and hi rows
        lo = jnp.stack([ll + lh, ll - lh], axis=3).reshape(B * T, Hh, Wh * 2, C)
        hi = jnp.stack([hl + hh, hl - hh], axis=3).reshape(B * T, Hh, Wh * 2, C)

        # Inverse along H: recover original rows
        out = jnp.stack([lo + hi, lo - hi], axis=2).reshape(B * T, Hh * 2, Wh * 2, C)
        return out.reshape(B, T, Hh * 2, Wh * 2, C)


class ConvResBlock2D(nnx.Module):
    """Per-frame 2D residual block (Conv2d analogue of ConvResBlock3D, no time-FiLM)."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.shortcut = (
            nnx.identity
            if in_channels == out_channels
            else nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)
        )
        self.norm1 = nnx.RMSNorm(out_channels, rngs=rngs)
        self.norm2 = nnx.RMSNorm(out_channels, rngs=rngs)
        self.conv1 = nnx.Conv(
            out_channels, out_channels, kernel_size=(3, 3), padding=1, rngs=rngs
        )
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=1,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.act = nnx.silu

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.shortcut(x)
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + x


def _nearest_upsample_2d(x: jnp.ndarray, scale_factor: int = 2) -> jnp.ndarray:
    """NHWC nearest-neighbor upsample, matching mx.nn.Upsample(mode='nearest')."""
    B, H, W, C = x.shape
    return jax.image.resize(
        x, (B, H * scale_factor, W * scale_factor, C), method="nearest"
    )


class WaveletVAE(nnx.Module):
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
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_downsamples = num_downsamples
        self.use_wavelet = use_wavelet
        self.latent_scale = latent_scale

        zero_init = nnx.initializers.zeros_init()

        enc_in = in_channels
        if use_wavelet:
            self.prepool = WaveletDownsampleConv(in_channels)
            self.unpool = WaveletUpsample()
            enc_in = in_channels * 4
            dec_out = out_channels * 4
        else:
            dec_out = out_channels

        # Encoder: stem -> [resblock + strided downsample] x num_downsamples -> mean/logvar.
        self.enc_stem = nnx.Conv(
            enc_in, base_channels, kernel_size=(3, 3), padding=1, rngs=rngs
        )
        enc_blocks = []
        enc_downs = []
        ch = base_channels
        for _ in range(num_downsamples):
            out_ch = ch * 2
            enc_blocks.append(ConvResBlock2D(ch, out_ch, rngs=rngs))
            enc_downs.append(
                nnx.Conv(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding=1,
                    rngs=rngs,
                )
            )
            ch = out_ch
        self.enc_blocks = nnx.List(enc_blocks)
        self.enc_downs = nnx.List(enc_downs)
        self.enc_mid = ConvResBlock2D(ch, ch, rngs=rngs)
        self.to_moments = nnx.Conv(
            ch,
            latent_channels * 2,
            kernel_size=(1, 1),
            kernel_init=zero_init,
            bias_init=zero_init,
            rngs=rngs,
        )

        # Decoder: mirror (nearest upsample + conv).
        self.from_latent = nnx.Conv(
            latent_channels, ch, kernel_size=(3, 3), padding=1, rngs=rngs
        )
        self.dec_mid = ConvResBlock2D(ch, ch, rngs=rngs)
        dec_blocks = []
        for _ in range(num_downsamples):
            out_ch = ch // 2
            dec_blocks.append(ConvResBlock2D(ch, out_ch, rngs=rngs))
            ch = out_ch
        self.dec_blocks = nnx.List(dec_blocks)
        self.dec_out = nnx.Conv(
            ch,
            dec_out,
            kernel_size=(3, 3),
            padding=1,
            kernel_init=zero_init,
            bias_init=zero_init,
            rngs=rngs,
        )

    def encode(self, frames: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        mean, logvar = (
            moments[..., : self.latent_channels],
            moments[..., self.latent_channels :],
        )
        logvar = jnp.clip(logvar, -30.0, 20.0)
        mean = mean.reshape(B, T, *mean.shape[1:])
        logvar = logvar.reshape(B, T, *logvar.shape[1:])
        return mean, logvar

    def reparameterize(
        self, mean: jnp.ndarray, logvar: jnp.ndarray, *, rngs: nnx.Rngs
    ) -> jnp.ndarray:
        eps = jax.random.normal(rngs.reparam(), mean.shape)
        return mean + jnp.exp(0.5 * logvar) * eps

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """``(B, T, h, w, latent_channels)`` -> ``(B, T, H, W, C)``."""
        B, T = z.shape[0], z.shape[1]
        x = z.reshape(B * T, *z.shape[2:])
        h = self.from_latent(x)
        h = self.dec_mid(h)
        for block in self.dec_blocks:
            h = block(_nearest_upsample_2d(h, scale_factor=2))
        out = self.dec_out(h)
        out = out.reshape(B, T, *out.shape[1:])
        if self.use_wavelet:
            out = self.unpool(out)
        return out

    def __call__(
        self, frames: jnp.ndarray, *, rngs: nnx.Rngs
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, logvar = self.encode(frames)
        z = self.reparameterize(mean, logvar, rngs=rngs)
        recon = self.decode(z)
        return recon, mean, logvar


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


def kl_divergence(mean: jax.Array, logvar: jax.Array) -> jax.Array:
    return -0.5 * jnp.mean(1.0 + logvar - mean**2 - jnp.exp(logvar))


def linear_warmup_decay_schedule(
    peak_lr: float,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    hold_steps: int = 0,
    final_lr: float | None = None,
) -> float | optax.Schedule:
    """Linear warmup from ``peak_lr / 10`` to ``peak_lr`` over ``warmup_steps``,
    hold at ``peak_lr`` for ``hold_steps``, then linear decay to ``final_lr``
    over the remaining steps.

    Every part is optional: ``warmup_steps=0`` skips the warmup, ``hold_steps=0``
    starts the decay right after warmup, and ``final_lr=None`` holds the LR
    constant after warmup. With everything disabled this returns the plain
    ``peak_lr`` float (a fixed learning rate).
    """
    if warmup_steps <= 0 and final_lr is None:
        return peak_lr

    constant = optax.constant_schedule(peak_lr)
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


class VAETrainer(nnx.Module):
    """Template: FlowMatchingTrainer. AdamW + grad-clip + EMA + jitted step."""

    def __init__(
        self,
        model: WaveletVAE,
        ema_model: WaveletVAE,
        *,
        learning_rate: float | optax.Schedule = 3e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 2.0,
        ema_decay: float = 0.999,
        kl_weight: float = 1e-6,
        recon_loss: ReconLoss = "l1",
        wavelet_loss: bool = False,
        detail_weight: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.model = model
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss
        self.wavelet_loss = wavelet_loss
        self.detail_weight = detail_weight
        # Fixed Haar DWT for wavelet-domain recon loss (filters are constants,
        # not a trainable parameter — see WaveletDownsampleConv).
        self.dwt = WaveletDownsampleConv(
            model.out_channels
        )  # if wavelet_loss else None
        self.max_grad_norm = max_grad_norm
        self.rngs = rngs
        # Grad-clip + AdamW chain; nnx.jit on train_step plays the role of
        # mx.compile (model/optimizer/EMA/rng state is tracked automatically).
        self.optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
            ),
            wrt=nnx.Param,
        )

    def _recon(self, recon: jax.Array, x: jax.Array) -> jax.Array:
        match self.recon_loss:
            case "l2":
                return jnp.mean((recon - x) ** 2)
            case "l1":
                return jnp.mean(jnp.abs(recon - x))
            case "l1+l2":
                return jnp.mean(jnp.abs(recon - x)) + jnp.mean((recon - x) ** 2)

    def _wavelet_detail(self, recon: jax.Array, x: jax.Array) -> jax.Array:
        """Recon error on the single-level Haar high-frequency sub-bands (LH/HL/HH).

        Added on top of the pixel recon loss to up-weight high-frequency detail
        (e.g. small sharp objects). Excludes the LL band since the pixel loss
        already covers low frequencies. ``2.0 *`` rescales the 0.25-scaled filters
        to an orthonormal Haar so the coefficients carry true sub-band energy.
        """
        C = self.model.out_channels
        wr = 2.0 * self.dwt(recon)
        wt = 2.0 * self.dwt(x)
        return self._recon(wr[..., C:], wt[..., C:])

    def loss(self, model: WaveletVAE, batch: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        recon, mean, logvar = model(batch, rngs=rngs)
        recon_loss = self._recon(recon, batch)
        if self.wavelet_loss:
            recon_loss = recon_loss + self.detail_weight * self._wavelet_detail(
                recon, batch
            )
        kl = kl_divergence(mean, logvar)
        return recon_loss + self.kl_weight * kl

    def eval_loss(self, batch: jax.Array) -> dict[str, jax.Array]:
        metrics = {}
        recon, mean, logvar = self.model(batch, rngs=self.rngs)
        metrics["recon"] = recon
        metrics["l2_loss"] = jnp.mean((recon - batch) ** 2)
        metrics["l1_loss"] = jnp.mean(jnp.abs(recon - batch))
        metrics["kl_loss"] = kl_divergence(mean, logvar)
        metrics["wavelet_loss"] = self._wavelet_detail(recon, batch)
        return metrics

    @nnx.jit
    def train_step(self, batch: jax.Array) -> jax.Array:
        # Draw the reparam key up front; the Rngs is rebuilt inside the grad
        # trace since its counter cannot be mutated across trace levels.
        key = self.rngs.reparam()
        loss, grads = nnx.value_and_grad(
            lambda model: self.loss(model, batch, nnx.Rngs(reparam=key))
        )(self.model)
        self.optimizer.update(self.model, grads)
        ema_update(self.ema_model, self.model, self.ema_decay)
        return loss


def encode_clips(vae: WaveletVAE, clips: jax.Array) -> jax.Array:
    """Per-frame encode ``(B, T, H, W, C)`` -> ``(B, T, h, w, latent_channels)``.

    Uses the posterior mean, scaled to ~unit variance by ``latent_scale``.
    """
    mean, _ = vae.encode(clips)
    return mean / vae.latent_scale


def decode_latents(vae: WaveletVAE, latents: jax.Array) -> jax.Array:
    """Per-frame decode ``(B, T, h, w, latent_channels)`` -> ``(B, T, H, W, C)``."""
    return vae.decode(latents * vae.latent_scale)


def _flat_params(vae: WaveletVAE) -> dict[str, jax.Array]:
    """Flatten trainable params to ``{'enc_stem.kernel': array, ...}`` for safetensors."""
    return {
        ".".join(map(str, path)): variable.value
        for path, variable in nnx.to_flat_state(nnx.state(vae, nnx.Param))
    }


def _load_flat_params(vae: WaveletVAE, weights_path: str | Path) -> None:
    tensors = load_file(str(weights_path))
    flat = nnx.to_flat_state(nnx.state(vae, nnx.Param))
    nnx.update(
        vae,
        nnx.from_flat_state(
            [
                (path, variable.replace(tensors[".".join(map(str, path))]))
                for path, variable in flat
            ]
        ),
    )


def save_vae(
    vae: WaveletVAE,
    save_dir: str | Path,
    *,
    config: dict,
    ema_model: WaveletVAE | None = None,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(_flat_params(vae), str(save_dir / "model.safetensors"))
    if ema_model is not None:
        save_file(_flat_params(ema_model), str(save_dir / "ema_model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_vae(
    save_dir: str | Path, *, prefer_ema: bool = True, rngs: nnx.Rngs | None = None
) -> WaveletVAE:
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    config = {k: v for k, v in config.items() if k != "step"}
    vae = WaveletVAE(**config, rngs=rngs if rngs is not None else nnx.Rngs(0, reparam=1))

    model_path = save_dir / "model.safetensors"
    ema_path = save_dir / "ema_model.safetensors"
    weights_path = ema_path if prefer_ema and ema_path.exists() else model_path
    _load_flat_params(vae, weights_path)
    return vae


def _calibrate_latent_scale(vae: WaveletVAE, frames, batch_size: int = 16) -> float:
    """`frames` may be a jax.Array or a (possibly memory-mapped) numpy array."""
    n = sum_ = sumsq = 0.0
    for i in range(0, frames.shape[0], batch_size):
        mean, _ = vae.encode(jnp.asarray(np.asarray(frames[i : i + batch_size])))
        sum_ += float(jnp.sum(mean))
        sumsq += float(jnp.sum(mean * mean))
        n += mean.size
    var = sumsq / n - (sum_ / n) ** 2
    return var**0.5


def train_vae_on_dataset(
    dataset: Dataset,
    model_config: VAEModelConfig | None = None,
    train_config: VAETrainConfig | None = None,
    sample_fps: float = 2.0,
):
    """Train the VAE on a `Dataset` of clips ``(B, T, H, W, C)``.

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

    input_config = {
        "in_channels": dataset.num_channels,
        "out_channels": dataset.num_channels,
    }
    full_model_config = {**input_config, **asdict(model_config)}

    rngs = nnx.Rngs(0, reparam=1)
    model = WaveletVAE(**full_model_config, rngs=rngs)
    ema_model = nnx.clone(model)
    lr_schedule = linear_warmup_decay_schedule(
        train_config.learning_rate,
        total_steps=train_config.vae_train_steps,
        warmup_steps=train_config.lr_warmup_steps,
        hold_steps=train_config.lr_hold_steps,
        final_lr=train_config.lr_final,
    )
    trainer = VAETrainer(
        model,
        ema_model,
        learning_rate=lr_schedule,
        ema_decay=train_config.ema_decay,
        kl_weight=train_config.kl_weight,
        recon_loss=train_config.recon_loss,
        wavelet_loss=train_config.wavelet_loss,
        detail_weight=train_config.detail_weight,
        rngs=rngs,
    )

    print("dataset clips:", dataset.dataset_size)
    print("train split:", dataset.train_size)
    print("val split:", dataset.val_size)
    print_param_table(model)

    log10_max = 20.0 * math.log10(2.0)
    avg_loss = 0.0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    batch_size = train_config.batch_size

    for step in range(1, train_config.vae_train_steps + 1):
        batch, *_ = dataset.sample_train_batch(batch_size)
        loss = trainer.train_step(jnp.asarray(np.array(batch)))
        avg_loss += loss

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.vae_train_steps
        ):
            val_batch, *_ = dataset.sample_val_batch(min(batch_size, dataset.val_size))
            val_batch = jnp.asarray(np.array(val_batch))

            eval_metrics = trainer.eval_loss(val_batch)
            recon = eval_metrics["recon"]
            psnr = log10_max - 10.0 * jnp.log10(
                jnp.maximum(eval_metrics["l2_loss"], 1e-12)
            )
            kl = eval_metrics["kl_loss"]

            window_steps = step - last_log_step
            loss_f = float(loss)
            avg_loss_f = float(avg_loss) / window_steps
            psnr_f, kl_f = float(psnr), float(kl)
            l2_loss_f = float(eval_metrics["l2_loss"])
            l1_loss_f = float(eval_metrics["l1_loss"])
            detail_loss_f = float(eval_metrics["wavelet_loss"])

            now = time.time()
            samples = window_steps * batch_size
            samples_per_sec = samples / max(now - last_log_time, 1e-8)

            lr_f = float(lr_schedule(step)) if callable(lr_schedule) else lr_schedule
            print(
                f"step={step:5d} sample/s={samples_per_sec:.2f} "
                f"loss={loss_f:.4f} avg={avg_loss_f:.4f} lr={lr_f:.2e} "
                f"val_psnr={psnr_f:.2f}dB val_kl={kl_f:.4f}"
            )
            if train_logger is not None:
                train_logger.log_train_metrics(
                    step,
                    {
                        "samples_per_second": samples_per_sec,
                        "loss": loss_f,
                        "avg_loss": avg_loss_f,
                        "learning_rate": lr_f,
                    },
                )
                train_logger.log_train_metrics(
                    step,
                    {
                        "psnr": psnr_f,
                        "kl": kl_f,
                        "l2_loss": l2_loss_f,
                        "l1_loss": l1_loss_f,
                        "detail_loss": detail_loss_f,
                    },
                    val=True,
                )
                train_logger.log_reconstructions(
                    step,
                    val_batch[:, -1],
                    {0.0: recon[:, -1]},
                )
            avg_loss = 0.0
            last_log_step = step
            last_log_time = time.time()

    # Latent-scale calibration pass on the training data (uses EMA weights).
    latent_scale = _calibrate_latent_scale(
        ema_model, dataset.train_videos, batch_size=train_config.batch_size * 4
    )
    latent_scale = latent_scale if latent_scale > 1e-6 else 1.0
    print(f"calibrated latent_scale={latent_scale:.4f}")
    full_model_config["latent_scale"] = latent_scale
    model.latent_scale = latent_scale
    ema_model.latent_scale = latent_scale

    return model, ema_model, full_model_config


def _self_test() -> None:
    rngs = nnx.Rngs(0, reparam=1)
    frames = jax.random.uniform(
        jax.random.key(0), shape=(2, 4, 32, 32, 3), minval=-1.0, maxval=1.0
    )
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=8,
        num_downsamples=1,
        use_wavelet=True,
    )
    model = WaveletVAE(**cfg, rngs=rngs)
    ema_model = nnx.clone(model)
    trainer = VAETrainer(
        model, ema_model, learning_rate=3e-3, kl_weight=1e-6, rngs=rngs
    )

    recon, _, _ = model(frames, rngs=rngs)
    assert recon.shape == frames.shape, (recon.shape, frames.shape)
    mse_start = float(jnp.mean((recon - frames) ** 2))
    for step in range(500):
        loss = float(trainer.train_step(frames))
    recon, _, _ = model(frames, rngs=rngs)
    mse_end = float(jnp.mean((recon - frames) ** 2))
    print(f"recon MSE: start={mse_start:.5f} end={mse_end:.5f}")
    assert mse_end < mse_start, "recon MSE did not drop"
    print("self-test passed: recon MSE dropped.")


# def self_wavelet_test() -> None:


if __name__ == "__main__":
    _self_test()
