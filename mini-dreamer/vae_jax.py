from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import lpips_jax
import numpy as np
import optax
from flax import nnx
from jax import lax
from safetensors.flax import save_file

# Type-only: a module-level import would be circular
# (diffusion_jax -> jax_utils/unet_jax -> vae_jax).
if TYPE_CHECKING:
    from diffusion_jax import Dataset
from jax_utils import (
    ema_update,
    flat_params,
    linear_warmup_decay_schedule,
    load_flat_params,
)
from logger_utils import RLLogger

ReconLoss = Literal["l1", "l2", "l1+l2"]


@dataclass(frozen=True)
class VAEModelConfig:
    latent_channels: int = 4
    base_channels: int = 32
    num_downsamples: int = 2
    use_wavelet: bool = True
    # Compute dtype of the conv trunk ("float32" or "bfloat16"); params,
    # optimizer state, EMA, norms, and posterior moments stay fp32.
    dtype: str = "float32"


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
    lpips_weight: float = 0.0
    lpips_net: str = "vgg16"  # "vgg16" or "alexnet"
    ema_decay: float = 0.99
    load_dir: str | None = None
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        self.shortcut = (
            nnx.identity
            if in_channels == out_channels
            else nnx.Conv(
                in_channels, out_channels, kernel_size=(1, 1), dtype=dtype, rngs=rngs
            )
        )
        # Norms compute in fp32 regardless of the block's compute dtype.
        self.norm1 = nnx.RMSNorm(out_channels, dtype=jnp.float32, rngs=rngs)
        self.norm2 = nnx.RMSNorm(out_channels, dtype=jnp.float32, rngs=rngs)
        self.conv1 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=1,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=1,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=dtype,
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
        dtype: str | jnp.dtype = "float32",
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
        # Mixed precision: ``dtype`` is the compute dtype of the conv trunk (a
        # string like "bfloat16" so it JSON round-trips through config.json);
        # params, norms, and posterior moments stay fp32. ``None`` keeps
        # flax's default promotion (plain fp32).
        dtype = jnp.dtype(dtype)
        self.dtype = None if dtype == jnp.float32 else dtype
        compute_dtype = self.dtype

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
            enc_in,
            base_channels,
            kernel_size=(3, 3),
            padding=1,
            dtype=compute_dtype,
            rngs=rngs,
        )
        enc_blocks = []
        enc_downs = []
        ch = base_channels
        for _ in range(num_downsamples):
            out_ch = ch * 2
            enc_blocks.append(ConvResBlock2D(ch, out_ch, dtype=compute_dtype, rngs=rngs))
            enc_downs.append(
                nnx.Conv(
                    out_ch,
                    out_ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding=1,
                    dtype=compute_dtype,
                    rngs=rngs,
                )
            )
            ch = out_ch
        self.enc_blocks = nnx.List(enc_blocks)
        self.enc_downs = nnx.List(enc_downs)
        self.enc_mid = ConvResBlock2D(ch, ch, dtype=compute_dtype, rngs=rngs)
        # Posterior moments stay fp32 so KL / reparameterization / latent-scale
        # calibration are exact.
        self.to_moments = nnx.Conv(
            ch,
            latent_channels * 2,
            kernel_size=(1, 1),
            kernel_init=zero_init,
            bias_init=zero_init,
            dtype=jnp.float32,
            rngs=rngs,
        )

        # Decoder: mirror (nearest upsample + conv).
        self.from_latent = nnx.Conv(
            latent_channels,
            ch,
            kernel_size=(3, 3),
            padding=1,
            dtype=compute_dtype,
            rngs=rngs,
        )
        self.dec_mid = ConvResBlock2D(ch, ch, dtype=compute_dtype, rngs=rngs)
        dec_blocks = []
        for _ in range(num_downsamples):
            out_ch = ch // 2
            dec_blocks.append(ConvResBlock2D(ch, out_ch, dtype=compute_dtype, rngs=rngs))
            ch = out_ch
        self.dec_blocks = nnx.List(dec_blocks)
        self.dec_out = nnx.Conv(
            ch,
            dec_out,
            kernel_size=(3, 3),
            padding=1,
            kernel_init=zero_init,
            bias_init=zero_init,
            dtype=compute_dtype,
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
        # Return fp32 under bf16 compute: losses reduce in fp32 and callers
        # (previews, numpy conversion) expect a numpy-compatible dtype.
        return out.astype(jnp.float32) if self.dtype is not None else out

    def __call__(
        self, frames: jnp.ndarray, *, rngs: nnx.Rngs
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, logvar = self.encode(frames)
        z = self.reparameterize(mean, logvar, rngs=rngs)
        recon = self.decode(z)
        return recon, mean, logvar


def kl_divergence(mean: jax.Array, logvar: jax.Array) -> jax.Array:
    return -0.5 * jnp.mean(1.0 + logvar - mean**2 - jnp.exp(logvar))


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
        lpips_weight: float = 0.0,
        lpips_net: str = "vgg16",
        rngs: nnx.Rngs,
    ):
        self.model = model
        self.ema_model = ema_model
        self.ema_decay = ema_decay
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss
        self.wavelet_loss = wavelet_loss
        self.detail_weight = detail_weight
        self.lpips_weight = lpips_weight
        self.lpips = lpips_jax.LPIPSEvaluator(replicate=False, net=lpips_net)

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

    def _lpips(self, recon: jax.Array, x: jax.Array) -> jax.Array:
        """Mean LPIPS distance over frames. Expects inputs in [-1, 1].

        The pretrained VGG/AlexNet backbones take 3-channel NHWC images, so
        clips collapse ``(B, T) -> B*T`` and grayscale is tiled to RGB.
        """
        B, T, H, W, C = x.shape
        r = recon.reshape(B * T, H, W, C)
        t = x.reshape(B * T, H, W, C)
        if C == 1:
            r = jnp.tile(r, (1, 1, 1, 3))
            t = jnp.tile(t, (1, 1, 1, 3))
        return jnp.mean(self.lpips(r, t))

    def loss(self, model: WaveletVAE, batch: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        recon, mean, logvar = model(batch, rngs=rngs)
        recon_loss = self._recon(recon, batch)
        if self.wavelet_loss:
            recon_loss = recon_loss + self.detail_weight * self._wavelet_detail(
                recon, batch
            )
        if self.lpips_weight > 0.0:
            recon_loss = recon_loss + self.lpips_weight * self._lpips(recon, batch)
        kl = kl_divergence(mean, logvar)
        return recon_loss + self.kl_weight * kl

    @nnx.jit
    def eval_loss(self, batch: jax.Array) -> dict[str, jax.Array]:
        metrics = {}
        recon, mean, logvar = self.model(batch, rngs=self.rngs)
        metrics["recon"] = recon
        metrics["l2_loss"] = jnp.mean((recon - batch) ** 2)
        metrics["l1_loss"] = jnp.mean(jnp.abs(recon - batch))
        metrics["kl_loss"] = kl_divergence(mean, logvar)
        metrics["wavelet_loss"] = self._wavelet_detail(recon, batch)
        metrics["lpips_loss"] = self._lpips(recon, batch)
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


def save_vae(
    vae: WaveletVAE,
    save_dir: str | Path,
    *,
    config: dict,
    ema_model: WaveletVAE | None = None,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file(flat_params(vae), str(save_dir / "model.safetensors"))
    if ema_model is not None:
        save_file(flat_params(ema_model), str(save_dir / "ema_model.safetensors"))
    (save_dir / "config.json").write_text(json.dumps(config, indent=2))


def _load_vae_config(save_dir: str | Path) -> dict:
    config = json.loads((Path(save_dir) / "config.json").read_text())
    return {k: v for k, v in config.items() if k != "step"}


def load_vae(
    save_dir: str | Path, *, prefer_ema: bool = True, rngs: nnx.Rngs | None = None
) -> WaveletVAE:
    save_dir = Path(save_dir)
    config = _load_vae_config(save_dir)
    vae = WaveletVAE(**config, rngs=rngs if rngs is not None else nnx.Rngs(0, reparam=1))

    model_path = save_dir / "model.safetensors"
    ema_path = save_dir / "ema_model.safetensors"
    weights_path = ema_path if prefer_ema and ema_path.exists() else model_path
    load_flat_params(vae, weights_path)
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
    if train_config.load_dir is not None:
        full_model_config = _load_vae_config(train_config.load_dir)
        model = load_vae(train_config.load_dir, prefer_ema=False, rngs=rngs)
        print(f"warmstarting training from: {train_config.load_dir}")
    else:
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
        lpips_weight=train_config.lpips_weight,
        lpips_net=train_config.lpips_net,
        rngs=rngs,
    )

    print("dataset clips:", dataset.dataset_size)
    print("train split:", dataset.train_size)
    print("val split:", dataset.val_size)
    # Batch of 1: tabulate jit-traces every submodule with these shapes and
    # none of it is reused by training, so keep the trace cheap.
    table_batch, *_ = dataset.sample_val_batch(1)
    table_batch = jnp.asarray(np.array(table_batch))
    print(nnx.tabulate(model, table_batch, rngs=nnx.Rngs(0), depth=2))

    log10_max = 20.0 * math.log10(2.0)
    avg_loss = 0.0
    start = time.time()
    last_log_time = start
    last_log_step = 0
    batch_size = train_config.batch_size

    for step in range(1, train_config.vae_train_steps + 1):
        batch, *_ = dataset.sample_train_batch(batch_size)
        loss = trainer.train_step(batch)
        avg_loss += loss

        if (
            step == 1
            or step % train_config.log_every == 0
            or step == train_config.vae_train_steps
        ):
            val_batch, *_ = dataset.sample_val_batch(min(batch_size, dataset.val_size))

            eval_metrics = trainer.eval_loss(val_batch)
            recon = eval_metrics["recon"]
            psnr = log10_max - 10.0 * jnp.log10(
                jnp.maximum(eval_metrics["l2_loss"], 1e-12)
            )
            kl = eval_metrics["kl_loss"]
            # Train PSNR on the batch just trained on, so the train/val gap
            # (overfitting signal) is visible in tensorboard.
            train_psnr = log10_max - 10.0 * jnp.log10(
                jnp.maximum(trainer.eval_loss(batch)["l2_loss"], 1e-12)
            )

            window_steps = step - last_log_step
            loss_f = float(loss)
            avg_loss_f = float(avg_loss) / window_steps
            psnr_f, kl_f = float(psnr), float(kl)
            train_psnr_f = float(train_psnr)
            l2_loss_f = float(eval_metrics["l2_loss"])
            l1_loss_f = float(eval_metrics["l1_loss"])
            detail_loss_f = float(eval_metrics["wavelet_loss"])
            lpips_loss_f = float(eval_metrics["lpips_loss"])

            now = time.time()
            samples = window_steps * batch_size
            samples_per_sec = samples / max(now - last_log_time, 1e-8)

            lr_f = float(lr_schedule(step)) if callable(lr_schedule) else lr_schedule
            print(
                f"step={step:5d} sample/s={samples_per_sec:.2f} "
                f"loss={loss_f:.4f} avg={avg_loss_f:.4f} lr={lr_f:.2e} "
                f"train_psnr={train_psnr_f:.2f}dB "
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
                        "psnr": train_psnr_f,
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
                        "lpips_loss": lpips_loss_f,
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


def evaluate_reconstructions(
    vae_dir: str | Path,
    data_dir: str | Path,
    *,
    num_pairs: int = 8,
    prefer_ema: bool = True,
    batch_size: int = 16,
) -> None:
    """Evaluate deterministic VAE reconstructions on all validation clips.

    The last frame of every validation clip is reconstructed through the
    posterior mean. Results are written directly to ``vae_dir``.
    """
    # Deferred: a module-level import would be circular (diffusion_jax ->
    # jax_utils/unet_jax -> vae_jax).
    from diffusion_jax import Dataset

    import matplotlib.pyplot as plt
    import seaborn as sns

    from video_utils import to_uint8_video

    vae_dir = Path(vae_dir)
    vae = load_vae(vae_dir, prefer_ema=prefer_ema)
    dataset = Dataset(data_dir=data_dir, memory_map=True)
    lpips = lpips_jax.LPIPSEvaluator(replicate=False, net="vgg16")
    metric_values = {"mse": [], "l1": [], "psnr": [], "lpips": []}
    grid_frames = grid_recons = None

    for start in range(0, dataset.val_size, batch_size):
        stop = min(start + batch_size, dataset.val_size)
        clips = jnp.asarray(np.asarray(dataset.val_videos[start:stop]))
        frames = clips[:, -1:]
        recon = decode_latents(vae, encode_clips(vae, frames))
        error = recon - frames
        mse = jnp.mean(error**2, axis=(1, 2, 3, 4))
        l1 = jnp.mean(jnp.abs(error), axis=(1, 2, 3, 4))
        psnr = 20.0 * math.log10(2.0) - 10.0 * jnp.log10(jnp.maximum(mse, 1e-12))
        B, _, H, W, C = frames.shape
        lpips_recon = recon.reshape(B, H, W, C)
        lpips_frames = frames.reshape(B, H, W, C)
        if C == 1:
            lpips_recon = jnp.tile(lpips_recon, (1, 1, 1, 3))
            lpips_frames = jnp.tile(lpips_frames, (1, 1, 1, 3))
        lpips_values = jnp.mean(lpips(lpips_recon, lpips_frames), axis=(1, 2, 3))

        for name, values in zip(metric_values, (mse, l1, psnr, lpips_values)):
            metric_values[name].extend(np.asarray(values).tolist())
        if grid_frames is None:
            grid_frames, grid_recons = frames[:num_pairs], recon[:num_pairs]

    summaries = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
        for name, values in metric_values.items()
    }
    metrics_path = vae_dir / "reconstruction_metrics.json"
    metrics_path.write_text(
        json.dumps({"num_samples": dataset.val_size, "metrics": summaries}, indent=2)
    )

    originals = to_uint8_video(np.asarray(grid_frames[:, 0]))
    recons = to_uint8_video(np.asarray(grid_recons[:, 0]))
    n = originals.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6), squeeze=False)
    for i in range(n):
        for row, image in enumerate((originals[i], recons[i])):
            ax = axes[row, i]
            ax.imshow(image.squeeze(-1) if image.shape[-1] == 1 else image, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("recon")
    fig.suptitle(f"Validation PSNR: {summaries['psnr']['mean']:.2f} dB")
    fig.tight_layout()
    grid_path = vae_dir / "reconstruction_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for ax, (name, values) in zip(axes.flat, metric_values.items()):
        sns.histplot(values, ax=ax, bins="auto")
        ax.set_title(name.upper())
        ax.set_xlabel(name)
    fig.tight_layout()
    histogram_path = vae_dir / "reconstruction_metrics_histograms.png"
    fig.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"saved reconstruction grid to: {grid_path}")
    print(f"saved metrics summary to: {metrics_path}")
    print(f"saved metric histograms to: {histogram_path}")


def _self_test(
    *, dtype: str = "float32", lpips_weight: float = 0.0, steps: int = 500
) -> None:
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
        dtype=dtype,
    )
    model = WaveletVAE(**cfg, rngs=rngs)
    ema_model = nnx.clone(model)
    trainer = VAETrainer(
        model,
        ema_model,
        learning_rate=3e-3,
        kl_weight=1e-6,
        lpips_weight=lpips_weight,
        rngs=rngs,
    )

    recon, _, _ = model(frames, rngs=rngs)
    assert recon.shape == frames.shape, (recon.shape, frames.shape)
    assert recon.dtype == jnp.float32, recon.dtype
    mse_start = float(jnp.mean((recon - frames) ** 2))
    for step in range(steps):
        loss = float(trainer.train_step(frames))
    recon, _, _ = model(frames, rngs=rngs)
    mse_end = float(jnp.mean((recon - frames) ** 2))
    print(
        f"[dtype={dtype} lpips_weight={lpips_weight}] "
        f"recon MSE: start={mse_start:.5f} end={mse_end:.5f}"
    )
    assert mse_end < mse_start, "recon MSE did not drop"
    print("self-test passed: recon MSE dropped.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VAE self-test (default), reconstruction evaluation, or calibration."
    )
    sub = parser.add_subparsers(dest="cmd")
    recon = sub.add_parser("eval", help="evaluate validation reconstructions")
    recon.add_argument("--vae-dir", required=True, help="dir saved via save_vae")
    recon.add_argument("--data-dir", required=True, help="rollout dir for Dataset")
    recon.add_argument("--num-pairs", type=int, default=8)
    recon.add_argument("--no-ema", action="store_true", help="use raw (non-EMA) weights")

    calibrate = sub.add_parser("calibrate", help="calibrate the latent scales")
    calibrate.add_argument("--vae-dir", required=True, help="dir saved via save_vae")
    calibrate.add_argument("--data-dir", required=True, help="rollout dir for Dataset")
    args = parser.parse_args()

    if args.cmd == "calibrate":
        from diffusion_jax import Dataset
        vae = load_vae(args.vae_dir, prefer_ema=True)
        dataset = Dataset(data_dir=args.data_dir, memory_map=True)
        latent_scale = _calibrate_latent_scale(
            vae, dataset.train_videos, batch_size=64
        )
        latent_scale = latent_scale if latent_scale > 1e-6 else 1.0
        print(f"calibrated latent_scale={latent_scale}")
    elif args.cmd == "eval":
        evaluate_reconstructions(
            args.vae_dir,
            args.data_dir,
            num_pairs=args.num_pairs,
            prefer_ema=not args.no_ema,
        )
    else:
        _self_test()
        _self_test(dtype="bfloat16")
        _self_test(dtype="bfloat16", lpips_weight=0.5, steps=100)
