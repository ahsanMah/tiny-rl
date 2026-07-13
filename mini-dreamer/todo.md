The current goal is to port the project piece by piece to JAX. We will test whether the components are working after the port. Once a full module is working, we will load the model and compare outputs to the MLX version.

- [x] Create port_utils.py that:
  - [x] Creates test set from Vizdoom (500 step rollouts)
  - [x] Loads VAE model from `/Users/smaug/dev/mini-dreamer-workspace/mini-dreamer/logs/vizdoom-vae` and saves sample reconstructions
  - [x] Loads vae from vae_jax.py and compares
- [x] Move WaveletDownsampleConv and WaveletUpsample into vae
- [x] Port port wavelets only

## Shared JAX utilities (unblocks diffusion port)

- [x] Extract the framework-generic pieces of vae_jax.py into a shared module (e.g. `jax_utils.py`) so diffusion_jax.py reuses them instead of duplicating: `ema_update`, `linear_warmup_decay_schedule`, safetensors save/load helpers (`_flat_params`/`_load_flat_params`), `print_param_table`

## Port unet.py (UNet3D) — prerequisite for diffusion.py

- [x] Create unet_jax.py with NNX ports of:
  - [x] GaussianFourierEmbedding + TimeEmbedding
  - [x] ConvResBlock3D (FiLM time conditioning)
  - [x] TransformerBlock (MQA attention + pooling)
  - [x] UNet3D: encode/decode split, action embeddings + null action, optional wavelet pre/unpool, optional reward head
- [x] Param-table parity check vs the MLX UNet3D (param counts match exactly for all 4 wavelet×reward configs; leaf names differ only by framework convention, weight/kernel)
- [x] port_utils: MLX→JAX weight-copy helpers for UNet3D (Conv3d kernel transposes, Linear/attention layouts, RMSNorm eps — extend the VAE copy helpers)
- [x] port_utils: load the `logs/vizdoom-latent-v2` checkpoint and compare `encode()`/`decode()` outputs on the fixed VizDoom test set (deterministic: fixed `t`, fixed actions, no sampling; pad_multiple=32 to match training)

## Port diffusion.py

Done so far (inference + persistence helpers, no trainer):

- [x] `make_final_frame_mask`, `sample_noise`/`sample_t_logit_normal` with explicit `jax.random` keys
- [x] `sample_euler` (incl. `return_intermediates` via `lax.scan`) + `sample_euler_to_mp4`
- [x] `generate_video` / `generate_env_video`
- [x] `save_model`/`load_model` via the shared safetensors helpers (same file layout as MLX)

### Remaining, in implementation order (each step is testable on its own)

1. [ ] Clean up the `t` grid in `sample_euler`: the eps-based `ts`/`dt` and the
   `jnp.linspace(0, 1, num_steps)` overwrite disagree (dt = `(1-2eps)/N` but t
   values from linspace, endpoint included). Pick one grid; MLX uses
   `t = step/N` and `dt = 1/N`.
   - Test: with `num_steps=4`, print the `(t, dt)` pairs the model sees and
     check the final `x` sits at t=1; later this is covered by the MLX parity
     rollout (step 8).
2. [x] `_loss_at_t` as a **free function** first (not a method): signature
   (done: implementation + shape/finiteness smoke tests in
   smoke_diffusion_jax.py; MLX parity part deferred until the step 10
   checkpoint export)
   `(model, x1, actions, t, *, noise, t_ctx=None, rewards=None, reward_loss_weight, reward_t_threshold, return_eval_aux=False)`.
   Take `noise` as an argument instead of sampling inside — this is what makes
   MLX parity testing possible, and keeps the function pure (jit-friendly).
   - Test: shapes + `jnp.isfinite(loss)`; then MLX parity — feed the identical
     `x1`/`noise`/`t`/`t_ctx`/actions through both `_loss_at_t`s on the
     converted checkpoint and compare loss/recon_mse/r2 to ~1e-4.
3. [ ] `_sample_context_t` + `_dropout_actions` as free functions taking a key.
   - Test: statistical — `t_ctx` samples land in `[min_context_t, 1]`;
     dropout rate over a large batch ≈ `action_dropout`; `min_context_t=1.0`
     and `action_dropout=0.0` are exact no-ops.
4. [ ] `FlowMatchingTrainer.__init__`: `nnx.Optimizer(model, optax.chain(clip_by_global_norm(max_grad_norm), adamw(lr_schedule, weight_decay)))`,
   plus an `nnx.Rngs` (or a raw key you split per step) for t/noise/dropout.
   No `mx.compile` state plumbing needed — that whole block disappears in JAX.
   - Test: constructs without error; `reward_loss_weight>0` without a reward
     head still raises.
5. [ ] `train_step` **eager first, no jit**: split keys, sample `t`/`t_ctx`,
   dropout actions, `nnx.value_and_grad` over step 2's loss, optimizer update,
   `jax_utils.ema_update`.
   - Test: run ~50 steps on one repeated random batch — loss should drop
     hard (overfits a single batch); check grads are nonzero and EMA params
     move slower than model params.
6. [ ] Wrap with `@nnx.jit` — mind the Rngs-across-trace-levels gotcha from
   VAETrainer.
   - Test: jit-vs-eager equivalence — same seeds, 3 steps, losses match to
     float tolerance; second call doesn't retrace (log with
     `jax.monitoring` or just time it).
7. [ ] `eval_loss_by_timestep` (loop over fixed timesteps calling step 2 with
   `return_eval_aux=True`).
   - Test: values on the converted checkpoint + fixed val batch vs MLX.
8. [ ] port_utils parity: fixed-noise `sample_euler` rollout vs MLX on the
   vizdoom-latent-v2 checkpoint (inject the same initial `x`, compare final
   frames), reusing the injected-noise `_loss_at_t` comparison from step 2.
9. [ ] `train_on_dataset`: numpy batches at the loop boundary (`Dataset`
   already yields numpy; convert with `jnp.asarray`), lr logging from the
   optax schedule, resume-ckpt saving.
   - Test: port the `train_overfit_random_noise` path and assert loss drops
     below a threshold in a few hundred steps.
10. [ ] port_utils: export the vizdoom-latent-v2 UNet checkpoint to a
    Flax-loadable one (like `export_flax_vae_checkpoint`) so steps 2/7/8 can
    use `load_model` directly.

### Testing strategy as you go

Layered, cheapest first:

- **Smoke**: every new function gets a tiny-shape call (e.g. `(2, 4, 16, 16, 3)`,
  `base_channels=8`) asserting output shapes and finiteness. Fast enough to run
  on every edit.
- **RNG helpers**: statistical assertions (range, mean, rate), never exact
  values — MLX and JAX RNGs will never bitwise-match.
- **Parity vs MLX**: only for deterministic paths, always by *injecting* the
  random inputs (noise, t, initial x) into both sides rather than seeding.
  This is why `_loss_at_t` should take `noise` as a parameter.
- **Behavioral**: overfit-one-batch (step 5) and overfit-tiny-dataset (step 9)
  catch wiring bugs (wrong mask, dead grads, EMA overwriting the model) that
  parity tests on a fixed checkpoint miss.
- **jit discipline**: implement eager, verify, then jit and diff against eager.
  Keeps "wrong math" and "wrong tracing" failures separate.

Keep these in a `smoke_diffusion_jax.py` (or the jax-tests notebook) so the
whole ladder reruns with one command.

## Port pretrainer.py

- [ ] Swap imports to the JAX modules (diffusion_jax, vae_jax); the config plumbing (dataclasses/click/tomllib) is framework-agnostic and stays
- [ ] Replace the `mx.array` conversions in the commands with np/jnp (preview encoding, sample clips)
- [ ] `train-vae` command runs end-to-end on vae_jax (short VizDoom run)
- [ ] `train` command runs end-to-end, pixel-space and latent-space (`--vae-dir` with the converted Flax VAE checkpoint)
- [ ] `generate` command runs end-to-end (wavelet debug previews, `generate_env_video`)
- [ ] Sanity run: short JAX pretraining on the fixed VizDoom rollout, loss/PSNR in the same ballpark as an equivalent MLX run
