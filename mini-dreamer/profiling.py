import time
import statistics

import jax
import jax.numpy as jnp
import numpy as np


def benchmark(fn, *args, n_warmup: int = 5, n_iters: int = 30,
              calls_per_iter: int = 1) -> dict:
    """Benchmark `fn(*args)`, returning timing stats in milliseconds."""
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)

    samples_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        for _ in range(calls_per_iter):
            out = fn(*args)
        jax.block_until_ready(out)
        samples_ms.append((time.perf_counter() - t0) * 1e3 / calls_per_iter)

    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "iqr_ms": np.subtract(*np.percentile(samples_ms, [75, 25])),
        "n": n_iters,
    }


def benchmark_attention():
    """A/B the manual MQA (`CrossAttention.__call__`) against the fused path
    (`CrossAttention.fast_call`, backed by jax.nn.dot_product_attention).

    Correctness is checked before any timing: a fast wrong kernel is worthless.
    """
    from flax import nnx

    from unet_jax import CrossAttention

    print("=" * 72)
    print("CrossAttention: manual MQA vs jax.nn.dot_product_attention")
    print("=" * 72)
    print(f"Backend: {jax.default_backend()}  Devices: {jax.devices()}")

    B, D, CTX_D, H = 4, 256, 128, 4
    dtype = jnp.bfloat16 if jax.default_backend() == "gpu" else jnp.float32

    attn = CrossAttention(D, CTX_D, num_heads=H, dtype=dtype, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(attn)

    @jax.jit
    def naive(state, x, context):
        return nnx.merge(graphdef, state)(x, context)

    @jax.jit
    def fused(state, x, context):
        return nnx.merge(graphdef, state).fast_call(x, context)

    # bf16 attention accumulates error over the softmax reduction, so the
    # tolerance scales with the compute dtype, not with the implementation.
    atol = 2e-2 if dtype == jnp.bfloat16 else 1e-4

    print(f"\n{'T':>6} | {'S':>6} | {'naive ms':>10} | {'fused ms':>10} | {'speedup':>8} | {'max |Δ|':>9}")
    for T, S in ((64, 64), (256, 64), (1024, 64), (1024, 256)):
        kx, kc = jax.random.split(jax.random.fold_in(jax.random.key(0), T * S), 2)
        x = jax.random.normal(kx, (B, T, D), dtype)
        context = jax.random.normal(kc, (B, S, CTX_D), dtype)

        out_naive = np.asarray(naive(state, x, context), dtype=np.float32)
        out_fused = np.asarray(fused(state, x, context), dtype=np.float32)
        max_abs_diff = np.max(np.abs(out_naive - out_fused))
        ok = max_abs_diff <= atol

        sn = benchmark(naive, state, x, context, calls_per_iter=5)
        sf = benchmark(fused, state, x, context, calls_per_iter=5)
        print(f"{T:>6} | {S:>6} | {sn['min_ms']:>10.3f} | {sf['min_ms']:>10.3f} "
              f"| {sn['min_ms'] / sf['min_ms']:>7.2f}x | {max_abs_diff:>9.2e}"
              f"{'' if ok else '  <- MISMATCH'}")

    print(f"\nTolerance: atol={atol:g} (dtype={jnp.dtype(dtype).name})")

    # Roofline context: the naive path materializes the (B, H, T, S) score
    # matrix in HBM, so its traffic grows with T*S; the fused path keeps the
    # scores on-chip. Expect the gap to widen with T on GPU, and to stay flat
    # on CPU where XLA lowers both to the same unfused loops.
    print()


if __name__ == "__main__":
    benchmark_attention()
