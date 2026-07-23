"""A didactic, standalone JAX profiling exemplar.

Run it anywhere (Mac CPU, Colab GPU) with:

    uv run profiling_exemplar.py

The script walks through the fundamentals of measuring JAX performance,
in order of importance:

  1. Async dispatch — why naive timing lies to you
  2. Compilation vs. execution — separating one-time cost from steady state
  3. A correct micro-benchmark harness — warmup, repeats, robust statistics
  4. Roofline thinking — is this op compute-bound or bandwidth-bound?
  5. Comparing implementations — naive attention vs. fused flash attention
  6. Trace profiling — capturing a timeline you can open in Perfetto

Each section prints what it measured and, more importantly, what the
number *means*. Read the comments top to bottom; they are the tutorial.
"""

import time
import timeit
import statistics

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Section 0: know your hardware.
#
# Every performance number is relative to the machine it ran on. Record the
# device in every benchmark log you keep — a kernel that is a win on a T4
# can be a loss on an A100 and vice versa, because their compute/bandwidth
# ratios differ.
# ---------------------------------------------------------------------------

def section_0_environment() -> None:
    print("=" * 72)
    print("0. Environment")
    print("=" * 72)
    print(f"JAX version : {jax.__version__}")
    print(f"Backend     : {jax.default_backend()}")
    print(f"Devices     : {jax.devices()}")
    print()


# ---------------------------------------------------------------------------
# Section 1: async dispatch.
#
# JAX ops are dispatched ASYNCHRONOUSLY: calling `f(x)` enqueues work on the
# device and returns a future-like array immediately. If you wrap the call
# in `time.perf_counter()` without forcing completion you measure *dispatch
# cost* (microseconds), not *execution cost*.
#
# Rule: always call `.block_until_ready()` on (one of) the outputs before
# stopping the clock. `jax.block_until_ready(pytree)` handles whole pytrees.
# ---------------------------------------------------------------------------

def section_1_async_dispatch() -> None:
    print("=" * 72)
    print("1. Async dispatch: the classic timing mistake")
    print("=" * 72)

    x = jnp.ones((2048, 2048))
    f = jax.jit(lambda a: a @ a)
    f(x).block_until_ready()  # warmup: compile + first run (see section 2)

    t0 = time.perf_counter()
    y = f(x)                       # returns before the matmul finishes!
    t_dispatch = time.perf_counter() - t0

    t0 = time.perf_counter()
    y = f(x)
    y.block_until_ready()          # forces the device to finish
    t_real = time.perf_counter() - t0

    print(f"Without block_until_ready: {t_dispatch * 1e6:8.1f} us   <- dispatch only, a lie")
    print(f"With    block_until_ready: {t_real * 1e6:8.1f} us   <- actual execution")
    print()
    # Note: async dispatch is a FEATURE in production code — it lets the
    # Python interpreter run ahead and keep the device queue full. You only
    # need to block when *measuring* (or when pulling results to the host).


# ---------------------------------------------------------------------------
# Section 2: compilation vs. execution.
#
# The first call to a jitted function traces the Python, compiles with XLA,
# then runs. Subsequent calls with the same input *shapes and dtypes* hit a
# cache and just run. These are different quantities:
#
#   - compile time: paid once per shape signature; matters for dev loops
#     and for programs that see many shapes (a common silent perf killer).
#   - execution time: paid every call; this is your steady-state latency.
#
# `f.lower(x).compile()` lets you do the two phases explicitly, and the
# compiled object exposes XLA's FLOP/memory estimates via cost_analysis().
# ---------------------------------------------------------------------------

def section_2_compile_vs_execute() -> None:
    print("=" * 72)
    print("2. Compilation vs. execution")
    print("=" * 72)

    x = jnp.ones((1024, 1024))

    def g(a):
        return jnp.tanh(a @ a) @ a

    t0 = time.perf_counter()
    lowered = jax.jit(g).lower(x)      # trace python -> StableHLO
    compiled = lowered.compile()       # StableHLO -> device code
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled(x).block_until_ready()
    t_exec = time.perf_counter() - t0

    print(f"Compile time : {t_compile * 1e3:8.2f} ms  (paid once per input shape)")
    print(f"Execute time : {t_exec * 1e3:8.2f} ms  (paid every call)")

    # XLA's own static estimate of the work in the compiled program.
    # Useful for roofline math without hand-counting FLOPs.
    cost = compiled.cost_analysis()
    if isinstance(cost, list):  # older JAX returned a list of dicts
        cost = cost[0]
    if cost:
        flops = cost.get("flops")
        bytes_accessed = cost.get("bytes accessed")
        print(f"XLA estimate : {flops:.3g} FLOPs, {bytes_accessed:.3g} bytes accessed")
    print()
    # Pitfall: if your code recompiles often (watch for it with the env var
    # JAX_LOG_COMPILES=1), the culprit is usually varying shapes (e.g. a
    # ragged final batch) or passing python scalars that change value as
    # static arguments. Pad to fixed shapes or mark args static deliberately.


# ---------------------------------------------------------------------------
# Section 3: a correct micro-benchmark harness.
#
# Ingredients of a trustworthy measurement:
#   a) Warmup: run the function a few times first so compilation, memory
#      allocator growth, and autotuning are out of the way.
#   b) Repeats: run many iterations; one sample is noise.
#   c) Robust statistics: report the MEDIAN (and spread), not the mean —
#      timing noise is one-sided (OS jitter, clock ramping only ever make
#      you slower), so the mean is biased upward by outliers.
#   d) Amortize dispatch: for very fast ops (<~100us), time a *batch* of N
#      calls per sample so per-call Python overhead doesn't dominate.
#
# This harness is the thing to copy into your own experiments.
# ---------------------------------------------------------------------------

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


def section_3_harness() -> None:
    print("=" * 72)
    print("3. Micro-benchmark harness")
    print("=" * 72)

    x = jnp.ones((2048, 2048), dtype=jnp.float32)
    f = jax.jit(lambda a: a @ a)

    stats = benchmark(f, x)
    print(f"2048x2048 f32 matmul: median {stats['median_ms']:.3f} ms, "
          f"min {stats['min_ms']:.3f} ms, IQR {stats['iqr_ms']:.3f} ms "
          f"(n={stats['n']})")
    print()
    # Reading the spread: if IQR is a large fraction of the median, the
    # machine is noisy (shared Colab GPU, thermal throttling, background
    # load) and small A/B differences are not trustworthy. Prefer comparing
    # MIN times for A/B tests on noisy machines: min is the least-noise
    # observation of the same fixed workload.


# ---------------------------------------------------------------------------
# Section 4: roofline thinking.
#
# Before optimizing an op, ask which resource limits it:
#
#   arithmetic intensity = FLOPs / bytes moved      [FLOP/byte]
#   machine balance      = peak FLOP/s / peak B/s   [FLOP/byte]
#
#   intensity < balance  -> BANDWIDTH-bound: fuse ops, shrink dtypes,
#                           avoid materializing intermediates. More FLOPs
#                           are free; trips to HBM are what cost you.
#   intensity > balance  -> COMPUTE-bound: better algorithms, tensor cores
#                           (bf16/fp16), or fewer FLOPs.
#
# Most pointwise/normalization ops are heavily bandwidth-bound; big matmuls
# are compute-bound. Attention is the interesting middle case — the naive
# form materializes an (L, L) matrix, making it bandwidth-bound; flash
# attention restructures it to never write that matrix to HBM.
# ---------------------------------------------------------------------------

def section_4_roofline() -> None:
    print("=" * 72)
    print("4. Roofline: compute-bound vs bandwidth-bound")
    print("=" * 72)

    n = 4096
    x = jnp.ones((n, n), dtype=jnp.float32)

    # --- a big matmul: high arithmetic intensity --------------------------
    matmul = jax.jit(lambda a: a @ a)
    s = benchmark(matmul, x)
    flops = 2 * n**3                     # n^2 outputs, n MACs each
    bytes_moved = 3 * n * n * 4          # read A, read B(=A), write C; f32
    intensity = flops / bytes_moved
    achieved_tflops = flops / (s["min_ms"] * 1e-3) / 1e12
    print(f"matmul   : {s['min_ms']:7.3f} ms | intensity {intensity:7.1f} FLOP/B "
          f"| achieved {achieved_tflops:.2f} TFLOP/s")

    # --- a pointwise op: intensity < 1, pure bandwidth --------------------
    pointwise = jax.jit(lambda a: jnp.tanh(a) * 2.0 + 1.0)
    s = benchmark(pointwise, x, calls_per_iter=10)
    bytes_moved = 2 * n * n * 4          # read x, write out (XLA fuses the chain)
    gbps = bytes_moved / (s["min_ms"] * 1e-3) / 1e9
    print(f"pointwise: {s['min_ms']:7.3f} ms | intensity ~   1.2 FLOP/B "
          f"| achieved {gbps:.1f} GB/s of memory bandwidth")
    print()
    # Compare 'achieved' against your device's datasheet peaks (e.g. A100:
    # ~19.5 f32 TFLOP/s without tensor cores, ~2039 GB/s HBM). Reaching
    # 60-80% of a peak means that resource is the limiter and you are
    # near it; reaching 5% means your bottleneck is elsewhere (launch
    # overhead, the other resource, or serialization).


# ---------------------------------------------------------------------------
# Section 5: comparing implementations — attention two ways.
#
# The A/B comparison is the bread and butter of optimization work. Rules:
#   - identical inputs, identical dtypes, identical output check FIRST
#     (a fast wrong kernel is worthless — always verify numerics before
#     trusting a speedup);
#   - same harness, same machine, back to back;
#   - vary the size that matters (here: sequence length) to see *scaling*,
#     not just one point.
#
# Naive attention materializes the (L, L) score matrix in HBM.
# jax.nn.dot_product_attention can use a fused flash-attention kernel
# (cudnn on NVIDIA GPUs) that keeps scores in on-chip SRAM.
# ---------------------------------------------------------------------------

def naive_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale   # (B, H, L, L) in HBM!
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(scores, axis=-1), v)


def section_5_attention_ab() -> None:
    print("=" * 72)
    print("5. A/B comparison: naive vs fused attention")
    print("=" * 72)

    on_gpu = jax.default_backend() == "gpu"
    # On GPU, ask for the cudnn flash-attention kernel; elsewhere use the
    # XLA implementation so the script still runs (it demonstrates the
    # methodology, not a speedup, on CPU).
    implementation = "cudnn" if on_gpu else "xla"
    fused = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(
        q, k, v, implementation=implementation))
    naive = jax.jit(naive_attention)

    B, H, D = 4, 8, 64
    key = jax.random.key(0)
    dtype = jnp.bfloat16 if on_gpu else jnp.float32  # cudnn wants 16-bit

    print(f"{'seq len':>8} | {'naive ms':>10} | {'fused ms':>10} | {'speedup':>8}")
    for L in (256, 512, 1024):
        kq, kk, kv = jax.random.split(jax.random.fold_in(key, L), 3)
        q = jax.random.normal(kq, (B, L, H, D), dtype)
        k = jax.random.normal(kk, (B, L, H, D), dtype)
        v = jax.random.normal(kv, (B, L, H, D), dtype)

        # Correctness first, speed second.
        np.testing.assert_allclose(naive(q, k, v), fused(q, k, v),
                                   atol=2e-2, rtol=2e-2)

        sn = benchmark(naive, q, k, v, calls_per_iter=5)
        sf = benchmark(fused, q, k, v, calls_per_iter=5)
        print(f"{L:>8} | {sn['min_ms']:>10.3f} | {sf['min_ms']:>10.3f} "
              f"| {sn['min_ms'] / sf['min_ms']:>7.2f}x")
    print()
    # On a GPU expect the gap to WIDEN with L: naive traffic grows O(L^2)
    # while flash attention's HBM traffic stays O(L). On CPU/XLA the two
    # may be comparable — that is itself a finding: an optimization only
    # exists relative to a bottleneck, and CPU has a different one.


# ---------------------------------------------------------------------------
# Section 6: trace profiling — see WHERE time goes, not just how much.
#
# Timers answer "how long?"; a trace answers "on what?". jax.profiler.trace
# records every kernel with timestamps. Open the output in Perfetto
# (https://ui.perfetto.dev -> open the .json.gz / use xprof for the full dir)
# or TensorBoard's profile plugin.
#
# Annotate your code so the trace speaks your language:
#   - jax.profiler.TraceAnnotation("name")  — labels host-side spans
#   - jax.named_scope("name")               — labels ops INSIDE jit, so
#     kernels in the trace are grouped under your model-level names
#     ("encoder/resblock0" instead of "fusion.127").
#
# Workflow: capture a few steady-state steps (never the compile step),
# then look for: the top-k longest kernels (optimize those), gaps between
# kernels (dispatch/host bottleneck), and memcpys you didn't expect
# (host<->device transfers hiding in your data path).
# ---------------------------------------------------------------------------

def section_6_trace(trace_dir: str) -> None:
    print("=" * 72)
    print("6. Trace capture")
    print("=" * 72)

    @jax.jit
    def step(a):
        with jax.named_scope("projection"):
            h = jnp.tanh(a @ a)
        with jax.named_scope("head"):
            return jnp.sum(h @ a)

    x = jnp.ones((1024, 1024))
    step(x).block_until_ready()  # compile OUTSIDE the trace window

    with jax.profiler.trace(trace_dir):
        for i in range(5):
            with jax.profiler.TraceAnnotation(f"train_step_{i}"):
                step(x).block_until_ready()

    print(f"Trace written to: {trace_dir}")
    print("View it with either:")
    print("  - https://ui.perfetto.dev  (load the .trace.json.gz inside)")
    print("  - tensorboard --logdir <trace_dir>  (Profile tab; needs")
    print("    `pip install tensorboard tensorboard-plugin-profile`)")
    print()
    # Also worth knowing (not run here):
    #   jax.profiler.save_device_memory_profile("mem.prof")  — a pprof-format
    #   snapshot of what is resident on the device; the tool for hunting
    #   OOMs and forgotten references keeping buffers alive.


def main() -> None:
    section_0_environment()
    section_1_async_dispatch()
    section_2_compile_vs_execute()
    section_3_harness()
    section_4_roofline()
    section_5_attention_ab()
    section_6_trace("/tmp/jax-profile-exemplar")


if __name__ == "__main__":
    main()
