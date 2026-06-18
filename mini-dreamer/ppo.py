"""
PPO v2 - a redesigned, standalone PPO.

This is a from-scratch rewrite of ``ppo.py`` in gym-practice aimed at being *simpler* and more
*amenable to mini-dreamer*, where the eventual goal is to train a policy inside
a learned world model ("the dream"). The redesign is organized around three
seams that the original monolithic ``run()`` did not expose:

1. **The network is injected.** ``run()`` takes a ``make_net`` factory instead
   of hardcoding ``TinyLinearNet``. Swap in a CNN/encoder over frames or latents
   and nothing else changes. The four ``*_init_scale`` knobs no longer leak into
   the top-level signature.

2. **Collection is separated from the update.** ``collect_rollout()`` turns a
   policy + environment into a flat ``Batch``; ``update()`` turns a ``Batch``
   into gradient steps. Neither knows about the other's internals.

3. **The environment is a contract, not a concrete type.** ``collect_rollout``
   only assumes the Gymnasium *vector* API: ``reset()``, ``step(actions) ->
   (obs, reward, terminated, truncated, info)`` (with ``info["final_obs"]`` on
   autoreset), plus ``num_envs`` / ``single_observation_space`` /
   ``single_action_space``. A ``DreamEnv`` that implements the same surface is a
   drop-in replacement for the real Gym env.

Correctness fixes over the original, all in one place now:

* **Consistent observation normalization.** Every forward pass goes through the
  net's ``__call__`` (no ``.forward()`` bypass), and the running statistics are
  frozen for the entire collect+update of an iteration, so the values used for
  GAE and the values learned in the update are on the same scale.
* **Real PPO update epochs.** The batch is reused for ``update_epochs`` passes
  with freshly shuffled minibatches each pass — the whole point of the clipped
  objective. (The original did a single pass, i.e. clipped A2C.)
* **Entropy bonus** in the policy loss and a **learnable** Gaussian log-std.
* **Per-minibatch advantage normalization.**
* **Proper truncation bootstrapping** in GAE (terminated -> 0, truncated ->
  V(final_obs)), done with a single vectorized reverse scan over a fixed-horizon
  ``(T, num_envs, ...)`` buffer instead of per-env Python lists + pointers.

Everything is plain MLX + Gymnasium + NumPy; there is no logging/eval dependency
so the file runs on its own (``python ppo_v2.py --env-id CartPole-v1``).
"""

import argparse
import math
import time
from dataclasses import dataclass, field

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Single source of truth for every PPO knob.

    Grouped and documented so the top-level entry point takes one object instead
    of ~25 positional kwargs + a ``**kwargs`` catch-all. Mirrors the dataclass /
    TOML pattern the world-model side of the project already uses.
    """

    # env / rollout
    env_id: str = "CartPole-v1"
    num_envs: int = 4
    num_steps: int = 512  # steps collected per env per iteration
    total_timesteps: int = 200_000

    # network (consumed by the default make_net; ignored if you inject your own)
    hidden_dims: tuple = (64, 64)
    final_init_scale: float = 0.01  # small last-layer init aids PPO stability

    # optimization
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    batch_size: int = 128  # batch size for the gradient updates

    # PPO / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    normalize_obs: bool = True
    normalize_adv: bool = True

    seed: int = 0

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size


@dataclass
class Batch:
    """A flat (num_envs*num_steps, ...) slab of experience ready for updating."""

    obs: mx.array
    actions: mx.array
    returns: mx.array
    advantages: mx.array


# ---------------------------------------------------------------------------
# Observation normalization (batched Welford, frozen during an iteration)
# ---------------------------------------------------------------------------
class RunningNorm:
    """Running mean/var over the feature axis, updated once per iteration.

    Stored as a plain object (not an ``mx.array`` / ``nn.Module``) so that when
    it is held as an attribute on a net, MLX leaves it untracked — the same
    trick the original used for ``state_stats``. ``mean``/``var`` are constants
    from the gradient's point of view.
    """

    def __init__(self, shape, eps: float = 1e-8):
        self.mean = mx.zeros(shape)
        self.var = mx.ones(shape)
        self.count = 1e-4
        self.eps = eps

    def update(self, x):
        """x: (N, *shape). Chan's parallel variance update."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        mean, var, count = np.asarray(self.mean), np.asarray(self.var), self.count
        delta = batch_mean - mean
        total = count + batch_count

        new_mean = mean + delta * batch_count / total
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * count * batch_count / total
        new_var = m2 / total

        self.mean = mx.array(new_mean.astype(np.float32))
        self.var = mx.array(new_var.astype(np.float32))
        self.count = total

    def normalize(self, x):
        return (x - self.mean) / mx.sqrt(self.var + self.eps)


# ---------------------------------------------------------------------------
# Network (injected via make_net; default is a small MLP)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Plain MLP with optional built-in observation normalization.

    All callers use ``__call__`` (which normalizes) — there is no second,
    un-normalized ``forward`` path to drift out of sync.
    """

    def __init__(self, in_dim, out_dim, hidden_dims, normalizer=None, final_scale=0.01):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        self.hidden = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.head = nn.Linear(dims[-1], out_dim)
        # Small last-layer init: near-uniform policy / near-zero value at start.
        self.head.weight = mx.random.normal(self.head.weight.shape) * final_scale
        self.head.bias = mx.zeros(out_dim)
        self.normalizer = normalizer  # plain attr -> untracked by MLX

    def __call__(self, x):
        if self.normalizer is not None:
            x = self.normalizer.normalize(x)
        for layer in self.hidden:
            x = nn.relu(layer(x))
        return self.head(x)


def make_net(in_dim, out_dim, cfg: PPOConfig, normalizer=None):
    """Default factory. Replace this callable to feed PPO a different encoder
    (e.g. a CNN over dreamed frames) without touching the algorithm."""
    return MLP(
        in_dim,
        out_dim,
        cfg.hidden_dims,
        normalizer=normalizer,
        final_scale=cfg.final_init_scale,
    )


# ---------------------------------------------------------------------------
# Action distributions (wrap a net; expose a common log_prob / entropy / sample)
# ---------------------------------------------------------------------------
class Categorical(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def _log_probs(self, obs):
        logits = self.net(obs)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def log_prob(self, obs, action):
        lp = self._log_probs(obs)
        return mx.take_along_axis(lp, action[:, None], axis=-1).squeeze(-1)

    def entropy(self, obs):
        lp = self._log_probs(obs)
        return -(mx.exp(lp) * lp).sum(axis=-1)

    def sample(self, obs):
        return mx.random.categorical(self.net(obs))

    def get_action(self, obs, sample=True):
        single = np.asarray(obs).ndim == 1
        o = mx.array(np.asarray(obs, dtype=np.float32))
        if single:
            o = o[None]
        a = self.sample(o) if sample else mx.argmax(self.net(o), axis=-1)
        a = np.array(a)
        return int(a[0]) if single else a


class DiagGaussian(nn.Module):
    def __init__(self, net, action_dim):
        super().__init__()
        self.net = net
        self.log_std = mx.zeros(action_dim)  # learnable, state-independent

    def log_prob(self, obs, action):
        mu = self.net(obs)
        var = mx.exp(2 * self.log_std)
        lp = (
            -0.5 * ((action - mu) ** 2 / var)
            - self.log_std
            - 0.5 * math.log(2 * math.pi)
        )
        return lp.sum(axis=-1)

    def entropy(self, obs):
        per_sample = (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
        return per_sample * mx.ones(obs.shape[0])

    def sample(self, obs):
        mu = self.net(obs)
        return mu + mx.exp(self.log_std) * mx.random.normal(mu.shape)

    def get_action(self, obs, sample=True):
        single = np.asarray(obs).ndim == 1
        o = mx.array(np.asarray(obs, dtype=np.float32))
        if single:
            o = o[None]
        a = self.sample(o) if sample else self.net(o)
        a = np.array(a)
        return a[0] if single else a


# ---------------------------------------------------------------------------
# Losses (module is the first arg so mx.value_and_grad differentiates its params)
# ---------------------------------------------------------------------------
def policy_loss_fn(policy, obs, action, old_log_prob, advantage, clip_coef, ent_coef):
    log_prob = policy.log_prob(obs, action)
    entropy = policy.entropy(obs).mean()

    ratio = mx.exp(log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = mx.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantage
    # optimizer minimizes, so negate the surrogate we want to maximize
    pg_loss = -mx.minimum(unclipped, clipped).mean()
    return pg_loss - ent_coef * entropy


def value_loss_fn(value_net, obs, returns):
    values = value_net(obs).reshape(-1)
    return 0.5 * mx.mean((values - returns) ** 2)


# ---------------------------------------------------------------------------
# Collection: (policy, env) -> Batch.  Knows nothing about the update.
# ---------------------------------------------------------------------------
def collect_rollout(env, policy, value_net, obs, cfg, discrete):
    """Run ``cfg.num_steps`` vectorized steps and return (batch, last_obs, eps).

    ``obs`` is carried across iterations so unfinished trajectories resume
    (longer effective horizon). The environment only needs to honor the
    Gymnasium vector contract documented at the top of this file — a DreamEnv
    can stand in here unchanged.
    """
    T, ne = cfg.num_steps, env.num_envs
    obs_space = env.single_observation_space
    obs_shape = obs_space.shape

    obs_buf = np.zeros((T, ne, *obs_shape), dtype=np.float32)
    act_buf = (
        np.zeros((T, ne), dtype=np.int32)
        if discrete
        else np.zeros((T, ne, *env.single_action_space.shape), dtype=np.float32)
    )
    rew_buf = np.zeros((T, ne), dtype=np.float32)
    done_buf = np.zeros((T, ne), dtype=np.float32)
    val_buf = np.zeros((T, ne), dtype=np.float32)
    term_val_buf = np.zeros((T, ne), dtype=np.float32)  # bootstrap on truncation

    ep_returns, ep_lengths = [], []
    ep_ret = np.zeros(ne, dtype=np.float64)
    ep_len = np.zeros(ne, dtype=np.int64)

    for t in range(T):
        obs_buf[t] = obs
        obs_mx = mx.array(obs, dtype=mx.float32)

        action = policy.sample(obs_mx)
        value = value_net(obs_mx).reshape(-1)
        mx.eval(action, value)

        val_buf[t] = np.array(value)
        action_np = np.array(action)
        act_buf[t] = action_np

        obs, reward, terminated, truncated, info = env.step(action_np)
        done = np.logical_or(terminated, truncated)
        rew_buf[t] = reward
        done_buf[t] = done

        ep_ret += reward
        ep_len += 1

        # Truncation (time limit) keeps a real successor: bootstrap from it.
        boot = np.logical_and(truncated, np.logical_not(terminated))
        if boot.any() and "final_obs" in info:
            idx = np.where(boot)[0]
            finals = np.stack([np.asarray(info["final_obs"][i]) for i in idx])
            fv = np.array(value_net(mx.array(finals.astype(np.float32))).reshape(-1))
            term_val_buf[t, idx] = fv

        for i in np.where(done)[0]:
            ep_returns.append(float(ep_ret[i]))
            ep_lengths.append(int(ep_len[i]))
            ep_ret[i] = 0.0
            ep_len[i] = 0

    # Bootstrap value for steps that ran off the end of the rollout still alive.
    next_value = np.array(value_net(mx.array(obs, dtype=mx.float32)).reshape(-1))

    # GAE: single vectorized reverse scan over the time axis.
    advantages = np.zeros((T, ne), dtype=np.float32)
    last_gae = np.zeros(ne, dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - done_buf[t]
        succ_value = next_value if t == T - 1 else val_buf[t + 1]
        # When done, the successor's value is the (truncation) bootstrap or 0.
        boot_value = np.where(done_buf[t] == 1.0, term_val_buf[t], succ_value)
        delta = rew_buf[t] + cfg.gamma * boot_value - val_buf[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + val_buf

    batch = Batch(
        obs=mx.array(obs_buf.reshape(T * ne, *obs_shape)),
        actions=mx.array(act_buf.reshape(T * ne, *act_buf.shape[2:])),
        returns=mx.array(returns.reshape(-1)),
        advantages=mx.array(advantages.reshape(-1)),
    )
    return batch, obs, ep_returns, ep_lengths


# ---------------------------------------------------------------------------
# Update: Batch -> gradient steps.  Knows nothing about the environment.
# ---------------------------------------------------------------------------
policy_grad_fn = mx.value_and_grad(policy_loss_fn)
value_grad_fn = mx.value_and_grad(value_loss_fn)


def update(policy, value_net, policy_opt, value_opt, batch, cfg):
    obs, actions = batch.obs, batch.actions
    returns, advantages = batch.returns, batch.advantages

    # Recompute old log-probs once, under the current (frozen) params/stats, so
    # the ratio starts at exactly 1 on the first epoch and stays consistent.
    old_log_probs = policy.log_prob(obs, actions)
    mx.eval(old_log_probs)

    n = obs.shape[0]
    p_losses, v_losses = [], []
    pbar = tqdm(range(cfg.update_epochs))
    for _ in pbar:
        perm = np.random.permutation(n)
        for start in range(0, n, cfg.batch_size):
            indices = mx.array(perm[start : start + cfg.batch_size])
            mb_adv = advantages[indices]
            if cfg.normalize_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            p_loss, p_grads = policy_grad_fn(
                policy,
                obs[indices],
                actions[indices],
                old_log_probs[indices],
                mb_adv,
                cfg.clip_coef,
                cfg.ent_coef,
            )
            p_grads, _ = optim.clip_grad_norm(p_grads, cfg.max_grad_norm)
            policy_opt.update(policy, p_grads)

            v_loss, v_grads = value_grad_fn(value_net, obs[indices], returns[indices])
            v_grads, _ = optim.clip_grad_norm(v_grads, cfg.max_grad_norm)
            value_opt.update(value_net, v_grads)

            mx.eval(
                policy.parameters(),
                value_net.parameters(),
                policy_opt.state,
                value_opt.state,
            )
            p_losses.append(float(p_loss))
            v_losses.append(float(v_loss))

        avg_p_loss = np.mean(p_losses[-n:])
        avg_v_loss = np.mean(v_losses[-n:])

        pbar.set_postfix(
            policy_loss=f"{avg_p_loss:.4f}", value_loss=f"{avg_v_loss:.4f}"
        )

    # Diagnostics on the full batch (Schulman's approx KL + clip fraction).
    new_log_probs = policy.log_prob(obs, actions)
    log_ratio = new_log_probs - old_log_probs
    approx_kl = float((mx.exp(log_ratio) - 1 - log_ratio).mean())
    clip_frac = float((mx.abs(mx.exp(log_ratio) - 1) > cfg.clip_coef).mean())

    return {
        "policy_loss": np.mean(p_losses),
        "value_loss": np.mean(v_losses),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_agent(env, cfg):
    """Construct (policy, value_net, normalizer, obs_shape, discrete)."""
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_shape = obs_space.shape
    in_dim = int(np.prod(obs_shape))

    discrete = not np.issubdtype(act_space.dtype, np.floating)
    action_dim = act_space.n if discrete else int(np.prod(act_space.shape))

    normalizer = RunningNorm(obs_shape) if cfg.normalize_obs else None
    policy_net = make_net(in_dim, action_dim, cfg, normalizer)
    value_net = make_net(in_dim, 1, cfg, normalizer)
    mx.eval(policy_net.parameters(), value_net.parameters())

    policy = (
        Categorical(policy_net) if discrete else DiagGaussian(policy_net, action_dim)
    )
    return policy, value_net, normalizer, obs_shape, discrete


def train_policy(
    env, policy, value_net, normalizer, policy_opt, value_opt, discrete, cfg
):
    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, cfg.num_iterations + 1):
        batch, obs, ep_returns, ep_lengths = collect_rollout(
            env, policy, value_net, obs, cfg, discrete
        )
        global_step += cfg.batch_size

        # Freeze-then-update: stats used for this whole iteration were fixed
        # during collection; fold this rollout in for the *next* iteration.
        if normalizer is not None:
            obs_shape = env.single_observation_space.shape
            normalizer.update(np.array(batch.obs).reshape(-1, *obs_shape))

        metrics = update(policy, value_net, policy_opt, value_opt, batch, cfg)

        sps = int(global_step / (time.time() - start_time))
        mean_return = np.mean(ep_returns) if ep_returns else float("nan")
        logger.info(
            f"iter {iteration:3d} | step {global_step:>8d} | "
            f"return {mean_return:7.2f} ({len(ep_returns)} eps) | "
            f"pi_loss {metrics['policy_loss']:+.3f} | v_loss {metrics['value_loss']:.3f} | "
            f"kl {metrics['approx_kl']:.4f} | clip {metrics['clip_frac']:.2f} | {sps} sps"
        )

    return policy, value_net


def run(cfg: PPOConfig):
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = gym.make_vec(
        cfg.env_id,
        num_envs=cfg.num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    policy, value_net, normalizer, obs_shape, discrete = build_agent(env, cfg)
    policy_opt = optim.AdamW(learning_rate=cfg.policy_lr)
    value_opt = optim.AdamW(learning_rate=cfg.value_lr)

    logger.info(
        f"obs_space={env.single_observation_space} act_space={env.single_action_space}"
    )
    logger.info(
        f"iterations={cfg.num_iterations} batch={cfg.batch_size} "
        f"minibatch={cfg.minibatch_size}"
    )

    policy, value_net = train_policy(
        env, policy, value_net, normalizer, policy_opt, value_opt, discrete, cfg
    )

    env.close()
    return policy, value_net


def main():
    parser = argparse.ArgumentParser(description="Standalone redesigned PPO")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PPOConfig(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
