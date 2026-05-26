import copy
import math
from typing import Iterable

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np

np.set_printoptions(precision=3, suppress=True)


class TinyLinearNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        """x: (4,) -> logits: (2,)"""
        x = nn.relu(self.fc1(x))
        return self.fc2(x)  # Raw logits


class EMA:
    def __init__(self, net: nn.Module, decay=0.999):
        self.decay = decay
        self.net = copy.deepcopy(net)

    def update(self, params):
        params = mlx_utils.tree_flatten(params, destination={})

        pdict: dict[str, mx.array] = {}
        mlx_utils.tree_flatten(self.net.parameters(), destination=pdict)

        for pname, pval in pdict.items():
            pdict[pname] = pval * self.decay + (1 - self.decay) * params[pname]

        self.net.update(mlx_utils.tree_unflatten(pdict))

    @property
    def parameters(self):
        return self.net.parameters()

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class SquashedGaussianDistribution(nn.Module):
    def __init__(self, state_dim, action_dim, range_limit):
        self.dim = action_dim  # Dimension of the gaussian
        self.net = TinyLinearNet(input_dim=obs_space, output_dim=self.dim * 2)
        self.range_limit = range_limit

    def log_prob_action(self, action, observation):
        # N x action_dim * 2
        params = self.net(observation)
        mu, log_std = params[:, : self.dim], params[:, self.dim :]
        std = mx.exp(log_std)
        var = std**2

        # Normal(. | mu, cov=diag(var)))
        U = mx.random.normal(shape=action.shape) * std + mu

        # logprob(Normal(mu, cov=diag(var)))
        normalization_constant = -0.5 * math.log(2 * math.pi)
        log_prob = -((U - mu) ** 2) / (2 * var) - log_std + normalization_constant

        # Sum over independent Gaussians
        log_prob = log_prob.sum(axis=1)

        # taken from Appendix C of SAC paper
        # To make sense of this, think about U = tanhh^-1(action)
        # and action comes by transforming U via mu and std
        # So in expectation your log prob *does* connect U and action via mu and std
        # eps for stability
        eps = 1e-8
        tanh_correction = mx.sum(mx.log(1 - mx.tanh(U) ** 2 + eps), axis=1)

        # logprob(tanh(Normal(action, mu, cov=diag(var))))
        return log_prob - tanh_correction

    def entropy_estimate(self, action, observation):
        """This only works in expectation i.e sample multiple actions and average"""
        return -self.log_prob_action(action, observation)

    def sample(self, observation):
        params = self.net(observation)
        mu, log_std = params[:, : self.dim], params[:, self.dim :]
        std = mx.exp(log_std)
        z = mx.random.normal(shape=mu.shape)
        a = mx.tanh(z * std + mu)
        return a * self.range_limit

    def get_action(self, observation, sample=True):
        """Sample action for training, deterministic action for evaluation."""
        if sample:
            return self.sample(observation)

        # No sampling, the mu is deterministic given a state
        mu = self.net(observation)
        return mu

    def __call__(self, observation):
        observation = mx.asarray(observation, dtype=mx.float32)
        return self.get_action(observation, sample=True)


class Buffer:
    def __init__(self, max_size: int = 1000) -> None:
        self.history: list[tuple[mx.array, ...]] = []
        self.max_size = max_size
        self.pointer = 0

    @property
    def size(self):
        return len(self.history)

    def append(self, batch):
        batch = tuple(map(mx.asarray, batch))
        if len(self.history) < self.max_size:
            self.history.append(batch)
            return

        # We evict in LIFO manner
        self.history[self.pointer] = batch
        self.pointer = (self.pointer + 1) % self.max_size

    def __getitem__(self, index) -> list[mx.array]:
        if isinstance(index, slice):
            # Resolve 'None' values and negative indices to absolute values
            start, stop, step = index.indices(self.size)

            batch = [self.history[b] for b in range(start, stop, step)]
            batched_dimensions: list[tuple[mx.array]] = list(zip(*batch))

            return list(map(lambda x: mx.contiguous(mx.concat(x)), batched_dimensions))

        return self.history[index]

    def _format_value(self, value) -> str:
        if isinstance(value, Iterable):
            return "".join(f"{v:.3f} " for v in value)
        return f"{value:.3f}" if isinstance(value, float) else value

    def __str__(self) -> str:
        """Minimal, readable summary of the buffer contents."""
        header = f"Buffer(size={self.size}, max_size={self.max_size}, pointer={self.pointer})"
        if not self.history:
            return f"{header}\n[]"
        step_size = self.size // 5
        rows = [
            f"{i * step_size:>3}: {[self._format_value(v) for v in sample]}"
            for i, sample in enumerate(self.history[::step_size])
        ]
        return f"{header}\n" + "\n".join(rows)


def rollout(env: gym.Env, policy, buffer, num_iters):
    state, _ = env.reset()

    for _ in range(num_iters):
        action = policy(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        buffer.append((state, action, reward, observation, terminated))
        state = observation

    return


def q_loss_fn(
    q_fn: nn.Module,
    double_q_fn_ema: tuple[EMA, EMA],
    policy: SquashedGaussianDistribution,
    buffer: list[mx.array],
    gamma: float,
    alpha: float,
):
    """Q estimates future return for action-value pairs"""

    state, action, reward, next_state, is_next_state_terminal = buffer
    q1_ema, q2_ema = double_q_fn_ema

    state_action_pair = mx.concatenate([state, action], axis=1)
    pred = q_fn(state_action_pair)

    # grab fresh action from policy
    next_action = policy.get_action(next_state)
    next_state_action_pair = mx.concatenate([next_state, next_action], axis=1)
    bellman_backup = mx.minimum(
        q1_ema(next_state_action_pair), q2_ema(next_state_action_pair)
    )
    zero_if_terminal = 1 - is_next_state_terminal

    target = reward + zero_if_terminal * gamma * (
        bellman_backup - alpha * policy.entropy_estimate(next_action, next_state)
    )

    loss = mx.mean((pred - target) ** 2)
    mx.eval(loss)
    return loss


q_grad_fn = mx.value_and_grad(q_loss_fn)


def q_update_step(
    q_fn: nn.Module,
    optimizer: optim.Optimizer,
    double_q_fn_ema: tuple[EMA, EMA],
    policy: SquashedGaussianDistribution,
    buffer: list[mx.array],
    gamma: float,
    alpha: float,
) -> tuple[float, float]:
    q_loss, q_grads = q_grad_fn(q_fn, double_q_fn_ema, policy, buffer, gamma, alpha)
    clipped_grads, total_grad_norm = optim.clip_grad_norm(q_grads, max_norm=2.0)
    optimizer.update(q_fn, clipped_grads)
    return float(q_loss), total_grad_norm


def policy_loss_fn(
    policy: SquashedGaussianDistribution,
    state: mx.array,
    double_q_fn_ema: tuple[EMA, EMA],
    alpha: float,
):
    """Policy always wants to maximize rewards / returns"""
    q1, q2 = double_q_fn_ema
    action = policy.get_action(state)

    state_action_pair = mx.concatenate([state, action], axis=1)
    q_estimate = mx.minimum(q1(state_action_pair), q2(state_action_pair))
    soft_return_estimate = q_estimate + alpha * policy.entropy_estimate(action, state)
    loss = mx.mean(-soft_return_estimate)
    return loss


def policy_update_step(
    policy: SquashedGaussianDistribution,
    optimizer: optim.Optimizer,
    state: mx.array,
    double_q_fn_ema: tuple[EMA, EMA],
    alpha: float,
) -> tuple[float, float]:
    policy_grad_fn = mx.value_and_grad(policy_loss_fn)
    loss, grads = policy_grad_fn(policy, state, double_q_fn_ema, alpha)
    clipped_grads, total_grad_norm = optim.clip_grad_norm(grads, max_norm=2.0)
    optimizer.update(policy, clipped_grads)
    return float(loss), total_grad_norm


gamma = 0.975
alpha = 0.2
buffer = Buffer(max_size=5)
env = gym.make_vec("Pendulum-v1", max_episode_steps=200)
obs_space = env.single_observation_space.shape[0]
action_dim = env.single_action_space.shape[0]
action_space_limit = env.single_action_space.high

print(f"{obs_space = } - {action_dim = }")
policy = SquashedGaussianDistribution(
    state_dim=obs_space, action_dim=action_dim, range_limit=action_space_limit
)

q1 = TinyLinearNet(input_dim=obs_space + action_dim, output_dim=1)
q2 = TinyLinearNet(input_dim=obs_space + action_dim, output_dim=1)
q1_ema = EMA(q1, decay=0.999)
q2_ema = EMA(q2, decay=0.999)

policy_optimizer = optim.AdamW(learning_rate=1e-3)
q1_optimizer = optim.AdamW(learning_rate=1e-3)
q2_optimizer = optim.AdamW(learning_rate=1e-3)

rollout(env, policy, buffer, 10)

batch = buffer[0:2]
states = batch[0]

q1_loss, q1_grad_norm = q_update_step(
    q1,
    q1_optimizer,
    (q1_ema, q2_ema),
    policy=policy,
    buffer=batch,
    gamma=gamma,
    alpha=alpha,
)
q2_loss, q2_grad_norm = q_update_step(
    q2,
    q2_optimizer,
    (q1_ema, q2_ema),
    policy=policy,
    buffer=batch,
    gamma=gamma,
    alpha=alpha,
)

# update EMAs
q1_ema.update(q1.parameters())
q2_ema.update(q2.parameters())


policy_loss, policy_grad_norm = policy_update_step(
    policy,
    policy_optimizer,
    states,
    (q1_ema, q2_ema),
    alpha=alpha,
)


print(
    f"{q1_loss = :.3f} - {q1_grad_norm = :.3f} - {q2_loss = :.3f} - {q2_grad_norm = :.3f}"
)

print(f"{policy_loss = :.3f} - {policy_grad_norm = :.3f}")
