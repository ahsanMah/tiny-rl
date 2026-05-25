import copy
import math
from typing import Iterable

import gymnasium as gym
import mlx.core as mx
import numpy as np

np.set_printoptions(precision=3, suppress=True)


class TinyLinearNet:
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        self.W1 = mx.random.normal((input_dim, hidden_dim)) * 0.1
        self.b1 = mx.zeros(hidden_dim)
        self.W2 = mx.random.normal((hidden_dim, output_dim)) * 0.1
        self.b2 = mx.zeros(output_dim)

    @property
    def params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def update(self, params):
        self.W1, self.b1, self.W2, self.b2 = params

    def __call__(self, x):
        """x: (4,) -> logits: (2,)"""
        x = mx.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        return x @ self.W2 + self.b2  # Raw logits


class EMA:
    def __init__(self, net: TinyLinearNet, decay=0.999):
        self.decay = decay
        self.net = copy.deepcopy(net)
        self._params = net.params

    def update(self, params):
        for p, q in zip(self._params, params):
            p = p * self.decay + (1 - self.decay) * q
        self.net.update(self._params)

    @property
    def params(self):
        return self._params

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class GaussianDistribution:
    def __init__(self, net):
        self.net = net

    @property
    def std(self):
        # assume diag covariance of 1 - should be learnable in future
        return 1

    def log_prob_action(self, action, observation):
        # N x action_dim
        mu = self.net(observation)
        var = self.std**2

        # logprob(Normal(action, mu=diag, cov=1))

        log_std = math.log(self.std)
        normalization_constant = -0.5 * math.log(2 * math.pi)
        log_prob = -((action - mu) ** 2) / (2 * var) - log_std + normalization_constant

        # Sum over independent Gaussians
        log_prob = log_prob.sum(axis=1)
        return log_prob

    def entropy(self, observation, action):
        return 1

    def sample(self, observation):
        mu = self.net(observation)
        z = mx.random.normal(shape=mu.shape)
        return z * self.std + mu

    def get_action(self, observation, sample=True):
        """Sample action for training, deterministic action for evaluation."""
        if sample:
            return self.sample(observation)

        # No sampling, the mu is deterministic given a state
        mu = self.net(observation)
        return mu

    def update(self, params):
        self.net.update(params)
        return

    def __call__(self, observation):
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
        state = mx.asarray(batch[0], dtype=mx.float32)
        action = mx.asarray(batch[1], dtype=mx.float32)
        reward = mx.array([batch[2]], dtype=mx.float32)
        next_state = mx.asarray(batch[3], dtype=mx.float32)
        terminated = mx.asarray([batch[4]], dtype=mx.bool_)

        batch = (state, action, reward, next_state, terminated)
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

            return list(map(lambda x: mx.contiguous(mx.stack(x)), batched_dimensions))

        return self.history[index]

    # def __slice__

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


def q_update_step(q_fn, double_q_fn_ema, policy, buffer, gamma, alpha):
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
        bellman_backup - alpha * policy.entropy(next_action, next_state)
    )

    loss = mx.mean((pred - target) ** 2)

    # optimize via gradient descent

    return loss


def policy_update_step(policy, state, double_q_fn_ema, alpha):
    """Policy always wants to maximize rewards / returns"""
    q1, q2 = double_q_fn_ema
    action = policy.get_action(state)

    state_action_pair = mx.concatenate([state, action], axis=1)
    q_estimate = mx.minimum(q1(state_action_pair), q2(state_action_pair))
    soft_return_estimate = q_estimate + alpha * policy.entropy(action, state)
    loss = mx.mean(-soft_return_estimate)

    # optimize (implcitly gradient ascent)

    return loss


gamma = 0.975
alpha = 0.2
buffer = Buffer(max_size=5)
env = gym.make("Pendulum-v1", max_episode_steps=200)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
policy = GaussianDistribution(
    net=TinyLinearNet(input_dim=obs_space, output_dim=action_space)
)

q1 = TinyLinearNet(input_dim=obs_space + action_space, output_dim=1)
q2 = TinyLinearNet(input_dim=obs_space + action_space, output_dim=1)
q1_ema = EMA(q1, decay=0.999)
q2_ema = EMA(q2, decay=0.999)

rollout(env, policy, buffer, 10)

batch = buffer[0:2]
states = batch[0]
policy_loss = policy_update_step(policy, states, (q1_ema, q2_ema), alpha)
q_loss = q_update_step(
    q1, (q1_ema, q2_ema), policy=policy, buffer=batch, gamma=gamma, alpha=alpha
)
print(f"{q_loss = } - {policy_loss = }")
