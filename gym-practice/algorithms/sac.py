import math

import gymnasium as gym
import mlx.core as mx
import numpy as np


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

    def sample(self, observation):
        mu = self.net(observation)
        z = mx.random.normal(shape=mu.shape)
        return z * self.std + mu

    def get_action(self, observation, sample=True):
        """Sample action for training, deterministic action for evaluation."""
        observation = mx.asarray(observation, dtype=mx.float32)
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
        self.history = []
        self.max_size = max_size
        self.pointer = 0

    @property
    def size(self):
        return len(self.history)

    def append(self, batch):
        if len(self.history) < self.max_size:
            self.history.append(batch)
            return

        # We evict in LIFO manner
        self.history[self.pointer] = batch
        self.pointer = (self.pointer + 1) % self.max_size

    def __str__(self) -> str:
        """Minimal, readable summary of the buffer contents."""
        header = (
            f"Buffer(size={self.size}, max_size={self.max_size}, pointer={self.pointer})"
        )
        if not self.history:
            return f"{header}\n[]"

        rows = [f"  {i:>3}: {item}" for i, item in enumerate(self.history)]
        return f"{header}\n" + "\n".join(rows)


def rollout(env: gym.Env, policy, buffer, num_iters):
    state, _ = env.reset()

    for _ in range(num_iters):
        action = policy(state)
        observation, reward, terminated, truncated, _ = env.step(action)
        buffer.append((state, action, reward, observation, terminated))

    return


def q_update_step(q_fn, double_q_fn_ema, policy, buffer, gamma, alpha):
    """Q estimates future return for action-value pairs"""

    state, action, reward, next_state, is_next_state_terminal = buffer
    q1_ema, q2_ema = double_q_fn_ema
    pred = q_fn(state, action)

    # grab fresh action from policy
    next_action = policy.get_action(next_state)
    bellman_backup = mx.minimum(
        q1_ema(next_state, next_action), q2_ema(next_state, next_action)
    )
    zero_if_terminal = 1 - is_next_state_terminal

    target = reward + zero_if_terminal * gamma * (
        bellman_backup - alpha * policy.entropy(next_action, next_state)
    )

    loss = mx.mean((pred - target) ** 2)

    # optimize

    return loss


env = gym.make("Pendulum-v1", max_episode_steps=200)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n
policy = GaussianDistribution(net=TinyLinearNet())
buffer = Buffer()
rollout(env, policy, buffer, 10)
print(buffer)
