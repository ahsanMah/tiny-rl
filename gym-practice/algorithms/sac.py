import copy
import math
import os
import sys
from typing import Iterable

import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_utils import VideoLogger

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
        self.net = TinyLinearNet(input_dim=state_dim, output_dim=self.dim * 2)
        self.range_limit = range_limit

    def log_prob_action(self, action, observation):

        if not isinstance(observation, mx.array):
            observation = mx.array(observation)

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
        # Note: entropy is -E[log_prob(action)]. When computing soft target,
        # we add alpha * H = alpha * (-log_prob), which is same as -alpha * log_prob
        return -self.log_prob_action(action, observation)

    def sample(self, observation):
        params = self.net(observation)
        mu, log_std = params[:, : self.dim], params[:, self.dim :]
        std = mx.exp(log_std)
        z = mx.random.normal(shape=mu.shape)
        U = z * std + mu
        a = mx.tanh(U)
        return a * self.range_limit

    def get_action(self, observation, sample=False):
        """Sample action for training, deterministic action for evaluation."""
        observation = mx.asarray(observation, dtype=mx.float32)
        if sample:
            return self.sample(observation)

        # No sampling, the mu is deterministic given a state
        mu = self.net(observation)
        return mu

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

    def _append(self, event):
        event = tuple(map(self._convert_to_mlx, event))
        if len(self.history) < self.max_size:
            self.history.append(event)
            return

        # We evict in LIFO manner
        self.history[self.pointer] = event
        self.pointer = (self.pointer + 1) % self.max_size

    def _convert_to_mlx(self, x):
        x = mx.asarray(x)
        if x.ndim <= 1:
            x = x.reshape(1, -1)
        return x

    def append(self, batch):
        # batch = tuple(map(self._convert_to_mlx, batch))
        num_samples = len(batch[0])
        for i in range(num_samples):
            event = tuple(e[i] for e in batch)
            self._append(event)

    def __getitem__(self, index) -> list[mx.array] | mx.array:
        if isinstance(index, int):
            index = [index]
        elif isinstance(index, Iterable):
            index = index.tolist()

        if isinstance(index, slice):
            # Resolve 'None' values and negative indices to absolute values
            start, stop, step = index.indices(self.size)
            batch = [self.history[b] for b in range(start, stop, step)]
        elif isinstance(index, Iterable):
            batch = [self.history[i] for i in index]

        batched_dimensions: list[tuple[mx.array]] = list(zip(*batch))
        return list(map(lambda x: mx.contiguous(mx.concat(x)), batched_dimensions))

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


def rollout(env: gym.Env, policy, buffer, num_iters, global_step, start_step=5000):
    state, _ = env.reset()
    entropy = 0.0
    for _ in range(num_iters):
        # Grab random actions in the beginning for improved exploration
        action = (
            policy(state) if global_step >= start_step else env.action_space.sample()
        )
        observation, reward, terminated, truncated, _ = env.step(action)
        buffer.append((state, action, reward, observation, terminated))
        state = observation

        entropy += policy.entropy_estimate(action, state).sum()

    return float(entropy) / num_iters


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
    next_action = policy(next_state)
    next_state_action_pair = mx.concatenate([next_state, next_action], axis=1)
    bellman_backup = mx.minimum(
        q1_ema(next_state_action_pair), q2_ema(next_state_action_pair)
    )
    zero_if_terminal = 1 - is_next_state_terminal

    target = reward + zero_if_terminal * gamma * (
        bellman_backup + alpha * policy.entropy_estimate(next_action, next_state)
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
    action = policy(state)

    state_action_pair = mx.concatenate([state, action], axis=1)
    q_estimate = mx.minimum(q1(state_action_pair), q2(state_action_pair))
    # Soft return estimate to maximize is Q(s,a) + alpha * H
    soft_return_estimate = q_estimate + alpha * policy.entropy_estimate(action, state)
    loss = -mx.mean(soft_return_estimate)
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


def run(
    gamma=0.975,
    alpha=0.2,
    num_epochs=20,
    num_updates_per_epoch=1000,
    batch_size=256,
    num_rollouts_per_epoch=1000,
    env_name="Pendulum-v1",
    record_eval_videos=False,
):
    env = gym.make_vec(env_name, num_envs=4, max_episode_steps=200)
    obs_space = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]
    action_space_limit = env.single_action_space.high
    print(f"{obs_space = } - {action_dim = }")

    eval_log_dir = "eval_logs/sac/"
    eval_video_logger = VideoLogger(
        env_name=env_name, exp_folder=f"{eval_log_dir}/test"
    )

    buffer = Buffer(max_size=100_000)
    policy = SquashedGaussianDistribution(
        state_dim=obs_space, action_dim=action_dim, range_limit=action_space_limit
    )

    # First evaluation pass
    global_step = 0

    q1 = TinyLinearNet(input_dim=obs_space + action_dim, output_dim=1)
    q2 = TinyLinearNet(input_dim=obs_space + action_dim, output_dim=1)
    q1_ema = EMA(q1, decay=0.999)
    q2_ema = EMA(q2, decay=0.999)

    policy_optimizer = optim.AdamW(learning_rate=1e-3)
    q1_optimizer = optim.AdamW(learning_rate=1e-3)
    q2_optimizer = optim.AdamW(learning_rate=1e-3)

    for i in range(num_epochs):
        entropy = rollout(env, policy, buffer, num_rollouts_per_epoch, global_step)

        for j in range(num_updates_per_epoch):
            global_step += 1

            sample_indices = mx.random.randint(0, buffer.size, shape=(batch_size,))
            batched_buffer = buffer[sample_indices]

            q1_loss, q1_grad_norm = q_update_step(
                q1,
                q1_optimizer,
                (q1_ema, q2_ema),
                policy=policy,
                buffer=batched_buffer,
                gamma=gamma,
                alpha=alpha,
            )
            q2_loss, q2_grad_norm = q_update_step(
                q2,
                q2_optimizer,
                (q1_ema, q2_ema),
                policy=policy,
                buffer=batched_buffer,
                gamma=gamma,
                alpha=alpha,
            )

            # update EMAs
            q1_ema.update(q1.parameters())
            q2_ema.update(q2.parameters())

            states = batched_buffer[0]
            policy_loss, policy_grad_norm = policy_update_step(
                policy,
                policy_optimizer,
                states,
                (q1_ema, q2_ema),
                alpha=alpha,
            )

        print(
            f"step {global_step}: {entropy = :<5.3f} - {policy_loss = :<5.3f} - {q1_loss = :<5.3f} - {q2_loss = :<5.3f} "
        )
        if (i + 1) % 5 == 0:
            eval_video_logger.record_evaluation(policy, global_step)
            print(
                f"{policy_grad_norm = :.3f} {q1_grad_norm = :.3f} - {q2_grad_norm = :.3f}"
            )


run()
