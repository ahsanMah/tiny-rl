# Add parent directory to sys.path to import logger
import os
import re
import sys
import time
from itertools import accumulate

import gymnasium as gym
import mlx.core as mx
from loguru import logger
from mlx.core import e
from mlx.core import linalg as la
from rich.pretty import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_utils import RLLogger, VideoLogger

# mx.random.seed(4321)

ENV_NAME = "Acrobot-v1"
ENV_NAME = "CartPole-v1"
NUM_PARALLEL_ENVS = 4

# Create our training environment - a cart with a pole that needs balancing
env = gym.make_vec(
    ENV_NAME,
    num_envs=NUM_PARALLEL_ENVS,
    vectorization_mode="sync",
    max_episode_steps=128,
)

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.

print(f"Starting observation: {observation}")
print(f"Action space:{env.action_space}")
print("Observation space:")
pprint(env.observation_space)

# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]


# Tiny linear net: 4 inputs -> hidden (16) -> 2 action logits
# Weights initialized with simple random values
class TinyLinearNet:
    def __init__(
        self,
        input_dim=4,
        hidden_dim=32,
        output_dim=2,
        init_scale=0.1,
        init_scale_final=0.01,
    ):
        self.W1 = mx.random.normal((input_dim, hidden_dim)) * init_scale
        self.b1 = mx.zeros(hidden_dim)
        self.W2 = mx.random.normal((hidden_dim, output_dim)) * init_scale_final
        self.b2 = mx.zeros(output_dim)

    @property
    def params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def update(self, params):
        self.W1, self.b1, self.W2, self.b2 = params

    def forward(self, x):
        """x: (4,) -> logits: (2,)"""
        x = mx.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        return x @ self.W2 + self.b2  # Raw logits


class CategoricalDistribution:
    def __init__(self, net):
        self.net = net

    def log_prob_action(self, action, observation):
        """Returns log_p(action | observation) assuming action is index"""
        logits = self.net.forward(observation)
        log_prob = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        selected = mx.take_along_axis(log_prob, action[:, None], axis=1).squeeze()
        return selected

    def sample(self, logits) -> list[int]:
        probs = mx.exp(logits - mx.logsumexp(logits, axis=1, keepdims=True))
        cdf = mx.cumsum(probs, axis=1)
        random_val = mx.random.uniform(0, 1, shape=(probs.shape[0], 1))
        mask = random_val < cdf

        # print("probs =", probs)
        # print("cdf =", cdf)
        # print("random_val =", random_val)
        # print("mask =", mask)

        # grab first nonzero index
        category_index = mx.argmax(mask, axis=1)

        return category_index.tolist()

    def get_action(self, observation, sample=True):
        """Sample action for training, argmax action for evaluation."""
        obs = mx.asarray(observation, dtype=mx.float32)
        logits = self.net.forward(obs)
        if sample:
            # int(mx.random.categorical(logits).item())
            return self.sample(logits)
        return int(mx.argmax(logits).item())


class Buffer:
    """Flat buffer of trajectories indexed by a pointer"""

    def __init__(self):
        self.state: list = []
        self.observation: list = []
        self.action: list[int] = []
        self.reward: list[float] = []
        self.value: list[float] = []
        self.reward_to_go: list[float] = []
        self.advantage: list[float] = []
        self.ptr: int = 0
        self.max_size: int = -1

        self.episode_return = []
        self.episode_steps = []

    def update(self, s, o, a, r, v):
        self.state.append(s)
        self.observation.append(o)
        self.action.append(a)
        self.reward.append(r)
        self.value.append(v)

    def reset(self):
        self.state = []
        self.observation = []
        self.action = []
        self.reward = []
        self.ptr = 0

    def complete(self, terminal_value=0.0):
        # Grab all recorded values and rewards
        trajectory_values = self.value[self.ptr :]
        rewards = self.reward[self.ptr :]

        v_st = trajectory_values
        # when state terminated due to death we add a zero reward
        # otherwise this should be the value_fn(state)
        v_st_next = v_st[1:] + [terminal_value]

        v_st = mx.array(v_st)
        v_st_next = mx.array(v_st_next)
        reward_vec = mx.asarray(rewards)

        td_residuals = reward_vec + gamma(1) * v_st_next - v_st
        td_residuals = td_residuals[::-1].tolist()
        # advantage = td_residual + ema_factor * gamma(1) * future_t_advantage
        advantage = accumulate(
            td_residuals,
            lambda future_adv, td: td + ema_factor * gamma(1) * future_adv,
        )
        advantage = list(reversed(list(advantage)))

        # rewards: list = rewards[::-1].tolist()
        reward_to_go = []
        # r[1] =                gamma(0)r[1] + gamma(1)r[2] + gamma(2)r[3]
        # r[0] = gamma(0)r[0] + gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
        reward_to_go = accumulate(
            rewards[::-1], lambda r_sum, rt: rt + gamma(1) * r_sum
        )
        reward_to_go = list(reversed(list(reward_to_go)))

        # print("rewards =", rewards)
        # print("reward_to_go =", reward_to_go)
        # assert len(reward_to_go) == len(rewards)
        assert reward_to_go[-1] == rewards[-1], (
            f"Final timepoint should match got {reward_to_go[-1]} but expected {rewards[-1]}"
        )
        self.advantage.extend(advantage)
        self.reward_to_go.extend(reward_to_go)
        self.episode_return.append(sum(rewards))
        self.episode_steps.append(len(rewards))

        assert len(self.reward) == len(self.advantage), print(
            f"{len(self.reward)}, {len(self.advantage)}"
        )
        assert len(self.state) == len(self.reward_to_go)
        assert len(self.action) == len(self.advantage), print(
            f"{len(self.action)}, {len(self.advantage)}"
        )

        self.ptr += len(rewards)


class VecStats:
    def __init__(self):
        """Computes running meand and variance according to Welford's algorithm"""
        self.mean = 0
        self.count = 0
        self.m2 = 0

    def update(self, x):
        self.count += 1
        prev_mean = self.mean * 1.0  # just to get a copy if arr
        self.mean += (x - self.mean) / self.count
        self.m2 += (x - prev_mean) * (x - self.mean)

    @property
    def var(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0


# Initialize the networks and start an episode

obs_shape = env.single_observation_space.shape[0]
action_shape = env.single_action_space.n

net = TinyLinearNet(input_dim=obs_shape, output_dim=action_shape)
value_fn = TinyLinearNet(
    input_dim=obs_shape,
    output_dim=1,
    init_scale=0.1,
    init_scale_final=1.0,
)
mx.eval(net.params)
mx.eval(value_fn.params)
print("================")
print("Using LinearNet:")
print(net)
print("================")

lr = 0.01
lr_val = 0.02
grad_clip_value = 2
num_epochs = 20
num_trajectories = 128
val_train_batch_size = 32
policy = CategoricalDistribution(net)
discount_factor = 0.95
ema_factor = 0.96
gamma = lambda t: discount_factor**t

metrics_logger = RLLogger("./tb-logs/", exp_name=f"{ENV_NAME}-vpg-gae")
eval_video_logger = VideoLogger(
    env_name=ENV_NAME, exp_folder=f"./eval-logs/{metrics_logger.run_name}"
)


state_min = env.single_observation_space.low
state_max = env.single_observation_space.high
print("state_min =", state_min)
print("state_max =", state_max)

# The statsitics will be recorded for the lifetime fo the algorithm
state_stats = VecStats()
reward_stats = VecStats()


def normalize(x, stats, eps=1e-8):
    mu, var = stats.mean, stats.var
    # print(f"State mean: {mu}, var: {var}, count: {state_stats.count}")
    if x.ndim > 1:
        mu = mu[None, :]
        var = var[None, :]
    return (x - mu) / (var + eps) ** 0.5


# Setting the params in net allows the gradients
# to be recorded by autograd
def loss_fn(params, action, state):
    policy.net.update(params)
    return policy.log_prob_action(action, state)


def vec_loss_fn(params, action, state, advantage):
    policy.net.update(params)
    log_prob_of_policy = policy.log_prob_action(action, state)
    # \sum (\grad theta) * r <==> \grad \sum (r * theta)
    log_prob_of_policy = log_prob_of_policy * advantage
    return log_prob_of_policy.sum()


def value_loss_fn(params, state, reward):
    value_fn.update(params)
    r_hat = value_fn.forward(state).flatten()
    return mx.mean((r_hat - reward).square())


def train_value_fn(states, rewards):

    num_samples = len(states)
    drop_samples = (
        num_samples % val_train_batch_size if val_train_batch_size < num_samples else 0
    )
    num_batches = max(1, num_samples // val_train_batch_size)

    states = states[drop_samples:]
    rewards = rewards[drop_samples:]

    # states = mx.stack([mx.asarray(s) for s in buffer.state[drop_samples:]], axis=0)
    # rewards = mx.stack(
    #     [mx.asarray(s) for s in buffer.reward_to_go[drop_samples:]], axis=0
    # )

    state_batch = states.split(num_batches, axis=0)
    reward_batch = rewards.split(num_batches, axis=0)

    avg_loss = 0.0
    for state, reward in zip(state_batch, reward_batch):
        # we update value_fn for each (state, reward) pair
        loss, grad_theta = value_grad_fn(value_fn.params, state=state, reward=reward)
        avg_loss += loss.item()

        # SGD step
        for p, grad in zip(value_fn.params, grad_theta):
            p -= lr_val * grad

    avg_loss /= num_batches
    return avg_loss


vec_policy_grad_fn = mx.value_and_grad(vec_loss_fn)
value_grad_fn = mx.value_and_grad(value_loss_fn)

# First evaluation pass
global_step = 0
# eval_video_logger.record_evaluation(policy, global_step)
# metrics_logger.log_video(
#     global_step, eval_video_logger.exp_folder, eval_video_logger.num_eval_episodes
# )

start_time = time.time()

for epoch in range(num_epochs):
    completed_episodes = 0
    # Note: If we do not reset every epoch, any unfinished trajectory *resumes*
    # Allowing the model to learn longer horizon tasks
    observation_vec, info = env.reset()

    # Each parallel env will have its own buffer updated
    buffer_vec = [Buffer() for n in range(env.num_envs)]

    while completed_episodes < num_trajectories:
        # save current state
        state_vec = observation_vec
        for state in state_vec:
            state_stats.update(state)

        # Choose an action using the tiny net (simulates a deterministic policy)
        action_vec = policy.get_action(observation_vec)

        # Take the action and see what happens
        observation_vec, reward_vec, terminated_vec, truncated_vec, info_vec = env.step(
            action_vec
        )

        for reward in reward_vec:
            reward_stats.update(reward)

        # reward: +1 for each step the pole stays upright
        # terminated: True if pole falls too far (agent failed)
        # truncated: True if we hit the time limit (500 steps)

        for i in range(env.num_envs):
            global_step += 1
            buffer = buffer_vec[i]
            state = state_vec[i]
            action = action_vec[i]
            observation = observation_vec[i]
            reward = reward_vec[i]
            terminated = terminated_vec[i]
            truncated = truncated_vec[i]

            state = normalize(state, state_stats)
            observation = normalize(state, state_stats)
            # reward = normalize(reward, reward_stats)
            value = value_fn.forward(state).item()

            # given state, policy produced action with probability that received reward

            if global_step % 10_000 == 0:
                print(state, observation, reward, terminated, truncated, action, value)
                print("state stats: mean =", state_stats.mean, "var =", state_stats.var)
            # print(state, observation)

            buffer.update(state, observation, action, reward, value)

            episode_over = terminated or truncated
            if terminated or truncated:
                completed_episodes += 1
                terminal_value = (
                    0 if terminated else value_fn.forward(observation).item()
                )
                buffer.complete(terminal_value)
                metrics_logger.log_episode(
                    global_step,
                    reward=buffer.episode_return[-1],
                    length=buffer.episode_steps[-1],
                )
            # onto the next!

    # if we do not check, we might be leaving incomplete trajectories in the buffers
    # check the final termination states and complete the path that was not finished!
    for i in range(env.num_envs):
        terminated = terminated_vec[i]
        truncated = truncated_vec[i]
        observation = observation_vec[i]
        buffer = buffer_vec[i]
        episode_over = terminated or truncated
        if not episode_over:
            terminal_value = value_fn.forward(observation).item()
            buffer.complete(terminal_value)
        # else we would have completed it in the while-loop

    # Update V(state) first so that we have good approx for policy grad
    # FIXME: currently we use stale value estimates to train the policy

    # Aggregate batch of trajectories across environments
    # N x obs_space
    state_batch = mx.stack(
        [mx.asarray(s) for buffer in buffer_vec for s in buffer.state], axis=0
    )

    # Normalize via the running mean and var
    # state_batch = normalize_state(state_batch)
    print(f"After normalization mean: {state_batch.mean(axis=0)}")

    # N x 1
    action_batch = mx.concatenate(
        [mx.asarray(buffer.action, dtype=mx.float32) for buffer in buffer_vec], axis=0
    )
    # N x 1
    advantage_batch = mx.concatenate(
        [mx.asarray(buffer.advantage) for buffer in buffer_vec], axis=0
    )

    reward_to_go_batch = mx.concatenate(
        [mx.asarray(buffer.reward_to_go) for buffer in buffer_vec], axis=0
    )
    print("state_batch.shape =", state_batch.shape)
    print("reward_to_go_batch.shape =", reward_to_go_batch.shape)
    value_fn_loss = train_value_fn(state_batch, reward_to_go_batch)

    if epoch == 0:
        continue

    log_probs, policy_grad = vec_policy_grad_fn(
        policy.net.params,
        action=action_batch,
        state=state_batch,
        advantage=advantage_batch,
    )

    avg_grad_norms = mx.array([la.norm(p) for p in policy_grad]).mean()
    avg_grad_norms /= num_trajectories

    episode_returns = mx.concatenate(
        [mx.array(buffer.episode_return) for buffer in buffer_vec], axis=0
    )
    episode_steps = mx.concatenate(
        [mx.array(buffer.episode_steps) for buffer in buffer_vec], axis=0
    )
    mu = reward_to_go_batch.mean().item()
    std = reward_to_go_batch.std().item()
    avg_num_steps = episode_steps.mean().item()

    for i in range(len(policy_grad)):
        # Approximate expectation via sample mean over sampled timesteps.
        policy_grad[i] /= num_trajectories

        # gradient clipping via the grad norm
        grad_norm = la.norm(policy_grad[i])
        scale = grad_clip_value / max(grad_clip_value, grad_norm)
        # Note that dividing by norm gives you unit-norm
        # so multiplying by clip rescales norm from 1 -> chosen value
        # This is a no-op when grad_norm < 1 as scale = 1
        policy_grad[i] *= scale

    old_log_p = policy.log_prob_action(action_batch, state_batch)

    # One gradient ascent step on the parameters
    for p, grad in zip(policy.net.params, policy_grad):
        p += lr * grad

    new_log_p = policy.log_prob_action(action_batch, state_batch)
    logratio = old_log_p - new_log_p

    # 1. Standard Monte Carlo Estimator
    # logratio = logprob_new - logprob_old
    # approx_kl_mc = logratio.mean()

    # 2. Schulman's Estimator (The standard in PPO/CleanRL)
    approx_kl_schulman = 0.5 * (logratio**2).mean()

    train_metrics = {
        "policy_log_probs": log_probs.item(),
        "policy_grad_norm": avg_grad_norms.item(),
        "value_loss": value_fn_loss,
        "approx_kl": approx_kl_schulman.item(),
    }

    metrics_logger.log_train_metrics(global_step, train_metrics)
    metrics_logger.log_speed(global_step, steps_done=global_step, start_time=start_time)

    logger.info(f"======= Epoch {epoch} ======= ")
    logger.info(f"Avg Steps: {avg_num_steps:.1f}")
    logger.info(f"Mean Return: {mu:.2f} +/- {std:.2f}")
    logger.info(f"Approx KL: {approx_kl_schulman:.4f}")
    logger.info(f"Value Loss: {value_fn_loss:.2f}")
    logger.info(f"Avg Policy Gradient Norm: {avg_grad_norms:.2f}")


env.close()

# Final evaluation pass
# eval_video_logger.record_evaluation(policy, global_step)
# metrics_logger.log_video(
#     global_step, eval_video_logger.exp_folder, eval_video_logger.num_eval_episodes
# )

metrics_logger.close()
