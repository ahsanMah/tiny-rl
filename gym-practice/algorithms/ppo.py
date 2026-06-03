"""
PPO built on vecotrized GAE

TODO
- [x] Add clipped surrogate reward
- [ ] Switch to Adam
- [ ] Add entropy factor
- [x] Switch to continuous action distribution (gaussian)


"""

import math
import os
import sys
import time
from itertools import accumulate

import gymnasium as gym
import mlx.core as mx
import numpy as np
from loguru import logger
from mlx.core import linalg as la

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_utils import RLLogger, VideoLogger

# mx.random.seed(4321)


def gamma(t):
    return discount_factor**t


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
        state_stats=None,
    ):
        self.W1 = mx.random.normal((input_dim, hidden_dim)) * init_scale
        self.b1 = mx.zeros(hidden_dim)
        self.W2 = mx.random.normal((hidden_dim, output_dim)) * init_scale_final
        self.b2 = mx.zeros(output_dim)
        self.state_stats = state_stats

    @property
    def params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def update(self, params):
        self.W1, self.b1, self.W2, self.b2 = params

    def forward(self, x):
        """x: (4,) -> logits: (2,)"""
        x = mx.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        return x @ self.W2 + self.b2  # Raw logits

    def __call__(self, state):
        if self.state_stats is not None:
            state = normalize(state, self.state_stats)
        return self.forward(state)


class CategoricalDistribution:
    def __init__(self, net):
        self.net = net

    def get_log_probs(self, observation):
        logits = self.net(observation)
        log_prob = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        return log_prob

    def log_prob_action(self, action, observation):
        """Returns log_p(action | observation) assuming action is index"""
        log_prob = self.get_log_probs(observation)
        selected = mx.take_along_axis(log_prob, action[:, None], axis=1).squeeze()
        return selected

    def sample(self, log_probs) -> list[int]:
        probs = mx.exp(log_probs)
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
        if sample:
            log_probs = self.get_log_probs(obs)
            # int(mx.random.categorical(logits).item())
            return self.sample(log_probs)

        # Note that this returns a single action
        # i.e it assumes batch of 1
        logits = self.net(observation)
        return int(mx.argmax(logits).item())

    def update(self, params):
        self.net.update(params)
        return


class GaussianDistribution:
    def __init__(self, net):
        self.net = net

        # Your action space could be a scalar
        # OR a multidim vector (think multiple joints moving)

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


class Buffer:
    """Flat buffer of trajectories indexed by a pointer"""

    def __init__(self):
        self.state: list = []
        self.action: list[int] = []
        self.reward: list[float] = []
        self.value: list[float] = []
        self.reward_to_go: list[float] = []
        self.advantage: list[float] = []
        self.ptr: int = 0
        self.max_size: int = -1

        self.episode_return = []
        self.episode_steps = []

    def update(self, s, a, r, v):
        self.state.append(s)
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


def value_loss_fn(params, state, reward):
    value_fn.update(params)
    r_hat = value_fn(state).flatten()
    return mx.mean((r_hat - reward).square())


def get_minibatches(x, batch_size=64):
    num_samples = len(x)
    drop_samples = num_samples % batch_size if batch_size < num_samples else 0
    num_batches = max(1, num_samples // batch_size)
    x = x[drop_samples:]
    return x.split(num_batches, axis=0)


def train_value_fn(states, rewards):

    state_batches = get_minibatches(states, val_train_batch_size)
    reward_batches = get_minibatches(rewards, val_train_batch_size)

    avg_loss = 0.0
    for state, reward in zip(state_batches, reward_batches):
        # we update value_fn for each (state, reward) pair
        loss, grad_theta = value_grad_fn(value_fn.params, state=state, reward=reward)
        avg_loss += loss.item()

        # SGD step
        grad_theta = clip_grad_norm(grad_theta, grad_clip_value)
        for p, grad in zip(value_fn.params, grad_theta):
            p -= lr_val * grad

    avg_loss /= len(state_batches)
    return avg_loss


def policy_loss_fn(params, old_log_prob, action, state, advantage, clip_ratio=0.2):

    policy.update(params)
    # log_prob of action given state
    log_prob = policy.log_prob_action(action, state)

    # Clipped surrogate objective
    # r_t = probability_ratio
    r_t = mx.exp(log_prob - old_log_prob)
    # TODO: keep track of how many timepoints were clipped
    r_t = mx.minimum(r_t, mx.clip(r_t, 1 - clip_ratio, 1 + clip_ratio))

    loss = r_t * advantage
    return loss.mean()


def train_policy(policy, states, actions, advantages):
    old_log_probs = policy.log_prob_action(actions, states)

    state_batches = get_minibatches(states)
    action_batches = get_minibatches(actions)
    advantage_batches = get_minibatches(advantages)
    old_log_prob_batches = get_minibatches(old_log_probs)

    # We will perform multiple gradient updates
    avg_loss = 0.0
    for state, action, advantage, old_log_prob in zip(
        state_batches,
        action_batches,
        advantage_batches,
        old_log_prob_batches,
    ):
        loss, policy_grad = policy_grad_fn(
            policy.net.params,
            old_log_prob=old_log_prob,
            state=state,
            action=action,
            advantage=advantage,
        )
        avg_loss += loss.item()
        # Gradient ascent step on the parameters sgd style
        policy_grad = clip_grad_norm(policy_grad, grad_clip_value)
        for p, grad in zip(policy.net.params, policy_grad):
            p += lr * grad

    avg_loss /= len(state_batches)

    new_log_probs = policy.log_prob_action(actions, states)
    logratio = old_log_probs - new_log_probs
    approx_kl_schulman = 0.5 * (logratio**2).mean().item()
    logger.info(f"Number of epochs of policy training: {len(state_batches)}")

    return avg_loss, approx_kl_schulman


def clip_grad_norm(grad, grad_clip_value):

    for i in range(len(grad)):
        # gradient clipping via the grad norm
        grad_norm = la.norm(grad[i])
        scale = grad_clip_value / max(grad_clip_value, grad_norm)
        # Note that dividing by norm gives you unit-norm
        # so multiplying by clip rescales norm from 1 -> chosen value
        # This is a no-op when grad_norm < 1 as scale = 1
        grad[i] *= scale

    return grad


policy_grad_fn = mx.value_and_grad(policy_loss_fn)
value_grad_fn = mx.value_and_grad(value_loss_fn)


def run(
    env_name,
    num_parallel_envs,
    num_timesteps_per_epoch,
    hidden_dim,
    init_scale,
    init_scale_final,
    value_init_scale,
    value_init_scale_final,
    policy_lr,
    value_lr,
    grad_clip,
    num_epochs,
    value_batch_size,
    state_normalization,
    discount,
    ema,
    seed,
    log_dir,
    eval_log_dir,
    record_eval_videos,
    **kwargs,  # catchall for unused args
):
    global \
        lr, \
        lr_val, \
        grad_clip_value, \
        val_train_batch_size, \
        discount_factor, \
        ema_factor, \
        policy, \
        value_fn

    lr = policy_lr
    lr_val = value_lr
    grad_clip_value = grad_clip
    val_train_batch_size = value_batch_size
    discount_factor = discount
    ema_factor = ema
    # num_timesteps_per_epoch = 10_000

    if seed is not None:
        mx.random.seed(seed)

    # Create our training environment - a cart with a pole that needs balancing
    env = gym.make_vec(
        env_name,
        num_envs=num_parallel_envs,
        vectorization_mode="sync",
        # max_episode_steps=max_episode_steps,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )

    # Reset environment to start a new episode
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)
    # observation: what the agent can "see" - cart position, velocity, pole angle, etc.

    print(f"Observation Space: {env.single_observation_space}")
    print(f"Action Space:{env.single_action_space}")

    # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    obs_shape = env.single_observation_space.shape[0]

    continuous_action_space = env.single_action_space.dtype == np.float32
    if continuous_action_space:
        action_shape = env.single_action_space.shape[0]
    else:
        action_shape = env.single_action_space.n

    # The statsitics will be recorded for the lifetime fo the algorithm
    state_stats = VecStats()
    reward_stats = VecStats()

    # Initialize the networks and start an episode
    net = TinyLinearNet(
        input_dim=obs_shape,
        hidden_dim=hidden_dim,
        output_dim=action_shape,
        init_scale=init_scale,
        init_scale_final=init_scale_final,
        state_stats=state_stats if state_normalization else None,
    )
    value_fn = TinyLinearNet(
        input_dim=obs_shape,
        hidden_dim=hidden_dim,
        output_dim=1,
        init_scale=value_init_scale,
        init_scale_final=value_init_scale_final,
        state_stats=state_stats if state_normalization else None,
    )
    mx.eval(net.params)
    mx.eval(value_fn.params)
    print("================")
    print("Using LinearNet:")
    print(net)
    print("================")

    policy = (
        GaussianDistribution(net)
        if continuous_action_space
        else CategoricalDistribution(net)
    )

    metrics_logger = RLLogger(
        log_dir,
        exp_name=f"{env_name}-ppo",
        dashboard_run_metadata={
            "algorithm": "ppo",
            "env_id": env_name,
            "seed": seed,
        },
        dashboard_hparams={
            "num_parallel_envs": num_parallel_envs,
            "num_timesteps_per_epoch": num_timesteps_per_epoch,
            "hidden_dim": hidden_dim,
            "init_scale": init_scale,
            "init_scale_final": init_scale_final,
            "value_init_scale": value_init_scale,
            "value_init_scale_final": value_init_scale_final,
            "policy_lr": policy_lr,
            "value_lr": value_lr,
            "grad_clip": grad_clip,
            "num_epochs": num_epochs,
            "value_batch_size": value_batch_size,
            "state_normalization": state_normalization,
            "discount": discount,
            "ema": ema,
        },
        dashboard_capabilities={
            "signals": ["step_reward", "cumulative_return", "value_estimate"],
            "signal_semantics": {
                "step_reward": {"unit": "reward"},
                "cumulative_return": {"unit": "return"},
                "value_estimate": {"unit": "return"},
            },
        },
    )
    eval_video_logger = VideoLogger(
        env_name=env_name, exp_folder=f"{eval_log_dir}/{metrics_logger.run_name}"
    )

    eval_signal_semantics = {
        "value_estimate": {"unit": "return"},
    }

    def _eval_value_estimate(obs, action):
        return float(value_fn(obs).item())

    state_min = env.single_observation_space.low
    state_max = env.single_observation_space.high
    print("state_min =", state_min)
    print("state_max =", state_max)

    # First evaluation pass
    global_step = 0
    if record_eval_videos:
        eval_video_logger.record_evaluation(
            policy,
            global_step,
            extra_signal_fns={"value_estimate": _eval_value_estimate},
            signal_semantics=eval_signal_semantics,
        )
        metrics_logger.log_video(
            global_step,
            eval_video_logger.exp_folder,
            eval_video_logger.num_eval_episodes,
        )

    start_time = time.time()

    for epoch in range(num_epochs):
        collected_timesteps = 0
        # Note: If we do not reset every epoch, any unfinished trajectory *resumes*
        # Allowing the model to learn longer horizon tasks
        observation_vec, info = env.reset()

        # Each parallel env will have its own buffer updated
        buffer_vec = [Buffer() for n in range(env.num_envs)]

        while collected_timesteps < num_timesteps_per_epoch:
            # save current state
            state_vec = observation_vec
            for state in state_vec:
                state_stats.update(state)

            # Choose an action using the tiny net (simulates a deterministic policy)
            action_vec = policy.get_action(state_vec, sample=True)

            # Take the action and see what happens
            observation_vec, reward_vec, terminated_vec, truncated_vec, info_vec = (
                env.step(action_vec)
            )

            for reward in reward_vec:
                reward_stats.update(reward)

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            for i in range(env.num_envs):
                global_step += 1
                collected_timesteps += 1
                buffer = buffer_vec[i]
                state = state_vec[i]
                action = action_vec[i]
                observation = observation_vec[i]
                reward = reward_vec[i]
                terminated = terminated_vec[i]
                truncated = truncated_vec[i]

                # reward = normalize(reward, reward_stats)
                value = value_fn.forward(state).item()

                # given state, policy produced action with probability that received reward
                if global_step % 10_000 == 0:
                    print("global_step =", global_step)
                    print(
                        state, observation, reward, terminated, truncated, action, value
                    )
                    print(
                        "state stats: mean =",
                        state_stats.mean,
                        "var =",
                        state_stats.var,
                    )

                buffer.update(state, action, reward, value)

                episode_over = terminated or truncated
                if terminated or truncated:
                    final_observation = (
                        normalize(info_vec["final_obs"][i], state_stats)
                        if state_normalization
                        else info_vec["final_obs"][i]
                    )
                    terminal_value = (
                        0 if terminated else value_fn.forward(final_observation).item()
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
            observation = (
                normalize(observation_vec[i], state_stats)
                if state_normalization
                else observation_vec[i]
            )
            buffer = buffer_vec[i]
            episode_over = terminated or truncated
            if not episode_over:
                terminal_value = value_fn.forward(observation).item()
                buffer.complete(terminal_value)
            # else we would have completed it in the while-loop

        # Aggregate batch of trajectories across environments
        # N x obs_space
        state_batch = mx.stack(
            [mx.asarray(s) for buffer in buffer_vec for s in buffer.state], axis=0
        )

        # N x 1
        action_batch = mx.concatenate(
            [mx.asarray(buffer.action, dtype=mx.int8) for buffer in buffer_vec],
            axis=0,
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

        # Update V(state) first so that we have good approx for policy grad
        # we will still use slightly stale value estimates to train the policy
        if epoch == 0:
            continue

        policy_loss, approx_kl_schulman = train_policy(
            policy, state_batch, action_batch, advantage_batch
        )

        # avg_grad_norms = mx.array([la.norm(p) for p in policy_grad]).mean()
        # avg_grad_norms /= num_trajectories

        episode_returns = mx.concatenate(
            [mx.array(buffer.episode_return) for buffer in buffer_vec], axis=0
        )
        episode_steps = mx.concatenate(
            [mx.array(buffer.episode_steps) for buffer in buffer_vec], axis=0
        )

        avg_num_steps = episode_steps.mean().item()

        train_metrics = {
            "policy_loss": policy_loss,
            # "policy_grad_norm": avg_grad_norms.item(),
            "value_loss": value_fn_loss,
            "approx_kl": approx_kl_schulman,
        }

        metrics_logger.log_train_metrics(global_step, train_metrics)
        metrics_logger.log_speed(
            global_step, steps_done=global_step, start_time=start_time
        )

        logger.info(f"======= Epoch {epoch} ======= ")
        logger.info(f"Avg Steps: {avg_num_steps:.1f}")
        logger.info(
            f"Mean Return: {episode_returns.mean().item():.2f} +/- {episode_returns.std().item():.2f}"
        )
        logger.info(
            f"Mean Reward-to-Go: {reward_to_go_batch.mean().item():.2f} +/- {reward_to_go_batch.std().item():.2f}"
        )
        logger.info(f"Approx KL: {approx_kl_schulman:.4f}")
        logger.info(f"Value Loss: {value_fn_loss:.2f}")
        logger.info(f"Policy Loss: {policy_loss:.2f}")
        # logger.info(f"Avg Policy Gradient Norm: {avg_grad_norms:.2f}")

        if epoch % 5 == 0 and record_eval_videos:
            eval_video_logger.record_evaluation(
                policy,
                global_step,
                extra_signal_fns={"value_estimate": _eval_value_estimate},
                signal_semantics=eval_signal_semantics,
            )
            metrics_logger.log_video(
                global_step,
                eval_video_logger.exp_folder,
                eval_video_logger.num_eval_episodes,
            )

    env.close()

    # Final evaluation pass
    if record_eval_videos:
        eval_video_logger.record_evaluation(
            policy,
            global_step,
            extra_signal_fns={"value_estimate": _eval_value_estimate},
            signal_semantics=eval_signal_semantics,
        )
        metrics_logger.log_video(
            global_step,
            eval_video_logger.exp_folder,
            eval_video_logger.num_eval_episodes,
        )

    metrics_logger.close()
