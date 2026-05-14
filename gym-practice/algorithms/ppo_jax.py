import math
import os
import sys
import time
from itertools import accumulate
from posix import stat
from typing import NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import numpy as np
import optax
from flax import nnx
from jax import jit, vmap
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_utils import RLLogger, VideoLogger

# mx.random.seed(4321)


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

    def complete(self, discount_factor, ema_factor, terminal_value=0.0):
        # Grab all recorded values and rewards
        trajectory_values = self.value[self.ptr :]
        rewards = self.reward[self.ptr :]

        v_st = trajectory_values
        # when state terminated due to death we add a zero reward
        # otherwise this should be the value_fn(state)
        v_st_next = v_st[1:] + [terminal_value]

        v_st = jnp.array(v_st)
        v_st_next = jnp.array(v_st_next)
        reward_vec = jnp.asarray(rewards)

        td_residuals = reward_vec + discount_factor * v_st_next - v_st
        td_residuals = td_residuals[::-1].tolist()
        # advantage = td_residual + ema_factor * discount_factor * future_t_advantage
        advantage = accumulate(
            td_residuals,
            lambda future_adv, td: td + ema_factor * discount_factor * future_adv,
        )
        advantage = list(reversed(list(advantage)))

        # rewards: list = rewards[::-1].tolist()
        reward_to_go = []
        # r[1] =                gamma(0)r[1] + gamma(1)r[2] + gamma(2)r[3]
        # r[0] = gamma(0)r[0] + gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
        reward_to_go = accumulate(
            rewards[::-1], lambda r_sum, rt: rt + discount_factor * r_sum
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


class TinyLinearNet(nnx.Module):
    def __init__(
        self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs, init_scale=1.0
    ):
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.ln = nnx.LayerNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

        # state stats
        self.count = nnx.Variable(jnp.zeros(1, dtype=jnp.int32))
        self.mu = nnx.Variable(jnp.zeros(shape=(1, din)))
        self.m2 = nnx.Variable(jnp.zeros(shape=(1, din)))

    def _normalize(self, x, eps=1e-5):
        self._update_stats(x)
        var = self.m2 / self.count
        return (x - self.mu) / (var + eps) ** 0.5

    def _update_stats(self, x):
        _x = jnp.mean(x, axis=0, keepdims=True)
        prev_mean = self.mu * 1.0

        self.count[...] += x.shape[0]
        self.mu[...] += (_x - self.mu) / self.count
        self.m2[...] += (_x - prev_mean) * (_x - self.mu)

    def __call__(self, x: jax.Array):
        x = self._normalize(x)
        x = nnx.gelu(self.ln(self.linear1(x)))
        return self.linear2(x)


class Policy(NamedTuple):
    net: nnx.Module


class Rollout(NamedTuple):
    """All the info collected in a rollout for PPO"""

    actions: jax.Array
    states: jax.Array
    advantages: jax.Array
    log_probs: jax.Array


def gaussian_sample(mu, log_std, key: jax.Array, shape: tuple = ()) -> jnp.ndarray:
    eps = jax.random.normal(key, shape=shape)
    return mu + jnp.exp(log_std) * eps  # reparameterization trick


def gaussian_log_prob(
    x: jnp.ndarray, mu: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    var = jnp.exp(log_std) ** 2
    normalization_constant = -0.5 * math.log(2 * math.pi)
    log_prob = -((x - mu) ** 2) / (2 * var) - log_std + normalization_constant

    # Sum over independent Gaussians
    log_prob = log_prob.sum(axis=1)
    return log_prob

    # def entropy(dist: Gaussian) -> jnp.ndarray:
    #     return jnp.sum(dist.log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)


def get_random_action(policy: nnx.Module, observation: jax.Array, key: jax.Array):
    mu = policy(observation)
    log_std = jnp.ones_like(mu)
    return gaussian_sample(mu, log_std, key, shape=mu.shape)


def normalize(x, stats, eps=1e-8):
    mu, var = stats.mean, stats.var
    # print(f"State mean: {mu}, var: {var}, count: {state_stats.count}")
    if x.ndim > 1:
        mu = mu[None, :]
        var = var[None, :]
    return (x - mu) / (var + eps) ** 0.5


def get_minibatches(x, batch_size=64):
    num_samples = len(x)
    num_batches = max(1, (num_samples + batch_size - 1) // batch_size)
    # drop_samples = num_samples % batch_size if batch_size < num_samples else 0
    # x = x[drop_samples:]
    # return jnp.split(x, num_batches, axis=0)

    xs = jnp.array_split(x, num_batches)
    xs = jnp.stack(xs)
    return xs


@nnx.jit
def value_train_step(value_fn, optimizer, states: jax.Array, rewards: jax.Array):
    def value_loss_fn(value_fn: nnx.Module, states: jax.Array, rewards: jax.Array):
        return jnp.mean((value_fn(states) - rewards) ** 2)

    loss, grads = nnx.value_and_grad(value_loss_fn)(value_fn, states, rewards)
    optimizer.update(value_fn, grads)  # In place updates.
    return loss


def train_value_fn(value_fn, optimizer, states, rewards, val_train_batch_size=128):
    state_batches = get_minibatches(states, val_train_batch_size)
    reward_batches = get_minibatches(rewards, val_train_batch_size)

    avg_loss = 0.0
    for state, reward in zip(state_batches, reward_batches):
        # we update value_fn for each (state, reward) pair
        loss = value_train_step(value_fn, optimizer, states=state, rewards=reward)
        avg_loss += loss.item()

    avg_loss /= len(state_batches)
    return avg_loss


def policy_log_prob(net: nnx.Module, action: jax.Array, state: jax.Array):
    # log_prob of action given state
    mu = net(state)
    log_std = jnp.ones_like(mu)
    return gaussian_log_prob(action, mu=mu, log_std=log_std)


# The function takes in net, action, states
# We want to use the same net but apply it to a list of batches
# Thus first argument is *not* batched but the other two are batched (at dim = 0)
batched_policy_log_prob = vmap(policy_log_prob, in_axes=[None, 0, 0])


def policy_loss_fn(net: nnx.Module, rollout: Rollout, clip_ratio: float):

    # log_prob of action given state
    # mu = net(rollout.states)
    # log_prob = gaussian_log_prob(rollout.actions, mu=mu, log_std=log_std)

    log_prob = policy_log_prob(net, action=rollout.actions, state=rollout.states)

    # Clipped surrogate objective
    # r_t = probability_ratio
    r_t = jnp.exp(log_prob - rollout.log_probs)
    # TODO: keep track of how many timepoints were clipped
    r_t = jnp.minimum(r_t, jnp.clip(r_t, 1 - clip_ratio, 1 + clip_ratio))

    loss = r_t * rollout.advantages
    return loss.mean()


@nnx.jit
def policy_train_step(net, optimizer, rollout: Rollout, clip_ratio=0.2):
    loss, grads = nnx.value_and_grad(policy_loss_fn)(
        net, rollout=rollout, clip_ratio=clip_ratio
    )
    optimizer.update(net, grads)  # In place updates.

    return loss


def train_policy(policy, optimizer, states, actions, advantages):

    state_batches = get_minibatches(states)
    action_batches = get_minibatches(actions)
    advantage_batches = get_minibatches(advantages)

    old_log_probs = batched_policy_log_prob(policy, action_batches, state_batches)
    old_log_probs = get_minibatches(old_log_probs)

    # We will perform multiple gradient updates
    avg_loss = 0.0
    num_batches = len(state_batches)
    for i in range(num_batches):
        rollout = Rollout(
            actions=action_batches[i],
            states=state_batches[i],
            advantages=advantage_batches[i],
            log_probs=old_log_probs[i],
        )
        avg_loss += policy_train_step(policy, optimizer, rollout=rollout)

    avg_loss /= len(state_batches)

    new_log_probs = batched_policy_log_prob(policy, action_batches, state_batches)
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
    num_trajectories,
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

    lr = policy_lr
    lr_val = value_lr
    grad_clip_value = grad_clip
    val_train_batch_size = value_batch_size
    discount_factor = discount
    ema_factor = ema
    # num_timesteps_per_epoch = 1024  # 10_000

    rng_key = jax.random.key(seed)

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
    policy = TinyLinearNet(
        din=obs_shape,
        dmid=hidden_dim,
        dout=action_shape,
        rngs=nnx.Rngs(seed),
        init_scale=init_scale,
        # init_scale_final=init_scale_final,
        # state_stats=state_stats if state_normalization else None,
    )

    value_fn = TinyLinearNet(
        din=obs_shape,
        dmid=hidden_dim,
        dout=1,
        rngs=nnx.Rngs(seed),
        init_scale=init_scale,
        # init_scale_final=init_scale_final,
        # state_stats=state_stats if state_normalization else None,
    )

    policy_optimizer = nnx.Optimizer(policy, optax.adam(1e-3), wrt=nnx.Param)
    value_optimizer = nnx.Optimizer(value_fn, optax.adam(1e-3), wrt=nnx.Param)

    print("================")
    print("Policy Network:")
    print(policy)
    print("================")

    metrics_logger = RLLogger(log_dir, exp_name=f"{env_name}-ppo-jax")
    eval_video_logger = VideoLogger(
        env_name=env_name, exp_folder=f"{eval_log_dir}/{metrics_logger.run_name}"
    )

    state_min = env.single_observation_space.low
    state_max = env.single_observation_space.high
    print("state_min =", state_min)
    print("state_max =", state_max)

    # First evaluation pass
    global_step = 0
    if record_eval_videos:
        eval_video_logger.record_evaluation(policy, global_step)
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

        terminated_vec, truncated_vec = [], []
        while collected_timesteps < num_timesteps_per_epoch:
            # save current state
            state_vec = observation_vec
            for state in state_vec:
                state_stats.update(state)

            # Choose an action using the tiny net (simulates a deterministic policy)
            rng_key, key = jax.random.split(rng_key)
            action_vec = get_random_action(policy, state_vec, key)

            # Take the action and see what happens
            observation_vec, reward_vec, terminated_vec, truncated_vec, info_vec = (
                env.step(action_vec)
            )

            observation_vec = jnp.asarray(observation_vec)
            reward_vec = jnp.asarray(reward_vec)

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

                value = value_fn(state).flatten().item()

                # given state, policy produced action with probability that received reward
                if global_step % 1_000 == 0:
                    print("global_step =", global_step)
                    print(
                        state, observation, reward, terminated, truncated, action, value
                    )
                    print(
                        "state stats: mean =",
                        value_fn.mu,
                        "var =",
                        value_fn.m2 / value_fn.count,
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
                        0 if terminated else value_fn(final_observation).item()
                    )
                    buffer.complete(discount_factor, ema_factor, terminal_value)
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
                terminal_value = value_fn(observation).item()
                buffer.complete(discount_factor, ema_factor, terminal_value)
            # else we would have completed it in the while-loop

        # Aggregate batch of trajectories across environments
        # concatenate_buffers = lambda x: jnp.concatenate([jnp.asarray()])

        # N x obs_space
        state_batch = jnp.stack(
            [jnp.asarray(s) for buffer in buffer_vec for s in buffer.state], axis=0
        )

        # N x 1
        action_batch = jnp.concatenate(
            [jnp.asarray(buffer.action, dtype=jnp.int8) for buffer in buffer_vec],
            axis=0,
        )
        # N x 1
        advantage_batch = jnp.concatenate(
            [jnp.asarray(buffer.advantage) for buffer in buffer_vec], axis=0
        )

        reward_to_go_batch = jnp.concatenate(
            [jnp.asarray(buffer.reward_to_go) for buffer in buffer_vec], axis=0
        )
        print("state_batch.shape =", state_batch.shape)
        print("reward_to_go_batch.shape =", reward_to_go_batch.shape)
        value_fn_loss = train_value_fn(
            value_fn, value_optimizer, state_batch, reward_to_go_batch
        )

        # Update V(state) first so that we have good approx for policy grad
        # we will still use slightly stale value estimates to train the policy
        if epoch == 0:
            continue

        policy_loss, approx_kl_schulman = train_policy(
            policy, policy_optimizer, state_batch, action_batch, advantage_batch
        )

        # avg_grad_norms = mx.array([la.norm(p) for p in policy_grad]).mean()
        # avg_grad_norms /= num_trajectories

        episode_returns = jnp.concatenate(
            [jnp.array(buffer.episode_return) for buffer in buffer_vec], axis=0
        )
        episode_steps = jnp.concatenate(
            [jnp.array(buffer.episode_steps) for buffer in buffer_vec], axis=0
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

    env.close()

    # Final evaluation pass
    if record_eval_videos:
        eval_video_logger.record_evaluation(policy, global_step)
        metrics_logger.log_video(
            global_step,
            eval_video_logger.exp_folder,
            eval_video_logger.num_eval_episodes,
        )

    metrics_logger.close()
