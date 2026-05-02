from dataclasses import dataclass
from itertools import accumulate

import gymnasium as gym
import mlx.core as mx
from mlx.core import linalg as la
from rich.pretty import pprint
from loguru import logger

# mx.random.seed(4321)

# Create our training environment - a cart with a pole that needs balancing
# env = gym.make_vec("CartPole-v1", num_envs=2, max_episode_steps=200)
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1", )

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
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=2):
        self.W1 = mx.random.normal((input_dim, hidden_dim)) * 0.1
        self.b1 = mx.zeros(hidden_dim)
        self.W2 = mx.random.normal((hidden_dim, output_dim)) * 0.1
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
        '''Returns log_p(action | observation) assuming action is index'''
        logits = self.net.forward(observation)
        log_prob = logits - mx.logsumexp(logits, axis=1, keepdims=True)
        # print("Action:", action.shape, "Observation:", observation.shape)
        idxs = mx.arange(len(observation))
        # print("Logits:", logits.shape, "indexed Log probs:", log_prob[idxs, action].shape)
        # probs = mx.exp(log_softmax)
        # return log_prob[idxs, action]
        #
        selected = mx.take_along_axis(log_prob, action[:, None], axis=1).squeeze()
        # print("selected:", selected.shape)
        # print(log_prob[idxs, action][:4], selected[:4])
        return selected


    def sample(self, logits):
        probs = mx.exp(logits - mx.logsumexp(logits))
        cdf = mx.cumsum(probs)
        random_val = mx.random.uniform(0, 1)

        # Binary search the index (or just linear for small logits)
        i = 0
        while random_val > cdf[i]:
            i+=1

        return i

    def get_action(self, observation, sample=True):
        """Sample action for training, argmax action for evaluation."""
        obs = mx.asarray(observation, dtype=mx.float32)
        logits = self.net.forward(obs)
        if sample:
            # int(mx.random.categorical(logits).item())
            return self.sample(logits)
        return int(mx.argmax(logits).item())

# @dataclass
class Buffer:
    '''Flat buffer of trajectories indexed by a pointer'''

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

    def complete(self, terminal_value=0):
        # Grab all recorded values and rewards
        trajectory_values = self.value[self.ptr:]
        rewards = self.reward[self.ptr:]

        v_st = trajectory_values
        # when state terminated due to death we add a zero reward
        # otherwise this should be the value_fn(state)
        v_st_next = v_st[1:] + [terminal_value]

        v_st = mx.array(v_st)
        v_st_next = mx.array(v_st_next)
        rewards = mx.array(rewards)

        td_residuals = rewards + gamma(1) * v_st_next - v_st
        td_residuals = td_residuals.tolist()[::-1]
        # advantage = td_residual + ema_factor * gamma(1) * future_t_advantage
        advantage = accumulate(
            td_residuals,
            lambda future_adv, td: td + ema_factor * gamma(1) * future_adv,
        )
        advantage = list(reversed(list(advantage)))

        rewards = rewards.tolist()[::-1]
        reward_to_go = []
        # r[1] =                gamma(0)r[1] + gamma(1)r[2] + gamma(2)r[3]
        # r[0] = gamma(0)r[0] + gamma(1)r[1] + gamma(2)r[2] + gamma(3)r[3]
        reward_to_go = accumulate(
            rewards, lambda r_sum, rt: rt + gamma(1) * r_sum,
        )
        reward_to_go = list(reversed(list(reward_to_go)))

        assert reward_to_go[-1] == rewards[-1], "Final timepoint should match"

        self.advantage.extend(advantage)
        self.reward_to_go.extend(reward_to_go)
        self.episode_return.append(sum(rewards))
        self.episode_steps.append(len(rewards))

        assert len(self.reward) == len(self.advantage), print(f"{len(self.reward)}, {len(self.advantage)}")
        assert len(self.state) == len(self.reward_to_go)


        self.ptr += len(rewards)



# Initialize the networks and start an episode
net = TinyLinearNet()
value_fn = TinyLinearNet(input_dim=4, output_dim=1)
mx.eval(net.params)
mx.eval(value_fn.params)
print("================")
print("Using LinearNet:")
print(net)
print("================")

lr = 0.01
lr_val = 0.001
num_epochs = 20
num_trajectories = 128
policy = CategoricalDistribution(net)
discount_factor = 0.99
ema_factor = 0.96
gamma = lambda t: discount_factor ** t

# Setting the params in net allows the gradients
# to be recorded by autograd
def loss_fn(params, action, state):
    policy.net.update(params)
    return policy.log_prob_action(action, state)

def vec_loss_fn(params, action, state, advantage):
    policy.net.update(params)
    log_prob_of_policy  = policy.log_prob_action(action, state)
    # \sum (\grad theta) * r <==> \grad \sum (r * theta)
    log_prob_of_policy = log_prob_of_policy * advantage
    return log_prob_of_policy.sum()

def value_loss_fn(params, state, reward):
    value_fn.update(params)
    r_hat = value_fn.forward(state).flatten()
    return mx.mean((r_hat - reward).square())


def train_value_fn():

    batch_size = 32
    step = 0

    drop_samples = len(buffer.state) % batch_size if batch_size < len(buffer.state) else 0
    num_batches = max(1, len(buffer.state) // batch_size)

    states = mx.stack([mx.asarray(s) for s in buffer.state[drop_samples:]], axis=0)
    rewards = mx.stack([mx.asarray(s) for s in buffer.reward_to_go[drop_samples:]], axis=0)

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
policy_grad_fn = mx.value_and_grad(loss_fn)

value_grad_fn = mx.value_and_grad(value_loss_fn)


# First evaluation pass
observation, info = eval_env.reset()
episode_over = False
initial_reward = 0
while not episode_over:
    action = policy.get_action(observation, sample=False)
    observation, reward, terminated, truncated, info = eval_env.step(action)
    initial_reward += reward
    episode_over = terminated or truncated
logger.info(f"Initial reward: {initial_reward}")

for epoch in range(num_epochs):

    batch_trajectories = []
    episode_returns = []
    steps_per_trajectory = []
    buffer = Buffer()

    for i in range(num_trajectories):
        observation, info = env.reset()
        total_reward = 0
        trajectory = []
        episode_over = False

        while not episode_over:
            # save current state
            state = observation

            # Choose an action using the tiny net (simulates a deterministic policy)
            action = policy.get_action(observation)
            # log_prob_action = policy.log_prob_action(action, observation)

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)

            if terminated:
                value = 0
            else:
                value = value_fn.forward(state).item()

            # given state, policy produced action with probability that received reward
            buffer.update(state, observation, action, reward, value)

            total_reward += reward
            episode_over = terminated or truncated


        buffer.complete()

    assert len(buffer.episode_return) == num_trajectories

    # Update V(state) first so that we have good approx for policy grad
    # FIXME: currently we use stale value estimates to train the policy
    value_fn_loss = train_value_fn()
    logger.info(f"Value Loss: {value_fn_loss}")

    # Aggregate batch of trajectories

    state_batch = mx.stack([mx.asarray(s) for s in buffer.state], axis=0)
    action_batch =  mx.array(buffer.action)
    advantage_batch = mx.array(buffer.advantage)
    log_probs, policy_grad = vec_policy_grad_fn(policy.net.params, action=action_batch,
        state=state_batch, advantage=advantage_batch)

    avg_grad_norms = mx.array([la.norm(p) for p in policy_grad])
    avg_grad_norms /= num_trajectories

    episode_returns = mx.array(buffer.episode_return)
    episode_steps = mx.array(buffer.episode_steps)
    mu = episode_returns.mean().item()
    std = episode_returns.std().item()
    avg_num_steps = episode_steps.mean().item()

    logger.info(
        f"Epoch {epoch+1}: Avg Steps: {avg_num_steps:.1f} - Mean Return: {mu:.2f} +/- {std:.2f} - Policy Grad Norm: {mx.mean(avg_grad_norms):.3f}"
    )

    n = num_trajectories # max(batch_steps, 1)

    for i in range(len(policy_grad)):
        # Approximate expectation via sample mean over sampled timesteps.
        policy_grad[i] /= n

        # gradient clipping via the grad norm
        # grad_norm = la.norm(policy_grad[i])
        # scale = clip_value / max(clip_value, grad_norm)
        # Note that dividing by norm gives you unit-norm
        # so multiplying by clip rescales norm from 1 -> chosen value
        # This is a no-op when grad_norm < 1 as scale = 1
        # policy_grad[i] *= scale

    # One gradient ascent step on the parameters
    for p, grad in zip(policy.net.params, policy_grad):
        p += lr * grad

    # print(f"Epoch {epoch + 1} complete...")

env.close()

# Final evaluation pass
observation, info = eval_env.reset()
episode_over = False
total_reward = 0
while not episode_over:
    action = policy.get_action(observation, sample=False)
    observation, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(terminated, truncated, info)
print(f"Episode finished! Total reward: {total_reward} - Initial reward was {initial_reward}")
eval_env.close()
