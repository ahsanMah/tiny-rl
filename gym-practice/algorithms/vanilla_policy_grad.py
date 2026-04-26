import gymnasium as gym
import mlx.core as mx
import mlx.nn as nn
from mlx.core import linalg as la
from rich.pretty import pprint

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1",
                # render_mode="human",
                max_episode_steps=5)

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
        log_prob = logits - mx.logsumexp(logits)
        # probs = mx.exp(log_softmax)
        return log_prob[action]

    def get_action(self, observation):
        """Get argmax action from observation (array or mx.array)"""
        obs = mx.asarray(observation, dtype=mx.float32)
        logits = self.net.forward(obs)
        return int(mx.argmax(logits))

# Initialize the network and start an episode
net = TinyLinearNet()
print("================")
print("Using LinearNet:")
print(net)
print("================")

lr = 0.01
episode_over = False
total_reward = 0
trajectory = []
policy = CategoricalDistribution(net)

discount_factor = 0.99
gamma = lambda t: discount_factor ** t

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
    
    # given state, policy produced action with probability that received reward
    trajectory.append([state, action, reward])

    total_reward += reward
    episode_over = terminated or truncated

# Compute loss = \grad_theta log_prob(a | s) * return
# Sum weighted reward within trajectory
# Sum batch of trajectories

# Accumulate rewards into return for the trajectory
discounted_return = sum(gamma(i) * t[-1] for i, t in enumerate(trajectory))
print(f"Discounted Return: {discounted_return}")

def loss_fn(params, action, state):
    policy.net.update(params)
    return policy.log_prob_action(action, state)

policy_grad_fn = mx.value_and_grad(loss_fn)

# policy_grad accumulates internally
policy_grad = [0.0] * len(net.params)
for state, action, reward in trajectory:
    # gradient wrt theta computed
    action = mx.array(action)
    state = mx.array(state)
    log_prob, policy_grad_t = policy_grad_fn(policy.net.params, action=action, state=state)
    
    # grad * return
    policy_grad = [(p + pt) * discounted_return for p,pt in zip(policy_grad, policy_grad_t)]
    
    # Logging only
    grad_norms = mx.array([la.norm(p) for p in policy_grad])
    print(f"Log Prob: {log_prob:.4f} - Policy Grad Norm: {mx.mean(grad_norms):.3f}")

policy_grad = [p * discounted_return for p in policy_grad]

# One gradient ascent step on the parameters
param_norms = [f"{la.norm(p).item():2f}" for p in policy.net.params]
print(f"Before Param Norms: {param_norms}")

for p, grad in zip(policy.net.params, policy_grad):
    p += lr * grad


param_norms = [f"{la.norm(p).item():2f}" for p in policy.net.params]
print(f"After Param Norms: {param_norms}")

print(f"Episode finished! Total reward: {total_reward}")
env.close()
