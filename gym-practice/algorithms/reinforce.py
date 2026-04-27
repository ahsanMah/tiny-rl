import gymnasium as gym
import mlx.core as mx
from mlx.core import linalg as la
from rich.pretty import pprint

# mx.random.seed(4321)

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", max_episode_steps=500)
eval_env = gym.make("CartPole-v1", render_mode="human")

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

# Initialize the networks and start an episode
net = TinyLinearNet()
value_fn = TinyLinearNet(input_dim=4, output_dim=1)
mx.eval(net.params)
mx.eval(value_fn.params)
print("================")
print("Using LinearNet:")
print(net)
print("================")

lr = 0.002
num_epochs = 10
num_trajectories = 25
policy = CategoricalDistribution(net)
discount_factor = 0.95
gamma = lambda t: discount_factor ** t

# Setting the params in net allows the gradients
# to be recorded by autograd
def loss_fn(params, action, state):
    policy.net.update(params)
    return policy.log_prob_action(action, state)

def value_loss_fn(params, state, reward):
    value_fn.update(params)
    r_hat = value_fn.forward(state)
    return mx.mean((r_hat - reward) ** 2)


def train_value_fn(batch_trajectories):

    grad_theta = [mx.zeros_like(p) for p in value_fn.params]
    avg_loss = 0.0
    step = 0
    for trajectory, rewards in batch_trajectories:
        # we update value_fn for each (state, reward) pair
        for t, r in zip(trajectory, rewards):
            state = t[0]
            loss, _grad_t = value_grad_fn(value_fn.params, state=state, reward=reward)
            avg_loss += loss
            for grad, _grad in zip(grad_theta, _grad_t): 
                grad += _grad

        # SGD step
        for p, grad in zip(value_fn.params, grad_theta):
            p -= lr * grad
    
        avg_loss /= len(trajectory)
        if step % 5 == 0:
            print(f"Value Function - Step: {step} - Loss: {avg_loss:.3f}")
        step += 1

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
print(f"Initial reward: {initial_reward}")

for epoch in range(num_epochs):
    
    batch_trajectories = []
    
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
            
            # given state, policy produced action with probability that received reward
            trajectory.append([state, action, reward])

            total_reward += reward
            episode_over = terminated or truncated

        discounted_rewards = [gamma(i) * t[-1] for i, t in enumerate(trajectory)]
        reward_to_go = mx.array(discounted_rewards)

        # each time point is accumulation of *future* rewards only
        # thus accum in reverse
        for i in range(len(trajectory)-2, -1, -1):
            reward_to_go[i] += reward_to_go[i+1]

        batch_trajectories.append((trajectory, reward_to_go))
    
    # Update V(state) first so that we have good approx for policy grad
    train_value_fn(batch_trajectories)

    # Aggregate batch of trajectories
    policy_grad = [mx.zeros_like(p) for p in net.params]
    batch_steps = 0
    avg_grad_norms = 0.0
    avg_discounted_return = 0.0
    avg_log_prob = 0.0

    for (trajectory, reward_to_go) in batch_trajectories:

        log_prob = 0.0

        for (state, action, _), reward in zip(trajectory, reward_to_go):
            action = mx.array(action)
            state = mx.asarray(state, dtype=mx.float32)

            log_prob_t, policy_grad_t = policy_grad_fn(policy.net.params, action=action, state=state)
            log_prob += float(log_prob_t.item())

            # Correct accumulation: add grad_t weighted by future return.
            policy_grad = [
                p + (pt * reward)
                for p, pt in zip(policy_grad, policy_grad_t)
            ]
            batch_steps += 1

        avg_grad_norms += mx.array([la.norm(p) for p in policy_grad])
        avg_discounted_return += sum(discounted_rewards)
        avg_log_prob += log_prob / len(trajectory)

    avg_grad_norms /= num_trajectories
    avg_discounted_return /= num_trajectories
    avg_log_prob /= num_trajectories

    print(
        f"Avg Discounted Return: {avg_discounted_return:2f} - Log Prob: {avg_log_prob / len(trajectory):.4f} - Policy Grad Norm: {mx.mean(avg_grad_norms):.3f}"
    )

    n = num_trajectories # max(batch_steps, 1) 
    
    for i in range(len(policy_grad)):
        # Approximate expectation via sample mean over sampled timesteps.
        policy_grad[i] /= n
        
        # Normalize the grad norm
        grad_norm = la.norm(policy_grad[i])
        policy_grad[i] /= grad_norm

    # One gradient ascent step on the parameters
    param_norms = [f"{la.norm(p).item():2f}" for p in policy.net.params]
    print(f"Before Param Norms: {param_norms}")

    for p, grad in zip(policy.net.params, policy_grad):
        p += lr * grad
    param_norms = [f"{la.norm(p).item():2f}" for p in policy.net.params]
    print(f"After Param Norms: {param_norms}")
    print(f"Epoch {epoch + 1} complete...")

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