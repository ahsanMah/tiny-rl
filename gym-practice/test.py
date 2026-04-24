# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import mlx.core as mx

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
print(f"Supp(lemental info: {info}")
print(f"Action space:{env.action_space}")
print(f"Observation space: {env.observation_space}")

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

    def forward(self, x):
        """x: (4,) -> logits: (2,)"""
        x = mx.maximum(x @ self.W1 + self.b1, 0)  # ReLU
        return x @ self.W2 + self.b2  # Raw logits

    def get_action(self, observation):
        """Get argmax action from observation (array or mx.array)"""
        obs = mx.asarray(observation, dtype=mx.float32)
        logits = self.forward(obs)
        return int(mx.argmax(logits))

# Initialize the network and start an episode
net = TinyLinearNet()
print("================")
print("Using LinearNet:")
print(net)
print("================")

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action using the tiny net (simulates a deterministic policy)
    action = net.get_action(observation)

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()