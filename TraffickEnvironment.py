import numpy as np
import gym
from gym import spaces


class TrafficEnvironment(gym.Env):
    """
    Custom Environment for simulating traffic signal optimization at a single intersection.
    """

    def __init__(self):
        super(TrafficEnvironment, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 2 actions: Change signal to Red or Green
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)  # Cars in two lanes

        # Initialize environment state
        self.reset()

    def reset(self):
        # Initialize the state (number of cars in lane 1 and lane 2)
        self.state = np.random.randint(0, 50, size=(2,))
        self.time = 0  # Reset time
        return self.state

    def step(self, action):
        """
        Perform one step in the environment based on the action.
        """
        lane_1, lane_2 = self.state

        if action == 0:  # Keep signal Green for Lane 1
            lane_1 = max(0, lane_1 - np.random.randint(5, 15))  # Reduce cars in Lane 1
            lane_2 = min(100, lane_2 + np.random.randint(1, 10))  # Increase cars in Lane 2
        elif action == 1:  # Keep signal Green for Lane 2
            lane_2 = max(0, lane_2 - np.random.randint(5, 15))  # Reduce cars in Lane 2
            lane_1 = min(100, lane_1 + np.random.randint(1, 10))  # Increase cars in Lane 1

        # Update state and time
        self.state = np.array([lane_1, lane_2])
        self.time += 1

        # Calculate reward: Negative sum of cars in both lanes (minimize congestion)
        reward = -np.sum(self.state)

        # Define if the episode is done (after 100 time steps)
        done = self.time >= 100
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Time: {self.time}, Lane 1: {self.state[0]} cars, Lane 2: {self.state[1]} cars")


# Simple Q-Learning Agent
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((101, 101, env.action_space.n))  # Q-Table for state-action pairs
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # Exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])
        td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        self.q_table[state[0], state[1], action] += self.alpha * (td_target - self.q_table[state[0], state[1], action])


# Training the RL Agent
env = TrafficEnvironment()
agent = QLearningAgent(env)

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward}")

# Testing the trained agent
state = env.reset()
env.render()
for _ in range(20):
    action = agent.choose_action(state)
    state, _, _, _ = env.step(action)
    env.render()
