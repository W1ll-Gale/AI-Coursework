import numpy as np
import random

# Define the grid world environment
class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.start_position = (0, 0)
        self.goal_position = (4, 4)
        self.obstacles = [(1, 1), (3, 3)]
        self.actions = ['up', 'down', 'left', 'right']
        self.state = self.start_position

    def reset(self):
        self.state = self.start_position
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size - 1, y + 1)

        new_state = (x, y)
        reward = self.get_reward(new_state)
        self.state = new_state
        done = self.state == self.goal_position
        return new_state, reward, done

    def get_reward(self, state):
        if state in self.obstacles:
            return -10
        elif state == self.goal_position:
            return 100
        else:
            return 1

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_index = self.env.actions.index(action)
        predict = self.q_table[x, y, action_index]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action_index] += self.alpha * (target - predict)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Main function to train the Q-learning agent
def main():
    env = GridWorld()
    agent = QLearningAgent(env)
    agent.train(episodes=100)

if __name__ == "__main__":
    main()