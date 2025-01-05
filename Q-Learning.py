import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the grid world environment
grid_size = 5
start_position = (0, 0)  # (1, 1) in 1-based indexing
goal_position = (4, 4)   # (5, 5) in 1-based indexing
obstacles = [(1, 1), (3, 3)]  # (2, 2) and (4, 4) in 1-based indexing

# Define the action space
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Define the reward function
def get_reward(state):
    if state == goal_position:
        return 100
    elif state in obstacles:
        return -10
    else:
        return 1

# Define the next state function
def get_next_state(state, action):
    next_state = (state[0] + action_dict[action][0], state[1] + action_dict[action][1])
    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
        return next_state
    else:
        return state

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

def train_agent(alpha, gamma, epsilon=0.1, num_episodes=1000):
    Q_table = np.zeros((grid_size, grid_size, len(actions)))
    rewards_history = []
    for episode in range(num_episodes):
        state = start_position
        total_reward = 0
        stepCount = 0
        while state != goal_position:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  # Explore
            else:
                action = actions[np.argmax(Q_table[state[0], state[1]])]  # Exploit

            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])
            td_target = reward + gamma * Q_table[next_state[0], next_state[1], best_next_action]
            td_error = td_target - Q_table[state[0], state[1], actions.index(action)]
            Q_table[state[0], state[1], actions.index(action)] += alpha * td_error

            state = next_state
            total_reward += reward
            stepCount += 1

        rewards_history.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward} in {stepCount} steps")


    return Q_table, rewards_history

Q_table, rewards_history = train_agent(alpha, gamma, epsilon, num_episodes)

# Print the Q-table
print("Q-table after training:")
print(Q_table)

# Function to get the optimal path
def get_optimal_path():
    state = start_position
    path = [state]
    while state != goal_position:
        action = actions[np.argmax(Q_table[state[0], state[1]])]
        state = get_next_state(state, action)
        path.append(state)
    return path

# Get the optimal path
optimal_path = get_optimal_path()
print("Optimal path from start to goal:")
print(optimal_path)

# Evaluate the trained agent's performance
def evaluate_agent(num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state = start_position
        total_reward = 0
        while state != goal_position:
            action = actions[np.argmax(Q_table[state[0], state[1]])]
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    return average_reward

# Evaluate the trained agent
average_reward = evaluate_agent()
print(f"Average total reward over 100 episodes: {average_reward}")

def visualize_policy(Q_table):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((j, 4-i), 1, 1, fill=False))
    
    # Draw obstacles
    for obs in [(1, 1), (3, 3)]:
        ax.add_patch(Rectangle((obs[1], 4-obs[0]), 1, 1, fill=True, color='red', alpha=0.3))
    
    # Draw start and goal
    ax.add_patch(Rectangle((0, 4), 1, 1, fill=True, color='green', alpha=0.3))
    ax.add_patch(Rectangle((4, 0), 1, 1, fill=True, color='blue', alpha=0.3))
    
    # Draw arrows for best actions
    for i in range(5):
        for j in range(5):
            if (i, j) not in [(1, 1), (3, 3)]:
                action = np.argmax(Q_table[i][j])
                if action == 0:    # up
                    dx, dy = 0, 0.3
                elif action == 1:  # down
                    dx, dy = 0, -0.3
                elif action == 2:  # left
                    dx, dy = -0.3, 0
                else:             # right
                    dx, dy = 0.3, 0
                ax.arrow(j + 0.5, 4-i + 0.5, dx, dy, head_width=0.1, color='black')
    
    plt.grid(True)
    plt.title("Learned Policy (Green: Start, Blue: Goal, Red: Obstacles)")
    plt.show()

def plot_rewards(rewards_history):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def plot_q_values(Q_table):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
    actions = ['Up', 'Down', 'Left', 'Right']
    cmap = plt.cm.viridis
    
    for i, (ax, action) in enumerate(zip(axes.flat, actions)):
        q_values = Q_table[:, :, i]
        
        im = ax.imshow(q_values, cmap=cmap)
        ax.set_title(f'Q-values for {action} action')
        ax.set_xlabel('Y coordinate')
        ax.set_ylabel('X coordinate')
        
        # Annotate the cells with Q-value
        for x in range(q_values.shape[0]):
            for y in range(q_values.shape[1]):
                ax.text(y, x, f"{q_values[x, y]:.1f}", ha='center', va='center', color='white' if q_values[x, y] < q_values.max() / 2 else 'black')
    
    # Add a colorbar to the last axis
    cbar = fig.colorbar(im, ax=axes[:, -1], shrink=0.6)
    cbar.set_label('Q-value')
    
    plt.show()

# Visualize results
visualize_policy(Q_table)
plot_rewards(rewards_history)
plot_q_values(Q_table)