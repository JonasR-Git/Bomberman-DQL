from .rewards import reward_from_events
import os
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def setup_training(self):
    self.episode_counter = 0
    self.batch_size = 1024
    self.game_score = 0
    self.game_score_arr = []
    self.highest_score = -np.inf
    self.losses = []
    self.moves_last_x_games = deque(maxlen=50)
    self.current_moves_counter = 0
    self.coins_collected = 0
    self.coins_collected_round = deque(maxlen=50)
    self.invalid_moves = 0
    self.invalid_moves_last_x_games = deque(maxlen=50)
    self.bombs_placed = 0
    self.bombs_placed_last_x_games = deque(maxlen=50)

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(self, events)
    self.game_score += reward
    self.current_moves_counter += 1
    if "COIN_COLLECTED" in events:
        self.coins_collected += 1
    if "INVALID_ACTION" in events:
        self.invalid_moves += 1
    if "BOMB_DROPPED" in events:
        self.bombs_placed += 1
    reward = reward_from_events(self, events)
    # Collecting training data and filling the experience buffer.
    self.dqn_agent.remember(old_game_state, self_action, reward, new_game_state)

def end_of_round(self, last_game_state, last_action, events):
    self.coins_collected_round.append(self.coins_collected)
    self.coins_collected = 0
    self.moves_last_x_games.append(self.current_moves_counter)
    self.current_moves_counter = 0
    self.invalid_moves_last_x_games.append(self.invalid_moves)
    self.invalid_moves = 0
    self.bombs_placed_last_x_games.append(self.bombs_placed)
    self.bombs_placed = 0
    reward = reward_from_events(self, events)
    self.game_score += reward
    if "COIN_COLLECTED" in events:
        self.coins_collected += 1
    if "INVALID_ACTION" in events:
        self.invalid_moves += 1
    if "BOMB_DROPPED" in events:
        self.bombs_placed += 1
    self.game_score_arr.append(self.game_score)
    self.highest_score = max(self.game_score, self.highest_score)
    self.game_score = 0
    # Check if enough data to start a replay
    if len(self.dqn_agent.memory) >= self.batch_size and self.episode_counter % 50 == 0:
        self.dqn_agent.replay(self.batch_size)

    # Decay the epsilon (for epsilon-greedy strategy)
    self.dqn_agent.decay_epsilon()

    # Increase episode counter
    self.episode_counter += 1

    # Save the agent state periodically
    if not os.path.exists('network_parameters'):
        os.makedirs('network_parameters')
    if (self.episode_counter - 100) % 50 == 0:
        self.dqn_agent.save(os.path.join('network_parameters', f'save_after_{self.episode_counter}_iterations'))

    if self.episode_counter % 50 == 0:
        print("Stats saved")
        write_stats_to_file(self)
        plot_scores(self)

def write_stats_to_file(self, filename="training_stats.txt"):
    with open(filename, "a") as f: 
        average_score = np.mean(self.game_score_arr[-50:])
        average_loss = np.mean(self.losses[-50:])
        average_moves_last_x = np.mean(self.moves_last_x_games)
        f.write(f"--- After {self.episode_counter} episodes ---\n")
        f.write(f"Average Score over the last 500 episodes: {average_score:.2f}\n")
        f.write(f"Average Loss over the last 500 episodes: {average_loss:.2f}\n")
        f.write(f"Highest Score so far: {self.highest_score:.2f}\n")
        f.write(f"Current Epsilon: {self.dqn_agent.epsilon:.4f}\n")
        f.write(f"Average Moves over the last 500 games: {average_moves_last_x:.2f}\n")
        f.write("------------------------------------------\n")

    with open("scores.txt", "a") as f: 
        for i in range(-50, 0):
            if i >= -len(self.game_score_arr): 
                f.write(f"Episode: {self.episode_counter + i + 1}, Score: {self.game_score_arr[i]}, Steps: {self.moves_last_x_games[i]}, Coins: {self.coins_collected_round[i]}, Invalid Moves: {self.invalid_moves_last_x_games[i]}, Bombs Dropped: {self.bombs_placed_last_x_games[i]}\n")

def moving_average(data, window_size):
    """Return a moving average of `data` using a specified `window_size`."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_scores(self):
    window_size = 100

    if len(self.game_score_arr) < window_size:
        print("Not enough data points to compute smoothed graph. Skipping...")
        return

    # Calculate moving average
    smoothed_scores = moving_average(self.game_score_arr, window_size)
    
    # Due to the nature of moving average, the smoothed_scores will be shorter by window_size - 1
    x = [i for i in range(window_size-1, len(self.game_score_arr))]
    
    # Linear approximation for all original points
    m, b = np.polyfit(range(len(self.game_score_arr)), self.game_score_arr, 1)
    linear_approximation = [m*i + b for i in range(len(self.game_score_arr))]
    
    plt.figure(figsize=(10,5))
    plt.plot(self.game_score_arr, alpha=0.5, label="Original Scores")  # Original scores with reduced opacity
    plt.plot(x, smoothed_scores, label="Smoothed Scores over Episodes", color="red")
    plt.plot(linear_approximation, label="Linear Approximation", color="blue", linestyle="--")  # Linear approximation line
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("smoothed_scores_plot.png")  # Save the figure with a different name to distinguish