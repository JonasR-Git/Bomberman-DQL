import numpy as np
from collections import deque
from .rewards import reward_from_events
from .agent import DQNAgent
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from .preprocess import preprocess

print(tf.test.is_built_with_cuda())

class Training:

    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.episode_counter = 0
        self.game_score = 0
        self.game_score_arr = []
        self.highest_score = -np.inf
        self.losses = []
        self.moves_last_50_games = deque(maxlen=50)
        self.current_moves_counter = 0
        self.coins_collected = 0
        self.coins_collected_round = deque(maxlen=30)
        self.invalid_moves = 0
        self.invalid_moves_last_50_games = deque(maxlen=50)
        self.bombs_placed = 0
        self.bombs_placed_last_50_games = deque(maxlen=50)

    def remember(self, state, action, reward, next_state):
        self.agent.remember(state, action, reward, next_state)

    def replay(self, batch_size):
        history = self.agent.replay(batch_size)
        
        # Store the model's loss after training
        if history is not None:
            self.losses.append(history.history['loss'][0])
        else:
            print("Warning: History object is None")

        return history

    def setup_training(self):
        pass

    def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
        # Reward management
        reward = reward_from_events(self, events)
        self.game_score += reward
        if "COIN_COLLECTED" in events:
            self.coins_collected += 1
        if "INVALID_ACTION" in events:
            self.invalid_moves += 1
        if "BOMB_DROPPED" in events:
            self.bombs_placed += 1

        # Remember this step
        self.remember(old_game_state, self_action, reward, new_game_state)
        self.current_moves_counter += 1

    def end_of_round(self, last_game_state, last_action, events):
        self.coins_collected_round.append(self.coins_collected)
        self.coins_collected = 0
        self.agent.decay_epsilon()
        self.moves_last_50_games.append(self.current_moves_counter)
        self.current_moves_counter = 0
        self.invalid_moves_last_50_games.append(self.invalid_moves)
        self.invalid_moves = 0
        self.bombs_placed_last_50_games.append(self.bombs_placed)
        self.bombs_placed = 0
        reward = reward_from_events(self, events)
        self.game_score += reward
        if "COIN_COLLECTED" in events:
            self.coins_collected += 1
        if "INVALID_ACTION" in events:
            self.invalid_moves += 1
        if "BOMB_DROPPED" in events:
            self.bombs_placed += 1
        # Reset game score for the next episode
        self.remember(last_game_state, last_action, reward, None)
        self.game_score_arr.append(self.game_score)
        self.highest_score = max(self.game_score, self.highest_score)
        self.game_score = 0

        # Train the agent if memory is sufficiently populated
        if len(self.agent.memory) >= 32:
            self.replay(32)
        
        # Update episode counter
        self.episode_counter += 1
        
        # Save agent's state periodically
        if (self.episode_counter - 100) % 200 == 0:
            if not os.path.exists('network_parameters'):
                os.makedirs('network_parameters')
            self.save_parameters('last_save')
            self.save_parameters(f'save_after_{self.episode_counter}_iterations')

        if self.episode_counter % 20 == 0:
            print("Stats saved")
            self.write_stats_to_file()
            self.plot_scores()

    def save_parameters(self, filename):
        self.agent.model.save(os.path.join('network_parameters', f'{filename}.h5'))

    def write_stats_to_file(self, filename="training_stats.txt"):
        with open(filename, "a") as f: 
            average_score = np.mean(self.game_score_arr[-100:])
            average_loss = np.mean(self.losses[-100:])
            average_moves_last_50 = np.mean(self.moves_last_50_games)
            f.write(f"--- After {self.episode_counter} episodes ---\n")
            f.write(f"Average Score over the last 100 episodes: {average_score:.2f}\n")
            f.write(f"Average Loss over the last 100 episodes: {average_loss:.2f}\n")
            f.write(f"Highest Score so far: {self.highest_score:.2f}\n")
            f.write(f"Current Epsilon: {self.agent.epsilon:.4f}\n")
            f.write(f"Average Moves over the last 50 games: {average_moves_last_50:.2f}\n")
            f.write("------------------------------------------\n")

        with open("scores.txt", "a") as f: 
            for i in range(-30, 0):
                if i >= -len(self.game_score_arr): 
                    f.write(f"Episode: {self.episode_counter + i + 1}, Score: {self.game_score_arr[i]}, Steps: {self.moves_last_50_games[i]}, Coins: {self.coins_collected_round[i]}, Invalid Moves: {self.invalid_moves_last_50_games[i]}, Bombs Dropped: {self.bombs_placed_last_50_games[i]}\n")

    def plot_scores(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.game_score_arr, label="Scores over Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("scores_plot.png")  # Save the figure

# Initialize training
training = Training(1472, 6)  # You need to specify the state and action sizes

def setup_training(self):
    training.setup_training()

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    training.game_events_occurred(old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state, last_action, events):
    training.end_of_round(last_game_state, last_action, events)
