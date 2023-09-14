import numpy as np
from collections import deque
from .rewards import reward_from_events
from .agent import DQNAgent
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from .preprocess import preprocess


class Training:

    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.episode_counter = 0
        self.game_score = 0
        self.game_score_arr = []
        self.highest_score = -np.inf
        self.losses = []

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

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
        # collect training data and fill experience buffer
        state = old_game_state
        next_state = new_game_state
        action = self_action
        reward = reward_from_events(self, events)
        done = False

        self.remember(state, action, reward, next_state, done)

    def end_of_round(self, last_game_state, last_action, events):
        self.agent.decay_epsilon()

        # Get reward from the current round
        round_reward = reward_from_events(self, events)
        
        # Update the game score for this episode
        self.game_score += round_reward
        
        # If the episode has finished
        self.game_score_arr.append(self.game_score)
        self.highest_score = max(self.game_score, self.highest_score)
        self.game_score = 0   # Reset game_score for the next episode

        if len(self.agent.memory) > 512:  
            batch_size = min(len(self.agent.memory), 256)  
            self.replay(batch_size)
        
        # update episode counter and game score
        self.episode_counter += 1
        # save network parameters and game scores periodically
        if self.episode_counter % 200 == 0:
            if not os.path.exists('network_parameters'):
                os.makedirs('network_parameters')
            self.save_parameters('last_save')
            self.save_parameters(f'save_after_{self.episode_counter}_iterations')

        if self.episode_counter % 30 == 0:
            print("Stats saved")
            self.write_stats_to_file()  # Write stats to file
            self.plot_scores()  # Plot the scores
            
    def save_parameters(self, filename):
        self.agent.model.save(os.path.join('network_parameters', f'{filename}.h5'))

    def write_stats_to_file(self, filename="training_stats.txt"):
        with open(filename, "a") as f:  # "a" means append mode, so you won't overwrite previous stats
            f.write(f"--- After {self.episode_counter} episodes ---\n")
            average_score = sum(self.game_score_arr[-100:]) / 100 if len(self.game_score_arr) > 100 else np.mean(self.game_score_arr)
            average_loss = np.mean(self.losses[-100:])
            f.write(f"Average Score over the last 100 episodes: {average_score:.2f}\n")
            f.write(f"Average Loss over the last 100 episodes: {average_loss:.2f}\n")
            f.write(f"Highest Score so far: {self.highest_score:.2f}\n")
            f.write(f"Current Epsilon: {self.agent.epsilon:.4f}\n")
            f.write("------------------------------------------\n")

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