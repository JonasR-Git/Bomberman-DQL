import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from .rewards import reward_from_events
from .agent import DQNAgent
from typing import List
import sys
import psutil
import objgraph


print(tf.test.is_built_with_cuda())

# Global variables
episode_counter = 0
game_score = 0
game_score_arr = deque(maxlen=1000)
highest_score = -np.inf
losses = deque(maxlen=1000)
moves_last_50_games = deque(maxlen=50)
current_moves_counter = 0
coins_collected = 0
coins_collected_round = deque(maxlen=30)
invalid_moves = 0
invalid_moves_last_50_games = deque(maxlen=50)
bombs_placed = 0
bombs_placed_last_50_games = deque(maxlen=50)
batch_size = 64

def setup_training(self):
    self.episode_counter = episode_counter
    self.game_score = game_score
    self.game_score_arr = game_score_arr
    self.highest_score = highest_score
    self.losses = losses
    self.moves_last_50_games = moves_last_50_games
    self.current_moves_counter = current_moves_counter
    self.coins_collected = coins_collected
    self.coins_collected_round = coins_collected_round
    self.invalid_moves = invalid_moves
    self.invalid_moves_last_50_games = invalid_moves_last_50_games
    self.bombs_placed = bombs_placed
    self.bombs_placed_last_50_games = bombs_placed_last_50_games

def replay(self, batch_size):
    history = self.dqn_agent.replay(batch_size)

    if history is not None:
        self.losses.append(history.history['loss'][-1])
    else:
        print("Warning: History object is None")

    return history

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(self, events)
    self.game_score += reward
    if "COIN_COLLECTED" in events:
        self.coins_collected += 1
    if "INVALID_ACTION" in events:
        self.invalid_moves += 1
    if "BOMB_DROPPED" in events:
        self.bombs_placed += 1

    self.dqn_agent.remember(old_game_state, self_action, reward, new_game_state)
    self.current_moves_counter += 1

def end_of_round(self, last_game_state, last_action, events):
    objgraph.show_backrefs(objgraph.by_type('list'), max_depth=5, filename='list_backrefs.png')



    '''print("-------------")
    print("MEMORY USAGE")
    print("-------------")
    
    # Create a dictionary to map variable names to their values
    variable_dict = {
        'episode_counter': episode_counter, 
        'game_score': game_score, 
        'game_score_arr': game_score_arr, 
        'highest_score': highest_score, 
        'losses': losses, 
        'moves_last_50_games': moves_last_50_games,
        'current_moves_counter': current_moves_counter, 
        'coins_collected': coins_collected, 
        'coins_collected_round': coins_collected_round, 
        'invalid_moves': invalid_moves, 
        'invalid_moves_last_50_games': invalid_moves_last_50_games, 
        'bombs_placed': bombs_placed, 
        'bombs_placed_last_50_games': bombs_placed_last_50_games, 
        'batch_size': batch_size, 
        'dqn_agent_memory': self.dqn_agent.memory
    }
    
    for var_name, var_value in variable_dict.items():
        print(f"Size of {var_name}: {sys.getsizeof(var_value)} bytes")
    
    print("-------------")'''

    reward = reward_from_events(self, events)
    self.game_score += reward
    if "COIN_COLLECTED" in events:
        self.coins_collected += 1
    if "INVALID_ACTION" in events:
        self.invalid_moves += 1
    if "BOMB_DROPPED" in events:
        self.bombs_placed += 1

    # Reset statistics for next round
    self.coins_collected_round.append(self.coins_collected)
    self.coins_collected = 0
    self.moves_last_50_games.append(self.current_moves_counter)
    self.current_moves_counter = 0
    self.invalid_moves_last_50_games.append(self.invalid_moves)
    self.invalid_moves = 0
    self.bombs_placed_last_50_games.append(self.bombs_placed)
    self.bombs_placed = 0

    self.dqn_agent.decay_epsilon()
    self.dqn_agent.remember(last_game_state, last_action, reward, None)
    self.game_score_arr.append(self.game_score)
    self.highest_score = max(self.game_score, self.highest_score)
    self.game_score = 0

    if len(self.dqn_agent.memory) >= batch_size:
        self.dqn_agent.replay(batch_size)

    self.episode_counter += 1
    print("-------------")
    print("END ROUND")
    print("-------------")

    if not os.path.exists('network_parameters'):
        os.makedirs('network_parameters')
    if (self.episode_counter - 100) % 400 == 0:
        print("Saved:", self.episode_counter)
        self.dqn_agent.save(os.path.join('network_parameters', f'save_after_{self.episode_counter}_iterations'))

    if self.episode_counter % 50 == 0:
        self.dqn_agent.clean_up_memory()

    if self.episode_counter % 250 == 0:
        print("Stats saved")
        write_stats_to_file(self.game_score_arr, self.moves_last_50_games,
                            self.invalid_moves_last_50_games, self.bombs_placed_last_50_games, 
                            self.episode_counter, self.highest_score, self.dqn_agent.epsilon)
        plot_scores(self.game_score_arr)

def write_stats_to_file(game_score_arr, moves_last_50_games, 
                        invalid_moves_last_50_games, bombs_placed_last_50_games, 
                        episode_counter, highest_score, epsilon, filename="training_stats.txt"):
    with open(filename, "a") as f: 
        average_score = np.mean(list(game_score_arr)[-200:])
        average_loss = np.mean(list(losses)[-50:])
        average_moves_last_50 = np.mean(list(moves_last_50_games))
        f.write(f"--- After {episode_counter} episodes ---\n")
        f.write(f"Average Score over the last 200 episodes: {average_score:.2f}\n")
        f.write(f"Average Loss over the last 50 episodes: {average_loss:.2f}\n")
        f.write(f"Highest Score so far: {highest_score:.2f}\n")
        f.write(f"Current Epsilon: {epsilon:.4f}\n")
        f.write(f"Average Moves over the last 50 games: {average_moves_last_50:.2f}\n")
        f.write(f"Invalid_moves_last_50_games: {np.mean(list(invalid_moves_last_50_games)):.2f}\n")
        f.write(f"bombs_placed_last_50_games: {np.mean(list(bombs_placed_last_50_games)):.2f}\n")
        f.write("------------------------------------------\n")

    with open("scores.txt", "a") as f: 
        for i in range(-30, 0):
            if i >= -len(game_score_arr): 
                f.write(f"Episode: {episode_counter + i + 1}, Score: {game_score_arr[i]}, Steps: {moves_last_50_games[i]}, Coins: {coins_collected_round[i]}, Invalid Moves: {invalid_moves_last_50_games[i]}, Bombs Dropped: {bombs_placed_last_50_games[i]}\n")

def plot_scores(game_score_arr):
    plt.figure(figsize=(10,5))
    plt.plot(game_score_arr, label="Scores over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("scores_plot.png")




