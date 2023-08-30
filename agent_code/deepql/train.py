import numpy as np
from collections import deque
from .rewards import reward_from_events
from .agent import DQNAgent
import tensorflow as tf
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Training:

    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.experience_buffer = deque(maxlen=10000)
        self.episode_counter = 0
        self.total_episodes = 20000  # specify the total number of episodes
        self.game_score = 0
        self.game_score_arr = []

    def remember(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.experience_buffer, min(len(self.experience_buffer), batch_size))
        for state, action, reward, next_state, done in minibatch:
            self.agent.remember(state, action, reward, next_state, done)
        self.agent.replay(batch_size)

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
        if len(self.experience_buffer) > 100:
            self.replay(32)
        
        # update episode counter and game score
        self.episode_counter += 1
        self.game_score += reward_from_events(self, events)
        self.game_score_arr.append(self.game_score)
        
        # save network parameters and game scores periodically
        if self.episode_counter % (self.total_episodes // 100) == 0:
            self.save_parameters('last_save')
            self.save_parameters(f'save_after_{self.episode_counter}_iterations')
            np.savetxt(f'game_score_{self.episode_counter}.txt', self.game_score_arr)
            
    def save_parameters(self, filename):
        self.agent.model.save(os.path.join('network_parameters', f'{filename}.h5'))


# Initialize training
training = Training(1183, 6)  # You need to specify the state and action sizes

def setup_training(self):
    training.setup_training()

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    training.game_events_occurred(old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state, last_action, events):
    training.end_of_round(last_game_state, last_action, events)
