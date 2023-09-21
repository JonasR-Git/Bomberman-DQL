from .rewards import reward_from_events
import os

def setup_training(self):
    self.episode_counter = 0
    self.batch_size = 512

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    reward = reward_from_events(self, events)
    # Collecting training data and filling the experience buffer.
    self.dqn_agent.remember(old_game_state, self_action, reward, new_game_state)

def end_of_round(self, last_game_state, last_action, events):
    # Check if enough data to start a replay
    if len(self.dqn_agent.memory) >= self.batch_size:
        self.dqn_agent.replay(self.batch_size)

    # Decay the epsilon (for epsilon-greedy strategy)
    self.dqn_agent.decay_epsilon()

    # Increase episode counter
    self.episode_counter += 1

    # Save the agent state periodically
    if not os.path.exists('network_parameters'):
        os.makedirs('network_parameters')
    if (self.episode_counter - 100) % 400 == 0:
        self.dqn_agent.save(os.path.join('network_parameters', f'save_after_{self.episode_counter}_iterations'))