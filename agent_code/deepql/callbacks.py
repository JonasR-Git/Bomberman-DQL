from .agent import DQNAgent
from .preprocess import preprocess
import os
import numpy as np

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

def setup(self):
    self.dqn_agent = DQNAgent(1472, len(ACTIONS))

def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)
    state = preprocess(game_state)
    return self.dqn_agent.act(state)
