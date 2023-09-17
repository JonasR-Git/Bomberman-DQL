from .agent import DQNAgent
from .preprocess import preprocess
import os
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    self.dqn_agent = DQNAgent(324, len(ACTIONS))

def act(self, game_state: dict) -> str:
    state = preprocess(game_state)
    return self.dqn_agent.act(state, game_state)
