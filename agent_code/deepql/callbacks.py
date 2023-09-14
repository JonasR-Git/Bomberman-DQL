from .agent import DQNAgent
from .preprocess import preprocess
import os
import numpy as np

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

def setup(self):
    model_path = "model.h5"

    if os.path.isfile(model_path):
        self.logger.info("Loading model from saved state.")
        self.dqn_agent = DQNAgent(1472, len(ACTIONS))
        self.dqn_agent.load(model_path)
    else:
        self.logger.info("Setting up model from scratch.")
        self.dqn_agent = DQNAgent(1472, len(ACTIONS))

def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS)
    state = preprocess(game_state)
    return self.dqn_agent.act(state)
