from .agent import DQNAgent
from .preprocess import preprocess

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

# callbacks.py
def setup(self):
    self.model = DQNAgent(1183, 6)

def act(self, game_state: dict):
    state = preprocess(game_state)
    return ACTIONS[self.model.act(state)]