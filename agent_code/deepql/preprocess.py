import numpy as np
from .settings import MAX_STEPS, EXPLOSION_TIMER, ROWS, COLS, BOMB_TIMER, BOMB_POWER, MAX_AGENTS

def preprocess(game_state):
    # Normalize 'round' and 'step'
    round = game_state['round']
    step = game_state['step'] / MAX_STEPS

    # Flatten 'field' and 'explosion_map'
    field = game_state['field'].flatten() / 1.0
    explosion_map = game_state['explosion_map'].flatten() / EXPLOSION_TIMER

    # Process 'bombs'
    bomb_map = np.zeros((ROWS, COLS))
    for (x, y), t in game_state['bombs']:
        bomb_map[y, x] = t / BOMB_TIMER
    bomb_map = bomb_map.flatten()

    # Process 'coins'
    coin_map = np.zeros((ROWS, COLS))
    for (x, y) in game_state['coins']:
        coin_map[y, x] = 1
    coin_map = coin_map.flatten()

    # Process 'self'
    self = np.zeros(5)
    _, score, bomb_available, (x, y) = game_state['self']
    self[0] = score / 100.0  # Assume max score 100
    self[1] = 1 if bomb_available else 0
    self[2] = x / COLS
    self[3] = y / ROWS

    # Process 'others'
    others = np.zeros(5 * MAX_AGENTS)
    for i, (_, score, bomb_available, (x, y)) in enumerate(game_state['others']):
        others[i * 5] = score / 100.0
        others[i * 5 + 1] = 1 if bomb_available else 0
        others[i * 5 + 2] = x / COLS
        others[i * 5 + 3] = y / ROWS

    # Concatenate all features into a single array
    state = np.concatenate([np.array([round, step]), field, bomb_map, explosion_map, coin_map, self, others])

    return state
