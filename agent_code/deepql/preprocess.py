import numpy as np
from .settings import MAX_STEPS, EXPLOSION_TIMER, ROWS, COLS, BOMB_TIMER, BOMB_POWER, MAX_AGENTS

def preprocess(game_state):
    # Normalize 'round' and 'step'
    round = game_state['round']
    step = (game_state['step'] / MAX_STEPS) - 0.5

    # Flatten 'field' and 'explosion_map'
    field = game_state['field'].flatten().astype(np.float32) / 1.0
    explosion_map = (game_state['explosion_map'].flatten().astype(np.float32) / EXPLOSION_TIMER) - 0.5

    # Safety map
    safety_map = -np.ones((ROWS, COLS), dtype=np.float32)
    for (x, y), t in game_state['bombs']:
        for dx in range(-BOMB_POWER, BOMB_POWER + 1):
            if 0 <= x + dx < COLS:
                safety_map[y, x + dx] = min(safety_map[y, x + dx], t)
        for dy in range(-BOMB_POWER, BOMB_POWER + 1):
            if 0 <= y + dy < ROWS:
                safety_map[y + dy, x] = min(safety_map[y + dy, x], t)
    safety_map = safety_map.flatten()

    # Process 'bombs'
    bomb_map = np.zeros((ROWS, COLS), dtype=np.float32)
    for (x, y), t in game_state['bombs']:
        bomb_map[y, x] = (t / BOMB_TIMER) - 0.5
    bomb_map = bomb_map.flatten()

    # Process 'coins'
    coin_map = np.zeros((ROWS, COLS), dtype=np.float32)
    for (x, y) in game_state['coins']:
        coin_map[y, x] = 1
    coin_map = coin_map.flatten()

    # Process 'self'
    self = np.zeros(5, dtype=np.float32)
    _, score, bomb_available, (x, y) = game_state['self']
    self[0] = (score / 100.0) - 0.5  # Assume max score 100
    self[1] = 1 if bomb_available else 0
    self[2] = (x / COLS) - 0.5
    self[3] = (y / ROWS) - 0.5

    # Process 'others'
    others = np.zeros(5 * MAX_AGENTS, dtype=np.float32)
    for i, (_, score, bomb_available, (x, y)) in enumerate(game_state['others']):
        others[i * 5] = (score / 100.0) - 0.5
        others[i * 5 + 1] = 1 if bomb_available else 0
        others[i * 5 + 2] = (x / COLS) - 0.5
        others[i * 5 + 3] = (y / ROWS) - 0.5

    # Concatenate all features into a single array
    state = np.concatenate([np.array([round, step], dtype=np.float32), field, bomb_map, explosion_map, safety_map, coin_map, self, others])

    return state
