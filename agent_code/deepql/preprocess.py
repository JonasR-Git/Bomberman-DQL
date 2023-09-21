import numpy as np
from settings import MAX_STEPS, EXPLOSION_TIMER, ROWS, COLS, BOMB_TIMER, BOMB_POWER, MAX_AGENTS


def preprocess(game_state):
    if game_state is None:
        return None

    _, _, _, (agent_x, agent_y) = game_state['self']

    # The range is now 4 to ensure a 7x7 field around the agent
    RANGE = 3 

    # Create a fixed size 9x9 field initialized with -1
    fixed_size_field = np.full((7, 7), -1)
    clipped_field = np.full((7, 7), -1)
    fixed_size_bomb_map = np.zeros_like(fixed_size_field, dtype=np.float32)
    fixed_size_safety_map = np.full_like(fixed_size_field, -1, dtype=np.float32)
    fixed_size_coin_map = np.zeros_like(fixed_size_field, dtype=np.float32)

    # Calculate bounds for the slice of the actual game map
    x_min = max(agent_x - RANGE, 0)
    x_max = min(agent_x + RANGE + 1, game_state['field'].shape[1])
    y_min = max(agent_y - RANGE, 0)
    y_max = min(agent_y + RANGE + 1, game_state['field'].shape[0])

    # Get the actual game map slice
    clipped_field[RANGE-agent_y+y_min:RANGE-agent_y+y_max, RANGE-agent_x+x_min:RANGE-agent_x+x_max] = game_state['field'][y_min:y_max, x_min:x_max]

    fixed_size_field = clipped_field

    for (x, y), t in game_state['bombs']:
        rel_x = x - agent_x + RANGE
        rel_y = y - agent_y + RANGE
        if 0 <= rel_x < 7 and 0 <= rel_y < 7:  # Ensure the coordinates are within the 7x7 field
            fixed_size_bomb_map[rel_y, rel_x] = t
            for dx in range(-BOMB_POWER, BOMB_POWER + 1):
                if 0 <= rel_x + dx < 7:
                    fixed_size_safety_map[rel_y, rel_x + dx] = min(fixed_size_safety_map[rel_y, rel_x + dx], t)
            for dy in range(-BOMB_POWER, BOMB_POWER + 1):
                if 0 <= rel_y + dy < 7:
                    fixed_size_safety_map[rel_y + dy, rel_x] = min(fixed_size_safety_map[rel_y + dy, rel_x], t)

    coins = game_state['coins']
    coins_distances = [(abs(agent_x - x) + abs(agent_y - y), (x, y)) for (x, y) in coins]
    coins_distances.sort(key=lambda x: x[0])
    for _, (x, y) in coins_distances[:2]:
        rel_x = x - agent_x + RANGE
        rel_y = y - agent_y + RANGE
        if 0 <= rel_x < 7 and 0 <= rel_y < 7:  # Ensure the coordinates are within the 7x7 field
            fixed_size_coin_map[rel_y, rel_x] = 1

    state_representation = np.stack([fixed_size_field, fixed_size_bomb_map, fixed_size_safety_map, fixed_size_coin_map], axis=-1)

    return state_representation