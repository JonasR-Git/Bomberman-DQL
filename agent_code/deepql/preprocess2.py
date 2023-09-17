import numpy as np

def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    global OPPONENT_MAPPING

    FIELDOFVIEW = 5
    COINCOUNT = 3

    if OPPONENT_MAPPING == {}:
        OPPONENT_MAPPING = {other[0]: i for i, other in enumerate (game_state['others'])}
    


    round = game_state['round']

    step = game_state['step']
    
    field = game_state['field']

    # unpacking bonbs
    bomb_positions = [bomb[0] for bomb in game_state['bombs']]
    bomb_timers = [bomb[1] for bomb in game_state['bombs']]
    bombs_positions = np.array(bomb_positions)
    bomb_timers = np.array(bomb_timers)
    
    coins = game_state['coins']
    
    explosions = game_state['explosion_map']

    
    self_position = np.array(game_state['self'][3]).flatten()
    
    self_bomb = game_state['self'][2]
    
    # unpacking opponents
    opponents = np.full((12), -1)
    for opponent in game_state['others']:
        tmp = [opponent[1], int(opponent[2])]
        tmp.extend(opponent[3])
        opponents[OPPONENT_MAPPING[opponent[0]]:OPPONENT_MAPPING[opponent[0]]+4] = tmp

    # calculate COINCOUNT nearest coins
    coins = np.array(coins)
    if len(coins) > 0:
        distance = coins - self_position
        coins = coins[np.argsort(np.linalg.norm(distance, axis=1))[:min(COINCOUNT, len(coins))]]
        # fill coins with -1 if there are less than COINCOUNT coins
        if len(coins) < COINCOUNT:
            coins = np.concatenate((coins.flatten(), np.full(((COINCOUNT-len(coins)) * 2), -1)))
        coins = coins.flatten()
    else:
        coins = np.full(shape=(COINCOUNT*2), fill_value=-1)

    # Padding explostion_map by two fields in each direction
    explosions = np.pad(explosions, FIELDOFVIEW//2, mode='constant', constant_values=-1)
    # Cut out field of view
    explosions = explosions[self_position[0]:self_position[0]+FIELDOFVIEW, self_position[1]:self_position[1]+FIELDOFVIEW]

    # Padding field by two fields in each direction
    field = np.pad(field, FIELDOFVIEW//2, mode='constant', constant_values=-1)
    # Cut out field of view
    field = field[self_position[0]:self_position[0]+FIELDOFVIEW, self_position[1]:self_position[1]+FIELDOFVIEW]

    # sort bombs by distance
    if len(bombs_positions) > 0:
        distance = bombs_positions - self_position
        bombs_positions = bombs_positions[np.argsort(np.linalg.norm(distance, axis=1))]
        bomb_timers = bomb_timers[np.argsort(np.linalg.norm(distance, axis=1))]
        # fill bombs_positions with -1 if there are less than 4 bombs_positions
        if len(bombs_positions) < 4:
            bombs_positions = np.concatenate((bombs_positions.flatten(), np.full(((4-len(bombs_positions)) * 2), -1)))
            bomb_timers = np.concatenate((bomb_timers.flatten(), np.full((4-len(bomb_timers)), -1)))
        bombs_positions = bombs_positions.flatten()
        bomb_timers = bomb_timers.flatten()
    else:
        bombs_positions = np.full(shape=(8), fill_value=-1)
        bomb_timers = np.full(shape=(4), fill_value=-1)

    # create single vector feature array
    int_features = np.array([round, step, self_bomb])
    field = field.flatten()
    explosions = explosions.flatten()
    vector_features = np.concatenate((self_position, field, bombs_positions, bomb_timers, coins, explosions, opponents))

    features = np.concatenate((int_features, vector_features)).flatten()
    # print("All Shapes")
    # print(f"Shape of int_features: {int_features.shape}")
    # print(f"Shape of vector_features: {vector_features.shape}")
    # print("Shape of all Vector Features")
    # print(f"Shape of self_position: {self_position.shape}")
    # print(f"Shape of field: {field.shape}")
    # print(f"Shape of bombs_positions: {bombs_positions.shape}")
    # print(f"Shape of bomb_timers: {bomb_timers.shape}")
    # print(f"Shape of coins: {coins.shape}")
    # print(f"Shape of explosions: {explosions.shape}")
    # print(f"Shape of opponents: {opponents.shape}")
    # print(f"Shape of features: {features.shape}")
    print(f"Shape of features: {features.shape}")
    return features