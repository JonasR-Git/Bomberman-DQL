import events as e

GAME_REWARDS = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 400,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -20,
        e.INVALID_ACTION: -100,
        e.BOMB_DROPPED: -2,
        e.COIN_FOUND: 15,      
        e.KILLED_SELF: -20,
        e.GOT_KILLED: -400,
    }

def reward_from_events(self, events) -> int:

    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: List of events from events.py that occurred in the game.

    :return: Cumulative reward based on the events.
    '''

    reward_sum = sum(GAME_REWARDS.get(event, 0) for event in events)
    if e.GOT_KILLED not in events: 
        reward_sum += 0.1
    return reward_sum