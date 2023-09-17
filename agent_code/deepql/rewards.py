import events as e

GAME_REWARDS = {
        e.COIN_COLLECTED: 40,
        e.KILLED_OPPONENT: 200,
        e.MOVED_RIGHT: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: -15,
        e.INVALID_ACTION: -60,
        e.BOMB_DROPPED: -1,
        e.COIN_FOUND: 8,      
        e.KILLED_SELF: -10,
        e.SURVIVED_ROUND: 40,
        e.GOT_KILLED: -300,
    }

def reward_from_events(self, events) -> int:

    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: List of events from events.py that occurred in the game.

    :return: Cumulative reward based on the events.
    '''

    reward_sum = sum(GAME_REWARDS.get(event, 0) for event in events)
    if e.GOT_KILLED not in events: 
        reward_sum += 0.05
    return reward_sum