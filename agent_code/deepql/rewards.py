import events as e

GAME_REWARDS = {
        e.COIN_COLLECTED: 150,
        e.KILLED_OPPONENT: 300,
        e.MOVED_RIGHT: -2,
        e.MOVED_LEFT: -2,
        e.MOVED_UP: -2,
        e.MOVED_DOWN: -2,
        e.WAITED: -5,
        e.INVALID_ACTION: -50,
        e.BOMB_DROPPED: 0,
        e.CRATE_DESTROYED: 5,   
        e.COIN_FOUND: 20,      
        e.KILLED_SELF: -300,
        e.SURVIVED_ROUND: 10,
        e.GOT_KILLED: -400,
    }

def reward_from_events(self, events) -> int:
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: List of events from events.py that occurred in the game.

    :return: Cumulative reward based on the events.
    '''

    reward_sum = sum(GAME_REWARDS.get(event, 0) for event in events)
    return reward_sum