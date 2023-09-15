import events as e

GAME_REWARDS = {
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: -10,
        e.INVALID_ACTION: -40,
        e.BOMB_DROPPED: 0,
        e.CRATE_DESTROYED: 20,   
        e.COIN_FOUND: 100,      
        e.KILLED_SELF: -500,
        e.SURVIVED_ROUND: 10,
        e.GOT_KILLED: -800,
    }

def reward_from_events(self, events) -> int:

    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: List of events from events.py that occurred in the game.

    :return: Cumulative reward based on the events.
    '''

    reward_sum = sum(GAME_REWARDS.get(event, 0) for event in events)

    return reward_sum