import events as e

GAME_REWARDS = {
        e.COIN_COLLECTED: 40,
        e.KILLED_OPPONENT: 200,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -10,
        e.CRATE_DESTROYED: 8,   
        e.COIN_FOUND: 12,      
        e.KILLED_SELF: -300,
        e.SURVIVED_ROUND: 4,
        e.GOT_KILLED: -300,
    }

def reward_from_events(self, events) -> int:

    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: List of events from events.py that occurred in the game.

    :return: Cumulative reward based on the events.
    '''

    print(events)
    reward_sum = sum(GAME_REWARDS.get(event, 0) for event in events)
    if e.BOMB_EXPLODED in events and e.GOT_KILLED not in events:
        reward_sum += 30
    return reward_sum