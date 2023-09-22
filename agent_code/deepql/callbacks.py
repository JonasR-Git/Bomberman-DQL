from .agent import DQNAgent
from .preprocess import preprocess
import os
import numpy as np
from collections import deque
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

dqn_agent = False
train = True   
replay_counter = 0
current_round = 0
bomb_history = deque([], 5)
coordinate_history = deque([], 20)
ignore_others_timer = 0

def setup(self):
    global dqn_agent
    if dqn_agent == False:
        print("create Agent")
        self.dqn_agent = DQNAgent(196, len(ACTIONS), 1)
        saved_model_path = os.path.join('network_parameters', f'save_after_{11100}_iterations')
        saved_model_path_index = os.path.join('network_parameters', f'save_after_{11100}_iterations.index')
        if os.path.exists(saved_model_path_index):
            print("Model initialised with saved weights")
            self.dqn_agent.load(saved_model_path)
        else:
            print("Saved model weights not found. Proceeding with uninitialized model.")
        dqn_agent = True

def act(self, game_state: dict) -> str:
    global train
    if train:
        return get_random_action(self, game_state)
    state = preprocess(game_state)
    action = self.dqn_agent.act(state)
    print(action)
    return action

def get_random_action(self, game_state):
    global replay_counter
    if np.random.rand() <= self.dqn_agent.get_epsilon():
        if (replay_counter < 30000):
            if (np.random.rand() < 0.9):
                action = rule_based_act(self, game_state)
                print("Random Rule-Based:", action)
                return action
        if (np.random.rand() < 0.4):
            action = rule_based_act(self, game_state)
            print("Random Rule-Based:", action)
            return action
        action = np.random.choice(get_valid_actions(self, game_state))
        print("Random:", action)
        return action
    state = preprocess(game_state)
    return self.dqn_agent.act(state)


def look_for_targets(free_space, start, targets):
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
        # Determine the first step towards the best found target tile
        current = best
        while True:
            if parent_dict[current] == start: return current
            current = parent_dict[current]

def reset_self():
    global bomb_history, coordinate_history, ignore_others_timer
    bomb_history = deque([], 5)
    coordinate_history = deque([], 20)
    ignore_others_timer = 0


def rule_based_act(self, game_state):
    global current_round, coordinate_history, bomb_history, ignore_others_timer
    """
    Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """
    if game_state["round"] != current_round:
        reset_self()
        current_round = game_state["round"]
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if coordinate_history.count((x, y)) > 2:
        ignore_others_timer = 5
    else:
        ignore_others_timer -= 1
    coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in bomb_history: valid_actions.append('BOMB')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                bomb_history.append((x, y))
            return a
        if valid_actions:
            return valid_actions[0]
        return 'WAIT'
    
def get_valid_actions(self, game_state):
    x, y = game_state['self'][3]
    field = game_state['field']
    ROWS, COLS = field.shape
        
    valid_actions = []

    bomb_positions = [bomb[0] for bomb in game_state['bombs']]

    # Check move UP
    if y > 0 and field[y - 1][x] == 0 and (x, y-1) not in bomb_positions:
        valid_actions.append('UP')
    # Check move DOWN
    if y < ROWS - 1 and field[y + 1][x] == 0 and (x, y+1) not in bomb_positions:
        valid_actions.append('DOWN')
    # Check move LEFT
    if x > 0 and field[y][x - 1] == 0 and (x-1, y) not in bomb_positions:
        valid_actions.append('LEFT')
    # Check move RIGHT
    if x < COLS - 1 and field[y][x + 1] == 0 and (x+1, y) not in bomb_positions:
        valid_actions.append('RIGHT')

    # If only one direction is available, restrict actions to that direction or BOMB
    if len(valid_actions) == 1:
        if game_state['self'][2] and (x, y) not in bomb_positions:  # If bomb is available and not at player's position
            valid_actions.append('BOMB')
        return valid_actions  # Return here, do not append 'WAIT'

    # Check for active bombs nearby if more than one direction is possible or no direction is possible
    if game_state['self'][2] and (x, y) not in bomb_positions:
        valid_actions.append('BOMB')

    # Always can wait if more than one direction is possible or no direction is possible
    valid_actions.append('WAIT')

    return valid_actions