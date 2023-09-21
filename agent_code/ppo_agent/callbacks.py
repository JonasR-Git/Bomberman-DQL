import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from agent_code.ppo_agent.helper import sample_action


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

OPPONENT_MAPPING = {}
FEATURE_LENGTH = 85

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)
    print(f"train: {self.train}")
    if not self.train:
        # colorize print statements

        print(f"{bcolors.OKGREEN}Loading best model from saved state! {bcolors.ENDC}")
        self.actor = keras.models.load_model("experiments/exp0/round4/checkpoints/latest_actor.keras")
    pass


def inference_sample_action(self, observation):
    logits = self.actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        gamestate = tf.reshape(state_to_features(game_state), shape=(1, FEATURE_LENGTH))
        self.logits, action = sample_action(gamestate)
        action = int(action.numpy()[0])
        # print(f"Action Code: {action} - Action: {ACTIONS[action]}")
        # print(f"Action Code: {action}")
        # return ACTIONS[action]
        return ACTIONS[action]
    else:
        gamestate = tf.reshape(state_to_features(game_state), shape=(1, FEATURE_LENGTH))
        _, action = inference_sample_action(self, gamestate)
        action = int(action.numpy()[0])
        print(f"Action Code: {action} - Action: {ACTIONS[action]}")
        return ACTIONS[action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array of length 198 (depending on your representation) 
    """
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
    
    return np.asarray(features).astype('float32')
