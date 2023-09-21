from collections import namedtuple, deque

import pickle
from typing import List

import agent_code.ppo_agent.helper as h
import agent_code.ppo_agent.hyperparameters as hyp
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

import events as e

import os
import sys


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

ACTION_MAPPING = {action:i for i, action in enumerate(ACTIONS)}

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
OPPONENT_MAPPING = {}

ExperimentID = 0
TROLL_THE_AGENT = True


EPOCHS = hyp.epochs
STEPS_PER_EPOCH = hyp.steps_per_epoch
num_actions = 0
obs_dims = 85
buffer = None

epoch_counter = 0

sum_return = 0
sum_length = 0
num_episodes = 0

mean_returns = []
mean_lengths = []
RoundID = 1

ExperimentPath = f"experiments/exp{ExperimentID}"


def reset_after_epoch():
    global epoch_counter, sum_return, sum_length, num_episodes, episode_length, episode_return, already_finished
    epoch_counter += 1
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    episode_length = 0
    episode_return = 0
    already_finished = False
    return None

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    global num_actions, buffer, actor, critic, value, policy_optimizer, value_optimizer, observation, episode_return, episode_length, already_finished

    num_actions = len(ACTIONS)

    # initialize buffer
    buffer = h.Buffer(obs_dims, STEPS_PER_EPOCH)

    # initialize actor and critic networks
    observation_input = keras.Input(shape=(obs_dims,), name="observation", dtype=tf.float32)
    logits = h.mlp(observation_input, list(h.hidden_sizes) + [num_actions], tf.nn.relu, None)
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze(h.mlp(observation_input, list(h.hidden_sizes) + [1], tf.nn.relu, None), axis=1)
    critic = keras.Model(inputs=observation_input, outputs=value)

    # initialize optimizers
    policy_optimizer = keras.optimizers.Adam(learning_rate=h.policy_learning_rate)
    value_optimizer = keras.optimizers.Adam(learning_rate=h.value_function_learning_rate)

    # initialize observations, episode return and episode length
    observation = None
    episode_return = 0
    episode_length = 0
    already_finished = False
    self.checkpoint = 0


    setup_experiment()
    return
    

def setup_experiment():
    # creating Experiment Directories if not existent
    if not os.path.exists(ExperimentPath):
        os.makedirs(f"{ExperimentPath}")
    else:
        # If experiment already started continue with next round
        # Get the latest round number
        global RoundID
        RoundID = len(os.listdir(ExperimentPath)) 
        if RoundID >= 1:
            # if experiment already started, load the latest model
            global actor, critic
            actor = keras.models.load_model(f"{ExperimentPath}/round{RoundID}/checkpoints/latest_actor.keras")
            critic = keras.models.load_model(f"{ExperimentPath}/round{RoundID}/checkpoints/latest_critic.keras")
            print(f"{h.bcolors.OKGREEN}Loaded latest model from round {RoundID}{h.bcolors.ENDC}\nLoaded from path: {h.bcolors.OKCYAN}{ExperimentPath}/round{RoundID}/checkpoints{h.bcolors.ENDC}")
        RoundID += 1
    global RoundPath
    RoundPath = f"{ExperimentPath}/round{RoundID}"
    # creating Round Directory and subdirs if not existent
    if not os.path.exists(RoundPath):
        os.makedirs(f"{RoundPath}/checkpoints")
        os.makedirs(f"{RoundPath}/analysis")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    global episode_return, episode_length, buffer, sum_return, sum_length, num_episodes, already_finished
    if episode_length < STEPS_PER_EPOCH:

        # get logits and action (before and after)
        observation = tf.reshape(state_to_features(old_game_state), shape=(1, obs_dims))
        if TROLL_THE_AGENT:
            logits, action = h.sample_action(observation)
        else:
            logits = self.logits
            action = ACTION_MAPPING[self_action]
        observation_new = tf.reshape(state_to_features(new_game_state), shape=(1, obs_dims))
        reward = reward_from_events(self, events)
        episode_return += reward
        episode_length += 1

        # get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = h.logprobabilities(logits, action)

        # store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)

        # update the observation
        observation = observation_new

    if (episode_length == STEPS_PER_EPOCH) and not already_finished:
        last_value = critic(observation)
        buffer.finish_trajectory(last_value)
        sum_return += episode_return
        sum_length += episode_length
        num_episodes += 1
        already_finished = True

    return None
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    global sum_return, sum_length, num_episodes, already_finished, buffer

    # finish trajectory if not already finished
    if not already_finished:
        last_value = 0
        buffer.finish_trajectory(last_value)
        sum_return += episode_return
        sum_length += episode_length
        num_episodes += 1
        already_finished = True

    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # update the policy and implement early stopping using KL divergence

    for _ in range(h.train_policy_iterations):
        k1 = h.train_policy(
            observation_buffer,
            action_buffer,
            logprobability_buffer,
            advantage_buffer
        )
        if k1 > 1.5 * h.target_kl:
            # early stopping
            break

    # update the value function
    for _ in range(h.train_value_iterations):
        h.train_value_function(observation_buffer, return_buffer)

    mean_return = sum_return / num_episodes
    mean_length = sum_length / num_episodes

    mean_returns.append(mean_return)
    mean_lengths.append(mean_length)

    print(
        f"Epoch: {epoch_counter}, Mean Return: {mean_return}, Mean Length: {mean_length}"
    )

    # save the model
    try:
        if epoch_counter % 10000 == 0:
            self.checkpoint += 1
            if mean_return == max(mean_returns):
                # saving best model
                print(f"Saving best model of epoch {epoch_counter}")
                actor.save(f"{RoundPath}/checkpoints/best_actor.keras")
                critic.save(f"{RoundPath}/checkpoints/best_critic.keras")
            
            # saving by checkpoint
            actor.save(f"{RoundPath}/checkpoints/actor_cp{self.checkpoint}.keras")
            critic.save(f"{RoundPath}/checkpoints/critic_cp{self.checkpoint}.keras")

            # save as most recent model
            actor.save(f"{RoundPath}/checkpoints/latest_actor.keras")
            critic.save(f"{RoundPath}/checkpoints/latest_critic.keras")
            print(f"Successfully saved model of epoch {epoch_counter}")
    except Exception as e:
        self.logger.info(f"Saving failed at epoch {epoch_counter}")
        print(f"Saving failed at epoch {epoch_counter} - error: {e}")
        pass

    # reset after epoch
    reset_after_epoch()

    

    


consecutive_waited = 0
def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    global consecutive_waited

    GAME_REWARDS = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 0,
        e.MOVED_RIGHT: 0.01,
        e.MOVED_LEFT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        e.WAITED: -0.05,
        e.INVALID_ACTION: -100,
        e.BOMB_DROPPED: -0.1,
        e.CRATE_DESTROYED: 0,
        e.COIN_FOUND: 0,
        e.KILLED_SELF: -1,
        e.SURVIVED_ROUND: 0.1,
        e.GOT_KILLED: 0,
    }

    # punish staying afk
    if e.WAITED in events:
        consecutive_waited += 1
    else: 
        consecutive_waited = 0

    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]

    # if consecutive_waited > 4:
    #     reward_sum += GAME_REWARDS[e.WAITED] * consecutive_waited

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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