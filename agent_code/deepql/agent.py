import numpy as np
import random
import operator
import bisect
from collections import deque
from .preprocess import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import sys
import os
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """Initialize the Segment Tree with given capacity, operation function and neutral element."""
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        self.tree = [neutral_element for _ in range(2 * capacity)]
        self.data = [None for _ in range(capacity)]
        self.data_pointer = 0
        

    def _operate_helper(self, start, end, node, query_start, query_end):
        if start == query_start and end == query_end:
            return self.tree[node]
        mid = (start + end) // 2
        if query_end <= mid:
            return self._operate_helper(start, mid, 2 * node, query_start, query_end)
        else:
            if mid + 1 <= query_start:
                return self._operate_helper(mid + 1, end, 2 * node + 1, query_start, query_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, query_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, query_end)
                )

    def operate(self, start=0, end=None):
        """Returns result of applying self.operation."""
        if end is None:
            end = self.capacity
        return self._operate_helper(0, self.capacity, 1, start, end)

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        return self.data[idx]

    def append(self, data, val):
        self.data[self.data_pointer] = data
        self[self.data_pointer] = val
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def retrieve(self, s, node=1, start=0, end=None):
        """ 
        Traverse the tree to find where s lies within leaf nodes.
        """
        if end is None:
            end = self.capacity - 1
    
        # If we're at a leaf node
        if start == end:
            return start, self.tree[node]
    
        mid = (start + end) // 2
    
        # If the value at the left child is greater than s, search in the left subtree
        if self.tree[2 * node] > s:
            return self.retrieve(s, 2 * node, start, mid)
        # Otherwise, search in the right subtree with adjusted s
        else:
            return self.retrieve(s - self.tree[2 * node], 2 * node + 1, mid + 1, end)
        
    def __len__(self):
        return self.capacity

            
class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta=0.4, beta_increment=0.001, max_beta=1.0):
        """Initialize Prioritized Replay Buffer."""
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self.alpha = alpha
        self.beta = beta  
        self.beta_increment = beta_increment  
        self.max_beta = max_beta  
        self.buffer_capacity = size        
        self.data = []
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.sum_tree = SegmentTree(it_capacity, operator.add, 0.0)
        self.min_tree = SegmentTree(it_capacity, min, float('inf'))

    def add(self, data):
        idx = self.tree_ptr
        # If the buffer is full, we should also ensure that the tree_ptr points to the oldest data.
        # This way, when we add new data to the SegmentTree, we are overwriting the oldest data.
        if len(self.data) >= self.buffer_capacity:
            self.data.pop(0)
            self.tree_ptr = idx % self.buffer_capacity
        self.data.append(data)
        self.sum_tree.append(data, self.max_priority)  # set to max_priority
        self.min_tree.append(data, self.max_priority)  # optionally, set to max_priority here as well
        self.tree_ptr = (self.tree_ptr + 1) % len(self.sum_tree)
        return idx

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self.data)
            self.max_priority = max(self.max_priority, priority)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

    def sample(self, n, beta):
        assert len(self.data) > 0
        res = []
        idxes = []
        weights = []
        p_min = self.min_tree.operate() / self.sum_tree.operate()
        max_weight = (p_min * len(self.data)) ** (-beta)
        p_total = self.sum_tree.operate(0, len(self.data) - 1)
        segment = p_total / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self._sample_from_segment(s)
            weight = (p * len(self.data)) ** (-beta) / max_weight
            weights.append(weight)
            idxes.append(idx)
            res.append(data)
        return res, idxes, weights
        

    def _sample_from_segment(self, s):
        idx, p = self.sum_tree.retrieve(s)
        data = self.data[idx]
        return idx, p, data
    
    def print_size(self):
        print("Memory data size:", len(self.data))
        print("Sum tree data size:", len(self.sum_tree.data))
        print("Min tree data size:", len(self.min_tree.data))
        print(sys.getsizeof(self.data), sys.getsizeof(self.sum_tree.data), sys.getsizeof(self.min_tree.data))
       
            
    def __len__(self):
        return len(self.data) 

    def __str__(self):
        return "First 10 items in buffer:\n" + "\n".join(str(self.data[i]) for i in range(min(10, len(self.data))))

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_counter = 0
        # Increase memory size
        self.memory = PrioritizedReplayBuffer(size=50000, alpha=0.6)
        self.update_freq = 10
        self.lazy_indices = []
        self.lazy_priorities = []
        self.beta = 0.4  # starting value of beta
        self.beta_increment = 0.0001  # the amount by which beta is incremented at each step
        self.max_beta = 1.0  # maximum value of beta
        
        self.gamma = 0.95  # discount rate
        
        # Initialize with a higher exploration rate
        self.epsilon = 1 
        # Slow down epsilon decay
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.001
        
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        # Output Layer
        model.add(Dense(self.action_size, activation='linear'))
        
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state):
        preprocessed_state = preprocess(state)
        preprocessed_next_state = preprocess(next_state)
        action_index = ACTIONS.index(action)
        self.memory.add((preprocessed_state, action_index, reward, preprocessed_next_state))
        #self.memory.print_size()

    def act(self, state, game_state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.get_valid_actions(game_state))
            print("Random Action:", action)
            return action
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        max_val = np.max(act_values)
        act_values2 = act_values.flatten()
        best_actions = [action for action, q_val in zip(ACTIONS, act_values2) if q_val == max_val]
        print("Calculated Action:", best_actions)
        return random.choice(best_actions)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_q_values(self):
        return self.model.get_weights()  
    
    def print_size(self):
        self.memory.print_size()

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
        
    def replay(self, batch_size):
        print(len(gc.get_objects()))
        print("Train")
        if len(self.memory.data) < batch_size :
            return
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        minibatch, indices, weights = self.memory.sample(batch_size, self.beta)
        states, targets_f = [], []
        for idx, (state, action, reward, next_state) in enumerate(minibatch):
            if next_state is None:
                continue
            target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0]))
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            
            # Get the TD error and update priority
            old_val = target_f[0][action]
            td_error = np.clip(abs(old_val - target), -1, 1) + 1e-5
            self.lazy_indices.append(indices[idx])
            self.lazy_priorities.append(td_error)
            if self.replay_counter % self.update_freq == 0:
                self.memory.update_priorities(self.lazy_indices, self.lazy_priorities)
                self.lazy_indices, self.lazy_priorities = [], []
    
            target_f[0][action] = target
            target_f[0][action] *= weights[idx]  # Weighting the TD error with the importance sampling weight
            states.append(state.reshape(1, -1))
            targets_f.append(target_f[0])

        self.replay_counter += 1
        states = np.vstack(states)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        #objs_to_track = {
        #    "memory_data": self.memory.data,  # Assuming replay_buffer is an instance of PrioritizedReplayBuffer
        #    "sum_tree_data": self.memory.sum_tree.data,
        #    "min_tree_data": self.memory.min_tree.data
        #}
        #log_memory_usage(objs_to_track)
        return self.model.fit(states, np.array(targets_f), epochs=3, verbose=0, callbacks=[lr_scheduler])
    

    def lr_schedule(self, epoch):
        # A dynamic learning rate can be helpful. This is a simple step decay, but more sophisticated methods exist.
        # This decay can be adjusted based on the problem at hand.
        if epoch < 100:
            return self.learning_rate
        elif epoch < 500:
            return self.learning_rate * 0.9
        elif epoch < 1000:
            return self.learning_rate * 0.7
        return self.learning_rate * 0.5

    def clean_up_memory(self):
        gc.collect()
    
    def load(self, name):
        print("Model loaded")
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def log_memory_usage(objs):
    for name, obj in objs.items():
        size_in_bytes = sys.getsizeof(obj)
        size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to megabytes
        print(f"{name}: {size_in_mb:.2f} MB")