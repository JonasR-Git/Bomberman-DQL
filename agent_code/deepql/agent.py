import numpy as np
import random
import operator
import bisect
from collections import deque
from .preprocess import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
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
    def __init__(self, size, alpha):
        """Initialize Prioritized Replay Buffer."""
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self.alpha = alpha
        self.buffer_capacity = size        
        self.data = []
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.sum_tree = SegmentTree(it_capacity, operator.add, 0.0)
        self.min_tree = SegmentTree(it_capacity, min, float('inf'))

    def add(self, data):
        idx = self.tree_ptr
        if len(self.data) >= self.buffer_capacity:
            self.data.pop(0)
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
        if not isinstance(self.data[0], tuple):
            res = []
            idxes = []
            p_total = self.sum_tree.operate(0, len(self.data) - 1)
            segment = p_total / n
            priorities = []
            for i in range(n):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                idx, p, data = self._sample_from_segment(s)
                priorities.append(p)
                res.append(data)
                idxes.append(idx)
            return res, idxes, priorities
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
        self.memory = PrioritizedReplayBuffer(size=100000, alpha=0.6)
        self.update_freq = 10
        self.lazy_indices = []
        self.lazy_priorities = []
        self.beta = 0.4  # starting value of beta
        self.beta_increment = 0.001  # the amount by which beta is incremented at each step
        self.max_beta = 1.0  # maximum value of beta
        
        self.gamma = 0.95  # discount rate
        
        # Initialize with a higher exploration rate
        self.epsilon = 1.0  
        # Slow down epsilon decay
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        
        # Input Layer
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # Hidden Layer 1
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Hidden Layer 2
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Hidden Layer 3
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Output Layer
        model.add(Dense(self.action_size, activation='linear'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state):
        preprocessed_state = preprocess(state)
        preprocessed_next_state = preprocess(next_state)
        action_index = ACTIONS.index(action)
        self.memory.add((preprocessed_state, action_index, reward, preprocessed_next_state))

    def act(self, state, game_state):
        # Decay epsilon after taking an action
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.get_valid_actions(game_state))
            print(action)
            return action
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        max_val = np.max(act_values)
        act_values2 = act_values.flatten()
        best_actions = [action for action, q_val in zip(ACTIONS, act_values2) if q_val == max_val]
        
        return random.choice(best_actions)

    def get_valid_actions(self, game_state):
        x, y = game_state['self'][3]
        ROWS, COLS = field.shape
        field = game_state['field']
        
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
        
        # Check for active bombs nearby
        if game_state['self'][2] and (x, y) not in bomb_positions:
            valid_actions.append('BOMB')

        # Always can wait
        valid_actions.append('WAIT')

        return valid_actions


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def replay(self, batch_size):
        if len(self.memory.data) < batch_size:
            return
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        minibatch, indices, weights = self.memory.sample(batch_size, self.beta)
        states, targets_f = [], []
        for idx, (state, action, reward, next_state) in enumerate(minibatch):
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
        
        return self.model.fit(states, np.array(targets_f), epochs=5, verbose=1, callbacks=[lr_scheduler])
    

    def lr_schedule(self, epoch):
        # You can adjust the schedule as you see fit.
        if self.replay_counter > 10000:
            return self.learning_rate * 0.1
        elif self.replay_counter > 5000:
            return self.learning_rate * 0.5
        return self.learning_rate

    
    def load(self, name):
            self.model = tf.keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)
