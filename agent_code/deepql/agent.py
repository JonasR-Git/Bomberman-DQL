import numpy as np
import random
import operator
import bisect
from collections import deque
from .preprocess import preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

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
        self.sum_tree.append(0.0)
        self.min_tree.append(float('inf'))
        self.tree_ptr = (self.tree_ptr + 1) % len(self.sum_tree)
        return idx

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self.data)
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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_counter = 0        
        # Increase memory size
        self.memory = PrioritizedReplayBuffer(size=2000, alpha=0.6)
        
        self.gamma = 0.95  # discount rate
        
        # Initialize with a higher exploration rate
        self.epsilon = 1.0  
        # Slow down epsilon decay
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.learning_rate = 0.001
        # Adjust learning rate decay
        self.lr_decay = 0.00005

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        
        # Input Layer
        model.add(Dense(512, input_dim=self.state_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        # Hidden Layer 1
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden Layer 2
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden Layer 3
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Output Layer
        model.add(Dense(self.action_size, activation='linear'))
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        preprocessed_state = preprocess(state)
        preprocessed_next_state = preprocess(next_state)
        action_index = ACTIONS.index(action)
        self.memory.add((preprocessed_state, action_index, reward, preprocessed_next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTIONS)
        act_values = self.model.predict(state)
        return ACTIONS[np.argmax(act_values[0])]

        
    def replay(self, batch_size, beta=0.4):
        if len(self.memory) < batch_size:
            return
        minibatch, indices, weights = self.memory.sample(batch_size, beta)
        states, targets_f = [], []
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
            target_f = self.model.predict(state.reshape(1, -1))
            
            # Get the TD error and update priority
            old_val = target_f[0][action]
            td_error = abs(old_val - target)
            self.memory.update_priorities([indices[idx]], [td_error])
    
            target_f[0][action] = target
            states.append(state.reshape(1, -1))
            targets_f.append(target_f[0])

        self.replay_counter += 1
        states = np.vstack(states)
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        return self.model.fit(states, np.array(targets_f), epochs=1, verbose=0, callbacks=[lr_scheduler])

    def lr_schedule(self, epoch):
        # You can adjust the schedule as you see fit.
        if self.replay_counter > 1000:
            return self.learning_rate * 0.1
        elif self.replay_counter > 500:
            return self.learning_rate * 0.5
        return self.learning_rate

    
    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)
