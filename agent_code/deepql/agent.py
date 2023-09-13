import numpy as np
import random
from collections import deque
from .preprocess import preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Increase memory size
        self.memory = deque(maxlen=100000)  
        
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
        self.memory.append((preprocessed_state, action_index, reward, preprocessed_next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * 
                        np.amax(self.model.predict(next_state.reshape(1, -1))[0]))
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            states.append(state.reshape(1, -1))
            targets_f.append(target_f[0])
        states = np.vstack(states)  # Convert list of numpy arrays to a single numpy array
        self.model.fit(states, np.array(targets_f), epochs=1, verbose=0, callbacks=[LearningRateScheduler(self.lr_schedule)])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def lr_schedule(self, epoch):
        return self.learning_rate * (1. / (1. + self.lr_decay * epoch))

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)