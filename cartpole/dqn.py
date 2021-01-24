

from collections import deque
import pandas as pd
import numpy as np 
import json
import random
import time
import gym 


from IPython.display import clear_output

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model



class brain:
    def __init__(self, env, version, REPLAYMEMORY_SIZE, BATCHSIZE, WARMUPSTEPS, TRAINSTEPS, GAMMA, RENDER = False, epsilon_decay=10000):
        self.name = env.env.unwrapped.spec.id
        self.version = version
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.batchSize = BATCHSIZE
        self.warmupSteps = WARMUPSTEPS
        self.trainSteps = TRAINSTEPS
        self.render = RENDER
        self.globalSteps = 0
        self.main_network = None
        self.target_network = None
        self.gamma = GAMMA
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.replayMemory = deque(maxlen=REPLAYMEMORY_SIZE)
        self.logger = {
            'loss':[],
            'episodic_rewards':[],
            'q_values':[],
            'epsilon':[]
        }

    def get_action(self, state):
        if self.epsilon >= np.random.random():
            action = np.random.choice([0, self.n_actions-1])
        else:
            q_values = self.main_network.predict(state.reshape(1, 4)) # TODO model.predict
            action = np.argmax(q_values)
        return action

    def set_perception(self, action, state, reward, nextState, terminal):
        self.globalSteps += 1
        reward = reward if not terminal else -reward
        self.replayMemory.append((action, state, reward, nextState, terminal))

        if self.globalSteps >= self.warmupSteps:
            if self.globalSteps%self.trainSteps == 0:
                self.train()
            self.epsilon = max(self.epsilon-(1/self.epsilon_decay), self.epsilon_min)
            self.logger['epsilon'].append(self.epsilon)
    
    def train(self):
        batch = random.sample(self.replayMemory, self.batchSize)
        if self.render:
            print('start training agent')
        for state, action, reward, nextState, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.amax(self.main_network.predict(np.reshape(nextState, (1, 4)))[0]))
            q_values = self.main_network.predict(np.reshape(state, (1, 4)))
            q_values[0][action] = q_update
            self.logger['q_values'].append(q_values)
            loss = self.main_network.train_on_batch(np.reshape(state, (1, 4)), q_values)
            self.logger['loss'].append(loss)

    def create_model(self):
        states_input = Input(shape=self.input_shape)
        out_1 = Dense(24, activation='relu')(states_input)
        out_1 = Dense(24, activation='relu')(out_1)
        out_1 = Dense(self.n_actions, name='action_output')(out_1)

        model = Model(inputs=[states_input], outputs=[out_1])
        model.compile(
            optimizer = Adam(lr=1e-3, decay=1e-3),
            loss = "mse")
        return model

    def save_experiment(self):
        # prepare file
        self.logger['name'] = '-'.join([self.name,'experiment',self.version,str(pd.datetime.now().date())])
        logfile = self.logger
        logfile['loss'] = np.array(self.logger['loss']).tolist()
        logfile['q_values'] = [q_value.tolist() for q_value in self.logger['q_values']]
        
        # save main network
        self.main_network.save('data/networks/{}.h5'.format(self.logger['name']))
        # save log data
        with open('data/{}.json'.format(logfile['name']), 'w')as f:
            json.dump(logfile, f)

    def load_experiment(self, date=str(pd.datetime.now().date())):
        self.logger['name'] = '-'.join([self.name, 'experiment', self.version, date])
        
        with open('data/{}.json'.format(self.logger['name'])) as f:
            logger = json.load(f)

        logger['loss'] = [np.float32(loss) for loss in logger['loss']]
        logger['q_values'] = [np.array(q_value) for q_value in logger['q_values']]
        self.logger = logger

        # load network
        self.main_network = load_model('data/networks/{}.h5'.format(self.logger['name']))
