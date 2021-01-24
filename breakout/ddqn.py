

from collections import deque
import pandas as pd
import numpy as np 
from datetime import datetime
import json
import random
import time
import gym 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model


class brain:
    def __init__(self, env, version, REPLAYMEMORY_SIZE, TARGETUPDATERATE, BATCHSIZE,
                 WARMUPSTEPS, TRAINSTEPS, CHECKPOINT, GAMMA, RENDER = False, epsilon = 1, epsilon_decay=10000, epsilon_min=0.1):
        self.name = env.env.unwrapped.spec.id
        self.version = version
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.batchSize = BATCHSIZE
        self.warmupSteps = WARMUPSTEPS
        self.trainSteps = TRAINSTEPS
        self.target_network_update_rate = TARGETUPDATERATE
        self.checkpoint_rate = CHECKPOINT
        self.render = RENDER
        self.globalSteps = 0
        self.main_network = None
        self.target_network = None
        self.gamma = GAMMA
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replayMemory = deque(maxlen=REPLAYMEMORY_SIZE)
        self.logger = {
            'loss':[],
            'episodic_rewards':[],
            'q_values':[],
            'epsilon':[]
        }

    def get_action(self, state):
        if self.epsilon >= np.random.random():
            action = random.randrange(0, self.n_actions)
        else:
            q_values = self.main_network.predict(state.reshape(1, 128)) # TODO model.predict
            action = np.argmax(q_values)
        return action

    def set_perception(self, action, state, reward, nextState, terminal):
        self.globalSteps += 1
        reward = reward if not terminal else -reward
        self.replayMemory.append((action, state, reward, nextState, terminal))
        
        self.epsilon = max(self.epsilon-(1/self.epsilon_decay), self.epsilon_min)
        self.logger['epsilon'].append(self.epsilon)
        
        if self.globalSteps >= self.warmupSteps:
            if self.globalSteps%self.trainSteps == 0:
                self.train()
            if self.globalSteps%self.target_network_update_rate == 0:
                self.update_target_network()
            if self.globalSteps%self.checkpoint_rate == 0:
                self.save_experiment()
            
    def train(self):
        minibatch = random.sample(self.replayMemory, self.batchSize)
        if self.render:
            print('start training agent')
        states,  target_q_values = [], []
        for state, action, reward, nextState, terminal in minibatch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.gamma * np.amax(self.target_network.predict(np.reshape(nextState, (1, 128)))[0]))
            q_values = self.main_network.predict(np.reshape(state, (1, 128)))
            q_values[0][action] = q_update
            self.logger['q_values'].append(float(q_update))
            states.append(state)
            target_q_values.append(q_values[0])

        history = self.main_network.fit(np.array(states), np.array(target_q_values), batch_size=self.batchSize)
        self.logger['loss'].append(history.history['loss'][0])

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def create_model(self):
        states_input = Input(shape=self.input_shape)
        out_1 = Dense(128, activation='relu')(states_input)
        out_1 = Dense(256, activation='relu')(out_1)
        out_1 = Dense(self.n_actions, name='action_output')(out_1)

        model = Model(inputs=[states_input], outputs=[out_1])
        model.compile(
            optimizer = Adam(lr=1e-3, decay=1e-3),
            loss = "mse")
        return model

    def save_experiment(self):
        # prepare file
        self.logger['name'] = '-'.join([self.name,'experiment', self.version, str(datetime.now().date())])
        logfile = self.logger
        logfile['loss'] = np.array(self.logger['loss']).tolist()
        logfile['q_values'] = self.logger['q_values']
        logfile['replayMemorySize'] = len(self.replayMemory)
        logfile['targetNetworkUpdateRate'] = self.target_network_update_rate
        logfile['epsilonDecay'] = self.epsilon_decay

        # save main network
        self.main_network.save('data/networks/{}.h5'.format(self.logger['name']))
        # save log data
        with open('data/{}.json'.format(logfile['name']), 'w')as f:
            json.dump(logfile, f)

    def load_experiment(self, date=str(datetime.now().date())):
        self.logger['name'] = '-'.join([self.name, 'experiment', self.version, date])
        
        with open('data/{}.json'.format(self.logger['name'])) as f:
            logger = json.load(f)

        logger['loss'] = [np.float32(loss) for loss in logger['loss']]
        logger['q_values'] = [np.array(q_value) for q_value in logger['q_values']]
        self.logger = logger

        # load network
        self.main_network = load_model('data/networks/{}.h5'.format(self.logger['name']))
