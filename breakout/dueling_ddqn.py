

from collections import deque
import pandas as pd
import numpy as np 
from datetime import datetime
import json
import random
import time
import gym 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, add
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import load_model
from tensorflow import reduce_mean

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
        #self.logger['epsilon'].append(self.epsilon)
        
        if self.globalSteps >= self.warmupSteps:
            if self.globalSteps%self.trainSteps == 0:
                self.train()
            if self.globalSteps%self.target_network_update_rate == 0:
                self.update_target_network()
            if self.globalSteps%self.checkpoint_rate == 0:
                self.save_experiment()
            
    def train(self):
        minibatch = random.sample(self.replayMemory, self.batchSize)
        
        states, actions, rewards, nextStates, terminal = [], [], [], [], []
        for batch in minibatch:
            states.append(batch[0])
            actions.append(batch[1])
            rewards.append(batch[2])
            nextStates.append(batch[3])
            terminal.append(batch[4])

        q_updates = self.target_network.predict(np.array(nextStates))
        q_updates = [rewards[i] + self.gamma * np.amax(q_updates[i]) if not terminal[i] else rewards[i] for i in range(len(minibatch))]

        q_values = self.main_network.predict(np.array(states))

        for i in range(len(minibatch)):
            q_values[i][actions[i]] = q_updates[i]

        history = self.main_network.fit(np.array(states), np.array(q_values), batch_size=self.batchSize)
        
        self.logger['loss'].append(history.history['loss'][0])
        self.logger['q_values'].append(np.mean(q_updates))

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def create_model(self):
        states_input = Input(shape=self.input_shape)
        main = Dense(128, activation='relu', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(states_input)
        main = Dense(256, activation='relu', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(main)
        
        action = Dense(512, activation='relu', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(main)
        action = Dense(self.n_actions, name='action_output', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(action)
        
        value = Dense(512, activation='relu', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(main)
        value = Dense(1, name='value_output', kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.05))(value)
        
        out = add([value, action-reduce_mean(action, axis=1, keepdims=True)])
        
        model = Model(inputs=[states_input], outputs=[out])
        model.compile(
            optimizer = Adam(lr=1e-5, ),
            loss = 'huber_loss')
        
        return model

    def save_experiment(self):
        # prepare file
        self.logger['name'] = '-'.join([self.name,'experiment', self.version, str(datetime.now().date())])
        logfile = self.logger
        logfile['loss'] = np.array(self.logger['loss']).tolist()
        logfile['q_values'] = np.array(self.logger['q_values']).tolist()
        logfile['replayMemorySize'] = len(self.replayMemory)
        logfile['targetNetworkUpdateRate'] = self.target_network_update_rate
        logfile['epsilonDecay'] = self.epsilon_decay
        logfile['globalSteps'] = self.globalSteps

        # save main network
        self.main_network.save('data/networks/{}.h5'.format(self.logger['name']))
        # save log data
        with open('data/{}.json'.format(logfile['name']), 'w')as f:
            json.dump(logfile, f)

    def load_experiment(self, date=str(datetime.now().date()), name=''):
        self.logger['name'] = '-'.join([self.name, 'experiment', self.version, date])
        
        if name == '':
            with open('data/{}.json'.format(self.logger['name'])) as f:
                logger = json.load(f)
        else: 
            with open('data/{}.json'.format(name)) as f:
                    logger = json.load(f)
                 
        logger['loss'] = [np.float32(loss) for loss in logger['loss']]
        logger['q_values'] = [np.array(q_value) for q_value in logger['q_values']]
        self.logger = logger

        # load network
        self.main_network = load_model('data/networks/{}.h5'.format(self.logger['name']))
        self.target_network = load_model('data/networks/{}.h5'.format(self.logger['name']))