# BRAIN
import numpy as np
import cv2 
import random
import time
from collections import deque

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import RMSprop

from utils import show_image, show_RAM_usage
import psutil
import os


class Brain():
    def __init__(self, env, version, memory_size, warmup_steps, exploration_steps,
                 render=False, PER = False, batch_size=20, replay_periode = 1, update_periode=10000):
        self.version = str(version)
        self.global_steps = 0
        self.warmup_steps = warmup_steps
        self.training_steps = 0
        self.global_episodes = 0
        self.exploration_steps = exploration_steps
        self.global_rewards = []
        self.mean_rewards = []
        #self.global_q_values = []
        #self.mean_q_values = []
        self.current_state = deque(maxlen=4)
        self.memory_size = memory_size
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.actions = env.action_space.n
        self.replayMemory = None
        self.primary_network = None
        self.target_network = None
        self.epsilon = 1    # for e-greedy strategy
        self.epsilon_max = 1.0
        self.epsilon_min = 0.1
        self.gamma = 0.99   # for value function (discount factor)
        self.batch_size = batch_size
        self.render = render
        self.PER = PER
        self.replay_periode = replay_periode
        self.update_periode = update_periode
        self.first_episode = True
        self.error = []

        
    def setup(self, warmstart):
        if self.PER:
            from replay_memory_priority import Memory
        else: 
            from replay_memory import Memory

        # configure replayMemory
        self.replayMemory = Memory(self.memory_size, self.env_name, self.version, self.render)
        self.replayMemory.setup()
                
        if warmstart:
            print('warmstart!')
            #self.warmup_steps = 0
            # configure parameters
            parameters = self.replayMemory.load_parameters(self.version)

            # configure brain parameters
            for key in parameters:
                print('set key', key)
                setattr(self, key, parameters[key])

            # load saved brain
            self.primary_network = load_model('model/'+ self.env.unwrapped.spec.id +'-' + self.version + '.h5')
            self.target_network = load_model('model/'+ self.env.unwrapped.spec.id +'-' + self.version + '.h5')
            
        else:
            # configure networks
            self.primary_network = self.create_network()
            self.target_network = self.create_network(primary = False)

            
    def start_env(self):
        # load first 4 frames into memory
        first_state = self.env.reset()

        if self.first_episode:
            first_state = self.preprocess(first_state)
            self.replayMemory.first_frame = first_state 
            self.first_episode = False
        
        for i in range(4):
            self.current_state.append('first_frame')
        
    def get_action(self, state):
        # exploration: select random action 
        if random.random() <= self.epsilon:
            action = random.randrange(self.actions)
        else:
        # exploitation: select action with highest value
            state = self.replayMemory.get_state(state)
            state = self.stack_states(state)
            q_values = self.primary_network.predict_on_batch(state)
            action = np.argmax(q_values)
        return action

    def preprocess(self, state):
        state = np.dot(state, [0.2126, 0.7152, 0.0722])
        state = cv2.resize(state, dsize = (84, 84)) 
        return state
    
    def stack_states(self, state_queue):
        state_stacked = np.stack(state_queue, axis=2)
        state_stacked = np.reshape(state_stacked, [1, 84, 84, 4]).astype('float64')
        return state_stacked
    
    def set_perception(self, action, reward, nextState, done):
        self.global_steps += 1
        nextState = self.preprocess(nextState)
        current_state = list(self.current_state.copy())
        
        memory = {'state': current_state,
                  'action': int(action),
                  'reward': reward,
                  'nextState': nextState,
                  'done': done}
        
        if self.render:
            print('reward', reward, '\naction', action)
            show_image(nextState)
            time.sleep(0.5)
        
        # add next state
        self.current_state.append(nextState)
        #SAVE memory
        self.replayMemory.save_memory(memory)
        
        # train network
        if self.global_steps >= self.warmup_steps:
            if self.global_steps%self.replay_periode == 0:
                self.train_network()
                # save and update network        
                if self.training_steps%100000 == 0:
                    self.save_network()
                    self.replayMemory.save_parameters(self)
                if self.training_steps%self.update_periode == 0:
                    self.update_target_network()
    
    def get_priority(self, state, action, reward, nextState, done):

        if not done: # q_value + prediction of future value
            q_update = (reward + self.gamma*np.amax(self.target_network.predict_on_batch(nextState)[0]))
        else:
            q_update = -1
        q_values = self.primary_network.predict_on_batch(state)

        TD_error = np.abs(q_values[0][action]-q_update)
        priority = (TD_error+0.01) **0.06
        return priority
    
    
    def create_network(self, primary = True):
        model = Sequential()
        model.add(Conv2D(input_shape=(84, 84, 4),
                         filters=32, kernel_size=(8, 8),
                         strides=(4, 4), activation = 'relu',
                         name='conv2d_1'))
        model.add(Conv2D(filters=64, kernel_size=(4, 4),
                         strides=(2, 2), activation = 'relu', 
                         name='conv2d_2'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                        strides=(1, 1), activation = 'relu',
                         name='conv2d_3'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu',
                        name='dense_1'))
        model.add(Dense(self.actions,
                        name='out'))
        if primary:
            model.compile(
                optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                loss='mse',
                metrics=['mean_squared_error'])
        return model
    
    
    def train_network(self):
        print('start training brain...')
        minibatch = self.replayMemory.load_minibatch(self.batch_size)
        
        start = time.time()
        for batch in minibatch:
            # stack states
            nextState = (batch['state'] + [batch['nextState']])[1:]
            nextState = self.stack_states(nextState)
            state = self.stack_states(batch['state'])
            
            action = batch['action']
            reward = batch['reward']
            done = batch['done']
            
            if self.PER:
                idx = batch['idx']
                weight = batch['weigth']

            # Train 
            if not done: # q_value + prediction of future value
                q_update = (reward + self.gamma*np.amax(self.target_network.predict_on_batch(nextState)[0]))
            else:
                q_update = -1
            q_values = self.primary_network.predict_on_batch(state)
            q_values = q_values.numpy()
            q_values[0][action] = q_update
            
            self.training_steps += 1

            # UPDATE PER
            if self.PER:
                self.primary_network.train_on_batch(state, q_values, sample_weight=np.array(weight))
                priority = self.get_priority(state, action, reward, nextState, done)
                self.replayMemory.memory.update(idx, priority)
            else: 
                self.primary_network.train_on_batch(state, q_values)
            
        # Update Epsilon
        self.epsilon = max(self.epsilon-1/self.exploration_steps, self.epsilon_min) # epsilon decay
        
            
    def update_target_network(self):
        print('update_target_mdoel')
        self.target_network.set_weights(self.primary_network.get_weights())

    def save_network(self):
        print('saving network...')
        self.primary_network.save('model/'+ self.env.unwrapped.spec.id +'-' + str(self.version) +'.h5')

