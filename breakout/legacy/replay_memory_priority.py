# REPLAY MEMORY
import lmdb
import json
import numpy as np 
import psutil
import os
import time
from utils import show_RAM_usage, show_image
from collections import deque, OrderedDict
from IPython import display


print('initialized with priority experience replay memory')


class SumTree:
    # little modified from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0

    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, s):
        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

class Memory():
    def __init__(self, maxlen, env_name, version, render=False):
        self.map_size = 104857600
        self.path = 'memory/' + env_name + version
        self.env = None
        self.env_name = env_name
        self.state_store = OrderedDict()
        self.state_store_disc = None
        self.memory_store = None
        self.parameter_store = None
        self.memory = None
        self.render = render
        self.beta=0.4
        self.beta_increment_per_sample=0.001
        self.maxlen=maxlen
        self.first = False
        self.keycount = 0
        self.first_frame = None
        self.memory_counter = 0
        
    def setup(self):
        env = lmdb.open(path = self.path, map_size=self.map_size,  max_dbs=3, writemap=True)
        self.env = env
        self.memory = SumTree(capacity=self.maxlen)
        self.parameter_store = self.env.open_db(b'agent_parameters')
        self.first = True
   
    def generate_key(self):
        self.keycount += 1
        key = ''.join([self.env_name, str(self.keycount).zfill(7)])
        return key
    
    def save_state(self, state):
        key = self.generate_key()
        print('saving key', key)
            
        self.state_store[key] = state
        if len(self.state_store) > self.maxlen+4:
            self.state_store.popitem(False)
        return key
    
    def get_state(self, state_keys):
        states = []
        for key in state_keys:
            if key == 'first_frame':
                states.append(self.first_frame)
            else:
                states.append(self.state_store[key])
        return states
    
    def save_memory(self, memory):
        start = time.time()
        print('saving memory',memory['nextState'])
        
        if self.first:
            priority = 1
            self.first = False
        else:
            priority = np.max(self.memory.tree[-self.memory.capacity:])
            if self.render:
                print('saving memory with priority {}'.format(priority), '\n')
                print(memory)
                #time.sleep(2)
        self.memory.add(priority, memory)
        self.memory_counter += 1

    def load_minibatch(self, batchsize):
        #print('loading minibatch from replayMemory...')
        start = time.time()
        minibatch = []
        
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        segment = self.memory.total_p / batchsize
        min_prob = 0.01     # for later calculate ISweight
        for i in range(batchsize):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, prio, batch = self.memory.get_leaf(s)
            probability = prio / self.memory.total_p
            ISWeight  = np.power(probability/min_prob, -self.beta)
           
            # get idx and weight
            batch_ready = {}
            batch_ready['state'] = self.get_state(batch['state']) 
            batch_ready['nextState']  = self.state_store[batch['nextState']]
            batch_ready['action'] = batch['action']
            batch_ready['reward'] = batch['reward']
            batch_ready['done'] = batch['done']
            batch_ready['idx'] = idx
            batch_ready['weigth'] = np.array([ISWeight])
            minibatch.append(batch_ready.copy())
            
            if self.render:
                print('loading minibatch ', prio, ' weight: ', ISWeight)
                print('currentState')
                print('keys: ', batch['state'], '\nnext', batch['nextState'])
                for frame in batch_ready['state']:
                    show_image(frame)
                print('nextState', 'action: ', batch_ready['action'])
                show_image(batch_ready['nextState'])
                #time.sleep(2)
                display.clear_output(wait=True)
                
        return minibatch    
    

    def save_parameters(self, agent):

        parameters = {'epsilon': agent.epsilon,
                      'global_steps': agent.global_steps,
                      'training_steps': agent.training_steps,
                      'mean_rewards': agent.mean_rewards,
                      'global_episodes': agent.global_episodes}

        if agent.render:
            print('saving agent parameters\n', parameters)
        value_encoded = json.dumps(parameters)
        value_encoded = value_encoded.encode()
        key = agent.version.encode()
        with self.env.begin(db=self.parameter_store, write=True) as txn:
            txn.put(key=key, value=value_encoded, dupdata=False)

            
    def load_parameters(self, version):
        key = version.encode()
        with self.env.begin(db=self.parameter_store) as txn:
            cursor = txn.cursor()
            value = cursor.get(key)
        parameters = json.loads(value.decode())
        return parameters

    
    """
    def save_experience(self):
        with self.env.begin(write=True) as txn:
            txn.drop(db=self.state_store_disc)
        with self.env.begin(db=self.state_store_disc, write=True) as txn:
            for key, state in self.state_store.items():
                value_encoded = json.dumps(state.tolist()).encode()
                txn.put(key=key.encode(), value=value_encoded, dupdata=False)
    
    def load_experience(self):
        print('loading experience from state store on disc... ')
        counter = 0
        with self.env.begin(db=self.state_store_disc) as txn:
            cursor = txn.cursor()
            while cursor.next():
                counter += 1
                if counter % 1000 == 0:
                    print('loaded {} states from disc'.format(counter))
                self.state_store[cursor.key().decode()] = np.array(json.loads(cursor.value().decode()))
    """
                
                