# REPLAY MEMORY
import lmdb
import json
import numpy as np 
import psutil
import os
import time
from utils import show_RAM_usage, show_image

#%%

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
        self.map_size = 104857600000
        self.path = '../memory/' + env_name + version
        self.env = None
        self.env_name = env_name
        self.state_store = None
        self.memory_store = None
        self.parameter_store = None
        self.memory = None
        self.render = render
        self.beta=0.4
        self.beta_increment_per_sample=0.001
        self.maxlen=maxlen
        self.first = False

    def setup(self):
        env = lmdb.open(path = self.path, map_size=self.map_size,  max_dbs=3, writemap=True)
        self.env = env
        self.memory = SumTree(capacity=self.maxlen)
        self.state_store = self.env.open_db(b'state_store')
        self.memory_store = self.env.open_db(b'memory_store')
        self.parameter_store = self.env.open_db(b'agent_parameters')
        self.first = True
        #print(self.env.stat())
    
    def save_state(self, key, state): # 'epsilon', 'last entry'
        if self.render:
            print('saving ', key)
            show_image(state)
        key = key.encode()
        value_encoded = json.dumps(state.tolist()).encode()
        
        with self.env.begin(db=self.state_store, write=True) as txn:
            txn.put(key=key, value=value_encoded, dupdata=False)
        self.env.sync()
        
        return key.decode()
    
    def save_memory(self, memory):
        if self.first:
            priority = 1
            self.first = False
        else:
            priority = np.max(self.memory.tree[-self.maxlen:])
        self.memory.add(priority, memory)

    def load_minibatch(self, batchsize):
        #print('loading minibatch from replayMemory...')
        minibatch = []
        
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        segment = self.memory.total_p / batchsize

        for i in range(batchsize):
            batch_ready = {}
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, _, batch = self.memory.get_leaf(s)
            # get state images
            with self.env.begin(write=True) as txn:
                cursor_state = txn.cursor(db=self.state_store)
                batch_ready['state'] = []
                for state_key in batch['state']:
                    state = cursor_state.get(state_key.encode())
                    state = np.array(json.loads(state.decode()))
                    batch_ready['state'].append(state)
                nextState = cursor_state.get(batch['nextState'].encode())
                nextState = np.array(json.loads(nextState.decode()))
                batch_ready['nextState'] = nextState
            # get action reward done
            batch_ready['action'] = batch['action']
            batch_ready['reward'] = batch['reward']
            batch_ready['done'] = batch['done']
            batch_ready['idx'] = idx
            minibatch.append(batch_ready)    
        return minibatch

    def save_parameters(self, agent):
        
        memory_out = {
            'capacity': self.memory.capacity,   
            'tree': list(self.memory.tree),  
            'transitions': list(self.memory.transitions),
            'next_idx': self.memory.next_idx
        }

        parameters = {'epsilon': agent.epsilon,
                      'global_steps': agent.global_steps,
                      'training_steps': agent.training_steps,
                      'global_rewards': agent.global_rewards,
                      'global_episodes': agent.global_episodes,
                      'memory': memory_out}

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

