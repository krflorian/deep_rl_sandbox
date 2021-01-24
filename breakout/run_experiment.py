
#%%
import dueling_ddqn
import gym 
import time
import random
from collections import deque
from tensorflow.keras.models import load_model
from IPython.display import clear_output

from utils import show_RAM_usage, get_RAM_usage

# setup
# parameters
REPLAYMEMORY_SIZE = 1000000
EPSILON_DECAY = 1000000
epsilon = 0.2
TARGETUPDATERATE = 10000
BATCHSIZE = 32
TRAINSTEPS = 4
WARMUPSTEPS = 10000
CHECKPOINT = 100000
GAMMA = 0.99
RENDER = False

# agent
env = gym.make('Breakout-ram-v0')
version = 'dueling_ddqn_final'

agent = dueling_ddqn.brain(env, version, REPLAYMEMORY_SIZE, TARGETUPDATERATE, BATCHSIZE,
                   WARMUPSTEPS, TRAINSTEPS, CHECKPOINT, GAMMA, RENDER, epsilon, EPSILON_DECAY)
agent.main_network = load_model('data/networks/Breakout-ram-v0-experiment-dueling_ddqn.0.3-2020-05-20.h5')
agent.target_network = load_model('data/networks/Breakout-ram-v0-experiment-dueling_ddqn.0.3-2020-05-20.h5')

#run
EPISODES = 1000000
start = time.time()
for e in range(EPISODES):
    state = env.reset()
    steps = 0
    terminal = False
    episodic_reward = 0
    lives = 5
    # random shoot actions at beginning to start environment
    for i in range(random.choice(range(10))):
        env.step(1)
    
    while not terminal:
        steps += 1
        action = agent.get_action(state)
        
        nextState, reward, terminal, info = env.step(action)
        
        # stop episode if loose life
        if info['ale.lives'] < lives: # break if live lost
            agent.set_perception(state, action, reward, nextState, True)
            lives = info['ale.lives']
        else:
            agent.set_perception(state, action, reward, nextState, terminal)
            
        state = nextState
        episodic_reward += reward
        
        # safety net
        if get_RAM_usage() >= 98:
            agent.save_experiment()
            
        show_RAM_usage()    
        clear_output()
        print('action', action, 'lives ', info['ale.lives'], 'global steps ', agent.globalSteps)
        print('episode {} / {} steps {} reward {} epsilon {}  \r'.format(e, EPISODES,
                                                                         steps, episodic_reward, agent.epsilon))
    # end of episode    
    agent.logger['episodic_rewards'].append(episodic_reward)
print('end of training')
agent.logger['runtime'] = round(time.time()-start)/60
print('runtime {} minutes'.format(round(time.time()-start)/60))
env.close()
agent.save_experiment()


"""
DQN has only previously been applied to single-machine architectures, in practice leading to long training times.
For example, it took 12-14 days on a GPU to train the DQN algorithm on a single Atari game (Mnih et al., 2015)
The original Atari Breakout experiment done by Mnih et. al. ran a total of 100 epochs for training, where each training epoch contains 50000 minibatch updates
"""
