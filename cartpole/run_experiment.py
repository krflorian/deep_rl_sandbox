

#%%

import ddqn
import gym 
import time
from IPython.display import clear_output

from utils import show_RAM_usage

# setup
# parameters
REPLAYMEMORY_SIZE = 1000000
EPSILON_DECAY = 8000
TARGETUPDATERATE = 800
BATCHSIZE = 32
WARMUPSTEPS = 1000
TRAINSTEPS = 1
GAMMA = 0.99
RENDER = True

# agent
env = gym.make('CartPole-v1')
versions = ['ddqn-0.2.6', 'ddqn-0.2.7', 'ddqn-0.2.8', 'ddqn-0.2.9']

for version in versions:
    agent = ddqn.brain(env, version, REPLAYMEMORY_SIZE, TARGETUPDATERATE, BATCHSIZE, WARMUPSTEPS, TRAINSTEPS, GAMMA, RENDER, EPSILON_DECAY)
    agent.main_network = agent.create_model()
    agent.target_network = agent.create_model()

    #run
    EPISODES = 200
    start = time.time()
    for e in range(EPISODES):
        state = env.reset()
        steps = 0
        terminal = False
        while not terminal:
            steps += 1
            #env.render()
            action = agent.get_action(state)
            nextState, reward, terminal, _ = env.step(action)
            agent.set_perception(state, action, reward, nextState, terminal)
            state = nextState
            show_RAM_usage()
            print('episode {} / {} step {} epsilon {}  \r'.format(e, EPISODES, steps, agent.epsilon))
            if steps >= 200:
                break
        agent.logger['episodic_rewards'].append(steps)
    print('end of training')
    print('runtime {} minutes'.format(round(time.time()-start)/60))
    env.close()
    agent.save_experiment()

