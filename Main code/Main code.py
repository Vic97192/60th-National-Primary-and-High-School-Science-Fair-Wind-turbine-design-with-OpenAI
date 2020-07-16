import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from xfoil.model import Airfoil
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#env = gym.make('MountainCarContinuous-v0')
env = gym.make('XFoil-v0')
seconds = time.time()
env.seed(int((seconds - int(seconds))*10000))
seconds = time.time()
np.random.seed(int((seconds - int(seconds))*10000))

print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)

class Agent(nn.Module):
    def __init__(self, env, h_size=32):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = env.action_space.shape[0]
        # define layers
        '''只有兩層之類神經網路'''
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    '''將權重填入類神經網路模型中'''
    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        # set the weights for each layerobservation_space
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size

#        (.01, .4), (.05, .4), (1, 3), (0.05, 3), (0.4, 8), (1, 10)
#    [0.01, 0.1, 0.5, 0.5, 0.1, 1.0, 10] + [0.08813299, 0.28250898, 2.50168427, 2.56, 1.487, 8.54, 0]
    def forward(self, x):   # x <- state
        x = F.relu(self.fc1(x))
#        x = F.tanh(self.fc2(x))
        x = torch.tanh(self.fc2(x))
#        x1 = x.cpu().data * torch.tensor([0.01, 0.1, 0.5, 0.5, 0.1, 1.0, 5])
#        tmp = tmp * [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 5] + [0.1584, 0.1565, 2.1241, 1.8255, 3.827, 11.6983, 4]

        ''' 類神經網路輸出需乘上權重, 並加到NACA5410標準值之上 '''
        x1 = x.cpu().data * torch.tensor([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 5])
#        x1 = x1 + torch.tensor([0.08813299, 0.28250898, 2.80168427, 2.56204214, 1.48703742, 8.53824561, 0])
#        x1 = x1 + torch.tensor([0.1584, 0.1565, 2.1241, 1.8255, 11.6983, 3.827, 12])
        x1 = x1 + torch.tensor([0.1584, 0.1565, 2.1241, 1.8255, 11.6983, 3.827, 6])
        return x1
        

    def set_state(self):
        state = self.env.reset()
        return state

    def evaluate(self, weights, gamma=1.0, max_t=5000, state = []):
        self.set_weights(weights)
        episode_return = 0.0
#        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            if np.isnan(reward):
                reward = -10000
            else:
                episode_return += reward
            if done:
                break
        return [reward, action, state]

    def evaluate_act(self, weights, gamma=1.0, max_t=5000, state = [], action = []):
        self.set_weights(weights)
        episode_return = 0.0
#        state = self.env.reset()
#        for t in range(max_t):
        state = torch.from_numpy(state).float().to(device)
        action1 = self.forward(state)
        state, reward, done, x, y = self.env.step1(action)
        state1, reward1, done1, x1, y1 = self.env.step1(action1)
        if reward > reward1:
            return reward, action, x, y, state
        else:
            return reward1, action1, x1, y1, state
#           if np.isnan(reward):
 #               print(reward)
 #           else:
 #               episode_return += reward * math.pow(gamma, t)
 #           if done:
 #               break
        return reward, action, x, y, state

agent = Agent(env).to(device)

def cem(n_iterations=140, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.6, loop_no=10):
    """PyTorch implementation of the cross-entropy method.
        
    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite=int(pop_size*elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma*np.random.randn(agent.get_weights_dim())   # 一組
#    best_weight = 2*np.random.randn(agent.get_weights_dim())
    best_reward = 0
    best_state = agent.set_state()
    g_best_reward = 0

    for i_iteration in range(1, n_iterations+1):
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        sigma = sigma * 0.975
#        sigma = sigma * 0.95
#        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
        rewards = []
        actions = []
        states = []
        bb_reward = 0
        for weights in weights_pop:
            reward, action, state = agent.evaluate(weights, gamma, max_t, best_state)
            if reward > bb_reward:
                bb_reward = reward
            rewards.append(reward)
            actions.append(action)
            states.append(state)
        rewards = np.asarray(rewards, dtype=np.float32)
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        mean_weight = np.array(elite_weights).mean(axis=0)
        best_weight = weights_pop[elite_idxs[n_elite-1]]
        best_action = actions[elite_idxs[n_elite-1]]
        best_state = states[elite_idxs[n_elite-1]]
        best_reward = rewards[elite_idxs[n_elite-1]]

        '''如果將上面 best_state 輸入到最好的類神經網路平均權值, 看看是否有更好的 reward'''
        reward, action, x, y, state = agent.evaluate_act(mean_weight, gamma=1.0, action=best_action, state=best_state)
        print(i_iteration, best_reward, reward)
        best_x = x
        best_y = y
        if reward > best_reward:
            best_action = action
            best_reward = reward
            best_x = x
            best_y = y
            best_state = state
            best_weight = np.array(elite_weights).mean(axis=0)

        if best_reward > g_best_reward:
            g_best_action = action
            g_best_reward = best_reward
            g_best_x = best_x
            g_best_y = best_y
            g_best_state = best_state
            g_best_weight = best_weight
#        scores_deque.append(reward)
#        scores.append(reward)
        scores_deque.append(best_reward)
        scores.append(best_reward)

        torch.save(agent.state_dict(), 'checkpoint.pth')
        
        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque)>=190.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
            break
    print (g_best_action)
    print (g_best_x)
    print (g_best_y)
#    np.savetxt(file_name, best_reward)
#    np.savetxt(file_name, best_reward_list)
#    np.savetxt(file_name, best_action)
    env.env.xf.airfoil = Airfoil(x=g_best_x, y=g_best_y)
    #        test_airfoil = NACA4(2, 3, 15)
    #        a = test_airfoil.max_thickness()
    #        self.xf.airfoil = test_airfoil.get_coords()
    cl, cd, cm, cp = env.env.xf.a(g_best_action[6])

    file_name = "reward"+str(loop_no)+".txt"
    f = open(file_name, 'w')
    f.write("%5.5f\n\n" % g_best_reward)
    f.write("%5.5f\n" % cl)
    f.write("%5.5f\n" % cd)
    f.write("%5.5f\n" % cm)
    f.write("%5.5f\n\n" % cp)
    for i in range(len(g_best_action)):
        f.write("%5.5f\n" % g_best_action[i])
    f.write("\n")
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            f.write("0\n")
        else:
            f.write("%5.5f\n" % scores[i])

    f.close()

    #        state = self.state
    return scores

'''做二十次'''
for ii in range(0, 20):
    scores = cem(loop_no=ii)

# plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
#    plt.show()
    file_name = "reward"+str(ii)+".png"
    fig.savefig(file_name)

# load the weights from file
agent.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
while True:
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = agent(state)
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

env.close()