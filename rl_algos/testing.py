import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque
from itertools import count


class Policy(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.hidden = 128
        self.fc1 = nn.Linear(num_features, self.hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(self.hidden, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        action_scores = self.fc3(x)

        return F.softmax(action_scores, dim=1)  # my policy

    def act(self, state):
        pass


pol = Policy()

def get_action(state):
    probs = pol.forward()
    distr = Categorical(probs)
    action = distr.sample()  #samples from multinomial
    logprob = distr.log_prob(action)  # log likelihood evaluated at sample (=action)
    return action.item(), logprob

# PARAMS
gamma = 0.99  # discount factor
seed = 8312
eps = np.finfo(np.float32).eps.item()
env = gym.make('CartPole-v1')
num_features = env.observation_space.shape[0]
policy = Policy(num_features)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
# train loop


for episode in range(1000):
    state = env.reset()
    episode_reward = []
    done = False
    for frame in range(100000): # cart pole has only 500 frames
        action, logprob = get_action(state)
        state, reward, done, _ = env.step(action)


