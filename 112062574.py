import gym
import gym_multi_car_racing
import gym
import torch
import torch.nn as nn
import numpy as np
import cv2

# Parameters

'''
===================================================================================================
Diagonal Gaussian Distribution Module
===================================================================================================
'''

#AddBias module
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

#Gaussian distribution with given mean & std.
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)
    
    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

#Diagonal Gaussian module
class DiagGaussian(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        return FixedNormal(mean, logstd.exp())
    
'''
===================================================================================================
Policy Network & Value Network Module
===================================================================================================
'''
#Policy Network
class PolicyNet(nn.Module):
    #Constructor
    # TODO: Modify convolutional network
    def __init__(self, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
           nn.Linear(s_dim, 128),
           nn.ReLU(),
           nn.Linear(128, 128),
           nn.ReLU()
        )
        self.dist = DiagGaussian(128, a_dim)
    #Forward pass
    def forward(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        return action, dist.log_probs(action)
    
    #Choose an action (stochastically or deterministically)
    def choose_action(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            return dist.mode()

        return dist.sample()
    
    #Evaluate a state-action pair (output log-prob. & entropy)
    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        return dist.log_probs(action), dist.entropy()

#Value Network
class ValueNet(nn.Module):
    #Constructor
    # TODO: Modify convolutional network
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    #Forward pass
    def forward(self, state):
        return self.main(state)[:, 0]
    
class Buffer:
    def __init__(self, s_dim, a_dim, max_len):
        self.max_len = max_len
        self.states = np.zeros((max_len, s_dim), dtype=np.float32)
        self.actions = np.zeros((max_len, a_dim), dtype=np.float32)
        self.values = np.zeros((max_len,), dtype=np.float32)
        self.rewards = np.zeros((max_len,), dtype=np.float32)
        self.a_logps = np.zeros((max_len,), dtype=np.float32)
    
    

if __name__ == "__main__":

    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)
    
    states = env.reset()

    for episode in range(50):

        states, rewards, dones, infos = env.step((0, 0, 0))

    print(states.shape)
    gray = cv2.cvtColor(states[0], cv2.COLOR_RGB2GRAY)
    cv2.imshow("state", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()