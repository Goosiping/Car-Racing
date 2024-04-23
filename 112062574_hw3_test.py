import gym
import gym_multi_car_racing
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from collections import deque

# Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU()
        )
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha(x) + 1
        beta = self.beta(x) + 1

        return (alpha, beta), v

class Agent:
    def __init__(self):
        # Load model
        self.device = 'cpu'
        self.net = Net().float()
        self.net.load_state_dict(torch.load("112062574_hw3_data"))

        self.frame = 0
        self.frame_skip = 4
        self.last_action = (0, 0, 0)
        # self.stack_frames = [np.zeros((96, 96), dtype=np.float64)] * 4

    def act(self, observation):

        if not hasattr(self, 'stack_frames'):
            rgb = observation[0]
            gray = self._preprocess(rgb)
            self.stack_frames = [gray] * 4

        if self.frame % self.frame_skip == 0:
            
            # Stack frame
            rgb = observation[0]
            gray = self._preprocess(rgb)
            self.stack_frames.pop(0)
            self.stack_frames.append(gray)
            state = np.array(self.stack_frames)

            tensor_state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                (alpha, beta), _ = self.net(tensor_state)
            # alpha, beta = alpha.squeeze().detach().numpy(), beta.squeeze().detach().numpy()
            # action = np.random.beta(alpha, beta)
            action = alpha / (alpha + beta)
            action = action.squeeze().cpu().numpy()
            action = action * np.array([2., 1., 1.]) - np.array([1., 0., 0.])
            self.last_action = action

        else:
            action = self.last_action
        
        action = self._check_action(action)
        return action

    @staticmethod
    def _preprocess(state):

        gray = np.dot(state[..., :], [0.299, 0.587, 0.114])
        gray = gray / 128. - 1.

        return gray
    
    @staticmethod
    def _check_action(action):
        low = [-1.0, 0.0, 0.0]
        high = [1.0, 1.0, 1.0]
        for i in range(len(action)):
            if action[i] < low[i]:
                action[i] = low[i]
                print(action)
            if action[i] > high[i]:
                action[i] = high[i]
                print(action)

        return action

def plot():
    alpha = 0.9
    with open("log.txt") as file:
        lines = file.read().split('\n')

    episode = []
    scores = []
    smooth = []
    for line in lines:
        strs = line.split(' ')
        if len(strs) < 2:
            break
        if len(smooth) == 0:
            smooth.append(float(strs[1]))
        else:
            smooth.append(smooth[-1] * (1 - alpha) + float(strs[1]) * alpha)
        episode.append(int(strs[0]))
        scores.append(float(strs[1]))

    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.plot(episode, smooth)
    plt.show()

if __name__ == '__main__':
    
    a = 2
    if a == 0:
        plot()

    else:
        env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)
        
        agent = Agent()
        records = []

        for i in range(50):

            state = env.reset()
            done = False
            steps = 0
            total_reward = 0
            while not done:
                
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                # env.render()
                total_reward += reward
                steps += 1
            
            print("total_reward: {:.2f}, steps: {:d}".format(total_reward[0], steps))
            records.append(total_reward[0])

        print("average score: {:.2f}".format(np.mean(records)))
        
        


    
