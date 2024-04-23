import gym
import gym_multi_car_racing
import numpy as np
from collections import deque
import torch.nn as nn
import torch
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import time

episodes = 100000
checkpoints = 10
STACK_FRAME = 4
SKIP_FRAME = 4
INPUT_SHAPE = (4, 96, 96)
model_file = "model.pth"
reward_deque_maxlen = 100

learning_rate = 1e-3
gamma = 0.99
max_grad_norm = 0.5
clip_param = 0.1
ppo_epoch = 10
buffer_capacity = 2000
batch_size = 128
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

transition = np.dtype([('s', np.float64, (STACK_FRAME, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('next_s', np.float64, (STACK_FRAME, 96, 96))])

class Env:

    def __init__(self):
        self.env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                    use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                    use_ego_color=False)

    def step(self, action):

        all_reward = 0
        for i in range(SKIP_FRAME):
            rgb, reward, done, info = self.env.step(action)

            rgb = rgb[0]     # rgb: (96, 96, 3)

            reward = reward[0]
            self.reward_deque.append(reward)
            # Trick: green penalty
            if np.mean(rgb[:85, :, 1]) > 205.0:
                reward -= 0.05
            all_reward += reward

            # Trick: if no reward recently, end the episode
            if np.mean(self.reward_deque) <= -0.1:
                done = True

            if done:
                break

        gray = self._preprocess(rgb)
        self.stack.pop(0)
        self.stack.append(gray)

        return np.array(self.stack), all_reward, done, info
    
    @staticmethod
    def _preprocess(state):

        gray = np.dot(state[..., :], [0.299, 0.587, 0.114])
        gray = gray / 128. - 1.

        return gray
    
    def reset(self):
        state = self.env.reset()

        self.reward_deque = deque(maxlen=reward_deque_maxlen)
        gray = self._preprocess(state[0])
        self.stack = [gray] * STACK_FRAME

        # Trick? Skip the first few frames
        for i in range(STACK_FRAME):
            rgb, _, _, _ = self.env.step((0, 0, 0))
            gray = self._preprocess(rgb[0])
            self.stack[i] = gray

        return np.array(self.stack)
    
    def render(self):
        self.env.render()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(STACK_FRAME, 8, kernel_size=4, stride=2),
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
    def __init__(self) -> None:
        self.counter = 0
        self.net = Net().double().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        # Buffer
        self.buffer = np.empty(buffer_capacity, dtype=transition)

    def act(self, state):
        tensor_state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            (alpha, beta), _value = self.net(tensor_state)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp
    
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == buffer_capacity:
            self.counter = 0
            return True
        else :
            return False

    def update(self):
        
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        next_s = torch.tensor(self.buffer['next_s'], dtype=torch.double).to(device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(next_s)[1]
            adv = target_v - self.net(s)[1]

        # loss_record = []
        for epoch in range(ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(buffer_capacity)), batch_size, False):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])
                
                loss1 = ratio * adv[index]
                loss2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]
                action_loss = -torch.min(loss1, loss2).mean()

                value = self.net(s[index])[1]
                value_loss = F.smooth_l1_loss(value, target_v[index])
                loss = action_loss + value_loss * 2.
                # loss_record.append(loss.item())

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                self.opt.step()

        # np_loss = np.array(loss_record)
        # return np_loss.mean()

    def save(self):
        torch.save(self.net.state_dict(), model_file)

if __name__ == "__main__":

    # Environment
    env = Env()
    state = env.reset()

    # Agent
    agent = Agent()

    start_time = time.time()
    for episode in range(episodes):

        total_reward = 0
        steps = 0
        # loss = []
        state = env.reset()
        while True:
            action, a_logp = agent.act(state)
            next_state, reward, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            # env.render()
            # print(reward)

            if agent.store((state, action, a_logp, reward, next_state)):
                agent.update()
                print("INFO: Agent Updated time: ", time.time() - start_time)
                # mean_loss = agent.update()
                # loss.append(mean_loss)

            steps += 1
            total_reward += reward
            state = next_state
            if done:
                break

        # Finish an episode
        print("Episode: {:d} | Score: {:.2f} | Step: {:d} | Time: {:.2f}".format(
            episode, total_reward, steps, time.time() - start_time
        ))
        with open("log.txt", "a") as f:
            f.write(f"{episode} {total_reward}\n")
        # np_loss = np.array(loss)
        # print("Episode: {:d} | Score: {:.2f} | Loss: {:.2f} | Time: {:.2f}".format(
        #     episode, total_reward, np_loss.mean(), time.time() - start_time))
        # with open("log.txt", "a") as f:
        #     f.write(f"{episode} {total_reward} {np_loss.mean()}\n")
        if episode % checkpoints == 0:
            agent.save()
            print("INFO: Model saved at episode: ", episode)