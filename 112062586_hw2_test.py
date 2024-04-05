import numpy as np
import cv2
import gym
import random
import torch
import torch.nn as nn

class Dueling_DQN(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(Dueling_DQN, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 4, 2)
        self.layer3 = nn.Conv2d(64, 64, 3, 1)
        self.fc = nn.Linear(3136, 512) #torch.Size([1, 64, 7, 7])
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.layer3, self.fc, self.q, self.v)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = x.view(-1, 3136)
        x = torch.relu(self.fc(x))

        adv = self.q(x)
        v = self.v(x)

        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])

        #adv_average = torch.mean(adv, dim=1, keepdim=True)
        #q = v + (adv - adv_average)

        return q

class Agent():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_space = gym.spaces.Discrete(12)
        self.width = 84
        self.height = 84
        self.q = Dueling_DQN(1, 12, self.device).to(self.device)
        self.q.load_state_dict(torch.load("112062586_hw2_data.py", map_location=torch.device(self.device)))


    def wrap(self, observation):

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        observation = np.expand_dims(observation, -1)

        return observation

    def arange(self, observation):
        if not type(observation) == "numpy.ndarray":
            observation = np.array(observation)
        assert len(observation.shape) == 3
        ret = np.transpose(observation, (2, 0, 1))

        return np.expand_dims(ret, 0)

    def act(self, observation):

        s = self.wrap(observation)
        s = self.arange(s)

        #if random.random() <= 0.01:
            #return self.action_space.sample()
        #else:
        if self.device == "cpu":
            return np.argmax(self.q(s).detach().numpy())
        else:
            return np.argmax(self.q(s).cpu().detach().numpy())




