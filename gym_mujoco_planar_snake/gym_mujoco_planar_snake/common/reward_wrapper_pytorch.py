import time
from gym.core import RewardWrapper


import torch
import torch.nn as nn
import torch.nn.functional as F



class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, net):
        RewardWrapper.__init__(self, venv)

        # TODO need seed for consistency

        self.venv = venv
        self.counter = 0

        self.net = net

        self.horizon = torch.zeros(50, 27)

        self.time_measure = []

    def step(self, action):


        #start = time.time()

        #model = self.model_optional

        obs, rews, news, infos = self.venv.step(action)

        self.counter += 1
        # sess = self.sess

        #print("Step", str(self.counter))

        #end = time.time()
        #used = end - start
        #print(used)
        #self.time_measure.append(used)

        network = True

        if network:
            horizon_long = torch.cat((self.horizon, torch.from_numpy(obs).float().unsqueeze(0)), 0)

            self.horizon = horizon_long[1:, :]
            #print("in net")


            #rews = self.model(self.horizon.view(1350))
            rews, _ = self.net.cum_return(self.horizon.view(1350))


            '''        if self.counter == 500:
            import numpy as np

            print(np.array(self.time_measure).mean())

            import sys
            sys.exit()'''

        # self.history = horizon

        # TODO return reward or abs_reward
        return obs, rews.item(), news, infos

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def reward(self, reward):
        return reward

