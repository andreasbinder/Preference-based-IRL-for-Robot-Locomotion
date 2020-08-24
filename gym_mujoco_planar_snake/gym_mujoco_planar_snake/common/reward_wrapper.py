import time
from gym.core import RewardWrapper


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

from time import ctime

from baselines.common.running_mean_std import RunningMeanStd

from gym_mujoco_planar_snake.common.reward_nets import *


class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, nets, max_timesteps, log_dir, name_dir, ctrl_coeff):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.counter = 0
        self.ctrl_coeff = ctrl_coeff
        self.nets = nets



        self.cliprew = 10.
        self.epsilon = 1e-8

        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(nets))]

        # TODO compare reward development
        self.max_timesteps = max_timesteps
        self.rewards = []
        self.path = log_dir
        self.name_dir = name_dir


        self.reward_list = []


    def step(self, action):

        self.counter += 1

        obs, rews, news, infos = self.venv.step(action)

        #acs = self.last_actions

        # TODO save true reward

        r_hats = 0.

        for net,rms in zip(self.nets,self.rew_rms):
            # Preference based reward
            with torch.no_grad():
                pred_rews, _ = net.cum_return(torch.from_numpy(obs).float())
            r_hat = pred_rews.item()

            # Normalization only has influence on predicted reward
            # Normalize TODO try without, 2. run has no running mean
            rms.update(np.array([r_hat]))
            r_hat = np.clip(r_hat/ np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)

            # Sum-up each models' reward
            r_hats += r_hat

        pred = r_hats / len(self.nets) - self.ctrl_coeff*np.sum(action**2)


        # TODO normalize


        self.store_rewards(rews, pred)

        #self.reward_list.append(rews.item())

        # TODO render for debugging
        # do rendering by saving observations or actions
        '''if self.counter >= 200000:
            self.venv.render()'''

        # TODO return reward or abs_reward
        return obs, pred, news, infos

    def reset(self, **kwargs):


        if self.counter == self.max_timesteps:

            with open(os.path.join(self.path,  self.name_dir, "results.npy"), 'wb') as f:
                np.save(f, np.array(self.rewards))
                self.rewards = []

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))


class SingleStepRewardWrapper(RewardWrapper):

    def __init__(self, venv, net, max_timesteps, log_dir):
        RewardWrapper.__init__(self, venv)



        self.venv = venv
        self.counter = 0
        self.net = net

        # TODO compare reward development
        self.max_timesteps = max_timesteps
        self.rewards = []

        # TODO
        #self.path = "gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/"
        self.path = log_dir

        self.reward_list = []


    def step(self, action):

        self.counter += 1

        obs, rews, news, infos = self.venv.step(action)

        # TODO save true reward



        with torch.no_grad():
            pred_rews, _ = self.net.cum_return(torch.from_numpy(obs).float())


        self.store_rewards(rews, pred_rews.item())

        #self.reward_list.append(rews.item())

        # TODO render for debugging
        # do rendering by saving observations or actions
        '''if self.counter >= 200000:
            self.venv.render()'''

        # TODO return reward or abs_reward
        return obs, pred_rews.item(), news, infos

    def reset(self, **kwargs):


        if self.counter == self.max_timesteps:

            with open(os.path.join(self.path,  ctime() + str(self.counter) +'.npy'), 'wb') as f:
                np.save(f, np.array(self.rewards))
                self.rewards = []

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))


class SingleStepNormalizedRewardWrapper(RewardWrapper):

    def __init__(self, venv, net, max_timesteps):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.counter = 0
        self.net = net

        # TODO
        self.rms = RunningMeanStd(shape=())
        self.cliprew = 10.
        self.epsilon = 1e-8



        # TODO compare reward development
        self.max_timesteps = max_timesteps
        self.rewards = []
        self.path = "gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/"


        self.reward_list = []


    def step(self, action):

        self.counter += 1

        obs, rews, news, infos = self.venv.step(action)

        # TODO save true reward



        with torch.no_grad():
            pred_rews, _ = self.net.cum_return(torch.from_numpy(obs).float())


        r_hat = pred_rews.item()

        # TODO normalize
        self.rms.update(np.array([r_hat]))
        r_hat = np.clip(r_hat / np.sqrt(self.rms.var + self.epsilon), -self.cliprew, self.cliprew)



        self.store_rewards(rews, r_hat)

        #self.reward_list.append(rews.item())

        # TODO render for debugging
        # do rendering by saving observations or actions
        '''if self.counter >= 200000:
            self.venv.render()'''

        # TODO return reward or abs_reward
        return obs, r_hat, news, infos

    def reset(self, **kwargs):


        if self.counter == self.max_timesteps:

            with open(os.path.join(self.path,  ctime() + str(self.counter) +'.npy'), 'wb') as f:
                np.save(f, np.array(self.rewards))
                self.rewards = []

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))


class EnsembleNormalizedRewardWrapper(RewardWrapper):

    def __init__(self, venv, net, max_timesteps):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.counter = 0
        self.net = net

        # TODO
        #self.rms = RunningMeanStd(shape=())
        self.cliprew = 10.
        self.epsilon = 1e-8

        # online dataset cross
        '''net_paths = [
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 11:58:06 2020/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 11:58:44 2020/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 11:59:08 2020/model"
        ]'''

        # offline dataset cross
        net_paths = [
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 22:05:24 2020/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 22:05:41 2020/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 22:18:19 2020/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Tue Aug 18 22:18:58 2020/model"

        ]

        # offline dataset hinge
        net_paths = [
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/hinge_v1/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/hinge_v2/model",
            "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/hinge_v3/model"

        ]


        net1 = SingleStepPairNet()
        net1.load_state_dict(torch.load(net_paths[0]))

        net2 = SingleStepPairNet()
        net2.load_state_dict(torch.load(net_paths[1]))

        net3 = SingleStepPairNet()
        net3.load_state_dict(torch.load(net_paths[2]))

        '''net4 = SingleStepPairNet()
        net4.load_state_dict(torch.load(net_paths[3]))'''


        #self.nets = [net1, net2, net3, net4]

        self.nets = [net1, net2, net3]

        #for path in net_paths:


        #self.nets = [net.load_state_dict(torch.load(path)) for path in net_paths]

        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(net_paths))]



        # TODO compare reward development
        self.max_timesteps = max_timesteps
        self.rewards = []
        self.path = "gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/"


        self.reward_list = []


    def step(self, action):

        self.counter += 1

        obs, rews, news, infos = self.venv.step(action)

        # TODO save true reward

        r_hats = 0.

        for net,rms in zip(self.nets,self.rew_rms):
            # Preference based reward
            with torch.no_grad():
                pred_rews, _ = net.cum_return(torch.from_numpy(obs).float())
            r_hat = pred_rews.item()

            # Normalization only has influence on predicted reward
            # Normalize TODO try without, 2. run has no running mean
            '''rms.update(np.array([r_hat]))
            r_hat = np.clip(r_hat/ np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)'''

            # Sum-up each models' reward
            r_hats += r_hat

        pred = r_hats / len(self.nets) #- self.ctrl_coeff*np.sum(acs**2,axis=1)


        # TODO normalize





        self.store_rewards(rews, pred)

        #self.reward_list.append(rews.item())

        # TODO render for debugging
        # do rendering by saving observations or actions
        '''if self.counter >= 200000:
            self.venv.render()'''

        # TODO return reward or abs_reward
        return obs, pred, news, infos

    def reset(self, **kwargs):


        if self.counter == self.max_timesteps:

            with open(os.path.join(self.path,  ctime() + str(self.counter) +'.npy'), 'wb') as f:
                np.save(f, np.array(self.rewards))
                self.rewards = []

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))

