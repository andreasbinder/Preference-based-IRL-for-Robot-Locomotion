import argparse
from time import ctime
import os

import shutil

from gym.core import ObservationWrapper
from baselines.common import tf_util as U

from baselines import logger
import os.path as osp
import os
import time
from time import ctime

import time
from gym.core import RewardWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

from time import ctime

from baselines.common.running_mean_std import RunningMeanStd
from baselines.bench.monitor import Monitor, load_results, get_monitor_files

#from gym_mujoco_planar_snake.common.reward_nets import *
from src.common.ensemble import Ensemble


import random

from src.common.multi_agents import AgentSquad
from src.common.misc_util import Configs
#from gym_mujoco_planar_snake.common.evaluate import DataFrame
from src.common.ensemble import Ensemble as RFA
from src.common.env_wrapper import MyMonitor

from src.utils.model_saver import ModelSaverWrapper
from src.utils.agent import PPOAgent



import src.common.data_util as data_util

from baselines.common import set_global_seeds

import tensorflow as tf


import torch
import numpy as np

def set_seeds(seed):
    #seed = configs.get_seed()
    # tensorflow, numpy, random(python)

    #set_global_seeds(seed)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # TODO
    torch.manual_seed(seed)

class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, nets, ctrl_coeff):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.counter = 0
        self.ctrl_coeff = ctrl_coeff
        self.nets = nets
        #self.id = id

        #self.save_dir = save_dir
        #self.default_reward_dir = default_reward_dir

        self.cliprew = 10.
        self.epsilon = 1e-8

        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(nets))]

        # TODO compare reward development
        #self.max_timesteps = max_timesteps
        self.rewards = []

        self.reward_list = []


        # TODO apply sigmoid
        self.sigmoid = nn.Sigmoid()




    def step(self, action):

        self.counter += 1

        obs, rews, news, infos = self.venv.step(action)

        # acs = self.last_actions

        # TODO save true reward

        r_hats = 0.

        for net, rms in zip(self.nets, self.rew_rms):
            # Preference based reward
            with torch.no_grad():
                pred_rews = net.cum_return(torch.from_numpy(obs).float())
                # TODO apply sigmoid
                # pred_rews = self.sigmoid(pred_rews)
            r_hat = pred_rews.item()

            # Normalization only has influence on predicted reward
            # Normalize TODO try without, 2. run has no running mean
            rms.update(np.array([r_hat]))
            r_hat = np.clip(r_hat / np.sqrt(rms.var + self.epsilon), -self.cliprew, self.cliprew)

            # Sum-up each models' reward
            r_hats += r_hat

        pred = r_hats / len(self.nets) - self.ctrl_coeff * np.sum(action ** 2)

        #pred = r_hats / len(self.nets) * np.abs(1.0 - infos["power_normalized"])

        # TODO normalize

        self.store_rewards(rews, pred)




        # TODO return reward or abs_reward
        return obs, pred, news, infos

    def reset(self, **kwargs):




        '''if self.counter == self.max_timesteps:
            with open(os.path.join(self.save_dir, "results.npy"), 'wb') as f:
                np.save(f, np.array(self.rewards))

            with open(os.path.join(self.default_reward_dir,
                                   "results" + str(self.id) + ctime()[4:19].replace(" ", "_") + ".npy"), 'wb') as f:
                np.save(f, np.array(self.rewards))

            self.rewards = []'''

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))


# gen data here?
# ppo in common


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)

    configs.data["reward_learning"] = configs.data["train"]

    configs_tmp = configs
    configs = configs.data["train"]



    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    set_seeds(configs["seed"])

    TRAIN_PATH = "/home/andreas/Desktop/2020-10-03_00-13-18_0.5/train.npy"

    with open(TRAIN_PATH, 'rb') as f:
        TRAIN = np.load(f, allow_pickle=True)

    # main process
    train_set = data_util.generate_dataset_from_full_episodes(TRAIN, 50, 100)

    net_path = "/tmp/"

    ensemble = RFA(configs_tmp, net_path)

    ensemble.fit(train_set)

    net = RFA.load(net_path, 1)

    # seeds
    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'

    SAVE_DIR = "/tmp/pi_hat/"

    import gym
    import os, datetime
    from baselines.ppo1 import mlp_policy

    SAVE_DIR = os.path.join(SAVE_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(SAVE_DIR)

    num_runs = 1
    NUM_TIMESTEPS = 1000000
    sfs = 100000
    ctrl_coeff = 0.0

    for index in range(1, num_runs + 1):



        with tf.variable_scope(str(index)):

            SAVE = os.path.join(SAVE_DIR, str(index))

            env = gym.make(ENV_ID)

            env.seed(configs["seed"])

            env = ModelSaverWrapper(env, SAVE, sfs)
            env = Monitor(env, SAVE)
            #self, venv, nets, max_timesteps, save_dir, ctrl_coeff, default_reward_dir, id)
            env = MyRewardWrapper(env, net, ctrl_coeff)



            policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                ob_space=ob_space,
                                                                                ac_space=ac_space,
                                                                                hid_size=64,
                                                                                num_hid_layers=2
                                                                                )
            pi = policy_fn("pi", env.observation_space, env.action_space)

            sess = U.make_session(num_cpu=1, make_default=False)
            sess.__enter__()
            sess.run(tf.initialize_all_variables())
            with sess.as_default():

                agent = PPOAgent(env, pi, policy_fn)
                agent.learn(NUM_TIMESTEPS)








