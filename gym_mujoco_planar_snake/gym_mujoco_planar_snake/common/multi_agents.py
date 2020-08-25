#!/usr/bin/env python
import tensorflow as tf
import torch
import numpy as np

# TODO eager exec
# tf.enable_eager_execution()

from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import learn
from baselines.bench.monitor import Monitor
from baselines import logger

import gym, logging

import argparse
import os
import os.path as osp
from time import ctime

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper
from gym_mujoco_planar_snake.common.custom_action_wrapper import ClipActionWrapper
from gym_mujoco_planar_snake.common.custom_observation_wrapper import GenTrajWrapper
from gym_mujoco_planar_snake.common.reward_wrapper import *
from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.env_wrapper import prepare_env
from gym_mujoco_planar_snake.common.misc_util import Configs


class PPOAgent(object):

    def __init__(self, env, id):

        self.id = str(id)


        with tf.variable_scope(str(id)):

                self.sess = U.make_session(num_cpu=1, make_default=False)
                self.sess.__enter__()
                self.sess.run(tf.initialize_all_variables())
                #U.initialize()
                with self.sess.as_default():
                    #env = ModelSaverWrapper(env, "/home/andreas/Desktop/obs_test", 1000)
                    self.env = env
                    self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                             ob_space=ob_space,
                                                                                             ac_space=ac_space,
                                                                                             hid_size=64,
                                                                                             num_hid_layers=2
                                                                                             )
        self.policy = None



    def learn(self, num_timesteps):
        with tf.variable_scope(str(self.id)):
            self.policy = learn(self.env, self.policy_func,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=2048,
                                clip_param=0.2,
                                entcoeff=0.0,
                                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear',
                                )
        self.sess.close()
        return self.policy

    def act(self, obs, stochastic=False):
        return self.policy.act(stochastic, obs)


class AgentSquad(object):

    def __init__(self, path_to_configs, learned_reward=False):

        # TODO test value generation
        # load configurations from yaml file
        configs = Configs(path_to_configs)

        # TODO get id from config file
        self.env_id = "Mujoco-planar-snake-cars-angle-line-v1"
        self.check_point = True

        #
        self.env = gym.make(self.env_id)

        # initial settings
        self.num_initial_agents = 5
        self.num_initial_timesteps = 1000 #int(3e5)

        # improved settings
        self.num_improved_agents = 3
        self.num_improved_timesteps = 3

        if learned_reward:
            self.num_agents = self.num_improved_agents
            self.num_timesteps = self.num_improved_timesteps
        else:
            self.num_agents = self.num_initial_agents
            self.num_timesteps = self.num_initial_timesteps


    def learn(self):

        wrapped_env = prepare_env(self.env)

        agents = [PPOAgent(env=wrapped_env, id=id) for id in range(self.num_agents)]

        for agent in agents:
            agent.learn(self.num_timesteps)


if __name__ == "__main__":
    "Not Callable"
