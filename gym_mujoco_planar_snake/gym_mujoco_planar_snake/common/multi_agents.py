#!/usr/bin/env python
import tensorflow as tf
import torch


from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import learn
from baselines.bench.monitor import Monitor
from baselines import logger

import gym, logging
import os


#from gym_mujoco_planar_snake.common.reward_wrapper import *
#from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.env_wrapper import prepare_env, ModelSaverWrapper
from gym_mujoco_planar_snake.common.misc_util import Configs


class PPOAgent(object):

    def __init__(self, env, id, log_dir, sfs, save_check_points, learned_reward, configs):

        self.id = str(id)


        with tf.variable_scope(self.id):

                self.sess = U.make_session(num_cpu=1, make_default=False)
                self.sess.__enter__()
                self.sess.run(tf.initialize_all_variables())
                #U.initialize()

                with self.sess.as_default():

                    wrapped_env = prepare_env(env,
                                              save_check_points,
                                              log_dir,
                                              sfs,
                                              id,
                                              learned_reward,
                                              configs)
                    self.env = wrapped_env


                    #self.env.seed(configs.get_seed())
                    self.env.seed(id)

                    # TODO check if I need to seed action_space too
                    #self.env.action_space.seed(id)


                    self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                             ob_space=ob_space,
                                                                                             ac_space=ac_space,
                                                                                             hid_size=64,
                                                                                             num_hid_layers=2
                                                                                             )
        self.policy = None


    def learn(self, num_timesteps):

        with tf.variable_scope(str(self.id)):

            print("Agent %s starts learning" % self.id)

            self.policy = learn(self.env, self.policy_func,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=2048,
                                clip_param=0.2,
                                entcoeff=0.0,
                                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear',
                                )

            # print(self.env.get_episode_rewards())

        self.sess.close()
        return self.policy

    def act(self, obs, stochastic=False):
        return self.policy.act(stochastic, obs)


class AgentSquad(object):

    def __init__(self, configs, learned_reward, log_dir):

        # TODO test value generation
        # load configurations from yaml file
        # configs = Configs(path_to_configs)


        # initialize env
        self.configs = configs
        self.log_dir = log_dir
        self.env_id = configs.get_env_id() # "Mujoco-planar-snake-cars-angle-line-v1"
        self.env = gym.make(self.env_id)


        self.learned_reward = learned_reward

        if learned_reward:
            self.num_agents = configs.get_num_improved_agents()
            self.num_timesteps = configs.get_num_improved_timesteps()
            self.save_check_points = configs.get_save_improved_checkpoints()
            self.sfs = configs.get_improved_sfs()
        else:
            self.num_agents = configs.get_num_initial_agents()
            self.num_timesteps = configs.get_num_initial_timesteps()
            self.save_check_points = configs.get_save_initial_checkpoints()
            self.sfs = configs.get_initial_sfs()



    def learn(self):

        num_prev_agents = self.configs.get_num_initial_agents() if self.learned_reward else 0
        agents = [PPOAgent(env=self.env,
                           id=id,
                           log_dir=self.log_dir,
                           sfs=self.sfs,
                           save_check_points=self.save_check_points,
                           learned_reward=self.learned_reward,
                           configs=self.configs)

                  for id in range(num_prev_agents, self.num_agents + num_prev_agents)]

        for agent in agents:
            agent.learn(self.num_timesteps)


if __name__ == "__main__":
    "Not Callable"
