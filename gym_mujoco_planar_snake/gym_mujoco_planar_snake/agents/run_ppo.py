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
from gym_mujoco_planar_snake.common.custom_observation_wrapper import CustomObservationWrapper
from gym_mujoco_planar_snake.common.reward_wrapper_pytorch import *
from gym_mujoco_planar_snake.common.reward_nets import *

from gym_mujoco_planar_snake.common.performance_checker import evaluate_policy


class PPOAgent(object):

    def __init__(self, env, model_dir, reward_net_dir, sfs,
                 joints=None,
                 clip_value=None,
                 wrap_action=False,
                 wrap_reward=False,
                 wrap_monitor=True
                 ):
        # TODO set seed

        self.joints = joints
        self.clip_value = clip_value
        self.model_dir = model_dir
        self.sfs = sfs
        self.reward_net_dir = reward_net_dir

        self.wrap_env = lambda x: self.wrap(env=x,
                                            wrap_action=wrap_action,
                                            wrap_reward=wrap_reward,
                                            wrap_monitor=wrap_monitor
                                            )

        #self.env = self.wrap_env(env)

        self.env = env

        self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                                                                 ac_space=ac_space,
                                                                                 hid_size=64, num_hid_layers=2)
        self.policy = None

    def learn(self, num_timesteps):
        self.policy = learn(self.env, self.policy_func,
                            max_timesteps=num_timesteps,
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2,
                            entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                            gamma=0.99, lam=0.95,
                            schedule='linear',
                            )
        return self.policy

    def act(self, obs, stochastic=False):
        return self.policy.act(stochastic, obs)

    def wrap(self, env, wrap_action=False, wrap_reward=True, wrap_observation=True, wrap_monitor=True):

        if wrap_monitor:
            env = Monitor(env, None)

        if wrap_action:
            assert self.joints[self.joints >= 0].size == 0 and self.joints[
                self.joints <= 7].size == 0, "Joint Index does not exist"
            env = ClipActionWrapper(env, self.clip_value, self.joints)

        if wrap_reward:
            net = Net()
            net.load_state_dict(torch.load(self.reward_net_dir))
            # env = HorizonRewardWrapper_v3(env, self.reward_net_dir, model)
            env = MyRewardWrapper(env, net)

        if wrap_observation:
            log_dir = osp.join(logger.get_dir(), 'log_ppo')
            logger.log("log_dir: %s" % log_dir)
            env = bench.Monitor(env, log_dir)
            env = ModelSaverWrapper(env, self.model_dir, self.sfs)

        return env



def set_configs(args):
    gym.logger.setLevel(logging.WARN)
    logger.configure(dir=args.log_dir + "/" + args.name_dir)
    #logger.configure(dir=args.log_dir + "/" + ctime())
    set_global_seeds(args.seed)
    torch.manual_seed(0)

    model_dir = None
    if args.save:
        name = 'ppo'
        model_dir = osp.join(logger.get_dir(), 'models')
        os.mkdir(model_dir)
        model_dir = ModelSaverWrapper.gen_model_dir_path(model_dir, args.env, name)
        logger.log("model_dir: %s" % model_dir)
    return model_dir


def prepare_env(args, wrap_action, wrap_reward, wrap_observation, wrap_monitor, model_dir):

    env = gym.make(args.env)
    env.seed(args.seed)



    joints = np.array(args.fixed_joints)


    '''if wrap_monitor:
        env = Monitor(env, None)'''

    if wrap_action:
        assert joints[joints >= 0].size == 0 and joints[joints <= 7].size == 0, "Joint Index does not exist"
        env = ClipActionWrapper(env, args.clip_value, joints)

    if wrap_reward:
        #net = Net() if args.mode == 'pair' else TripletNet()
        net = SingleStepPairNet()
        net.load_state_dict(torch.load(args.reward_net_dir))
        # env = HorizonRewardWrapper_v3(env, self.reward_net_dir, model)
        #env = MyRewardWrapper(env, net)
        env = SingleStepRewardWrapper(env, net)

    if wrap_observation:
        log_dir = osp.join(logger.get_dir(), 'log_ppo')
        logger.log("log_dir: %s" % log_dir)
        env = bench.Monitor(env, log_dir)
        env = ModelSaverWrapper(env, model_dir, args.sfs)

    '''if args.save:
        env = ModelSaverWrapper(env, model_dir, args.sfs)'''

    env = Monitor(env, None)

    return env

def final_logs(args, env, model_dir):

    if args.custom_reward:
        # TODO use wrapper
        rewards = env.get_episode_rewards()

        with open(os.path.join(args.log_dir + "/" + args.name_dir, 'rewards.npy'), 'wb') as f:
            np.save(f, rewards)

    else:
        rewards = env.get_episode_rewards()

        with open(os.path.join(args.log_dir + "/" + args.name_dir, 'rewards.npy'), 'wb') as f:
            np.save(f, rewards)





def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6
    parser.add_argument('--sfs', help='save_frequency_steps', type=int, default=10000)  # for mujoco
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--log_dir', help='log directory', default='gym_mujoco_planar_snake/log/initial_PPO_runs/')
    parser.add_argument('--injured', type=bool, default=False)
    parser.add_argument('--clip_value', type=float, default=1.5)
    parser.add_argument('--fixed_joints', nargs='*', type=int, default=None)
    parser.add_argument('--custom_reward', type=bool, default=False)
    parser.add_argument('--reward_net_dir', default='gym_mujoco_planar_snake/log/PyTorch_Models/Thu Aug  6 20:30:12 2020/model')
    parser.add_argument('--name_dir', default=ctime())
    parser.add_argument('--save', default=True)
    parser.add_argument('--mode', default='pair')




    args = parser.parse_args()

    ##################################################################################

    #args.log_dir = 'gym_mujoco_planar_snake/log/PPOAgent_test/'
    '''args.sfs = 50000
    args.num_timesteps = 7000
    reward_net_dir = '/home/andreas/Desktop/Wed Aug  5 13:56:18 2020/Wed Aug  5 13:56:18 2020'
    args.custom_reward = True'''

    #args.num_timesteps = 1000000

    ##################################################################################
    U.make_session(num_cpu=1, make_default=True)  # interactive

    model_dir = set_configs(args)

    env = prepare_env(args=args,
                      wrap_action=False,
                      wrap_reward=args.custom_reward,
                      wrap_observation=args.save,
                      wrap_monitor=True,
                      model_dir=model_dir
                      )

    #env = CustomObservationWrapper(env, 500)

    #env = Monitor(env, None)

    agent = PPOAgent(env=env,
                     model_dir=model_dir,
                     reward_net_dir=args.reward_net_dir,
                     sfs=args.sfs,
                     wrap_reward=args.custom_reward,
                     wrap_monitor=False)

    pi = agent.learn(args.num_timesteps)

    #pi = agent.policy

    #print(pi)

    #assert False, "policy"

    # TODO as numpy

    #logger.log(env.get_episode_rewards())
    #logger.log(env.get_episode_lengths())

    '''print(env.get_episode_rewards())
    print(env.get_episode_lengths())'''

    #evaluate_policy(args.env, pi, custom=False)

    #final_logs(args,env, model_dir)


if __name__ == '__main__':
    main()
