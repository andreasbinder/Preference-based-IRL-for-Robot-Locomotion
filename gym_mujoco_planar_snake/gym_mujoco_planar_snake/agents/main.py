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
import subprocess
from time import ctime

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper
from gym_mujoco_planar_snake.common.custom_action_wrapper import ClipActionWrapper
#from gym_mujoco_planar_snake.common.custom_observation_wrapper import GenTrajWrapper
from gym_mujoco_planar_snake.common.reward_wrapper import *
from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.ensemble import *
from gym_mujoco_planar_snake.common.results import *

from gym.core import ObservationWrapper
import numpy as np
import os
from time import ctime

# 3000 traj in one run
# 1000 epi, 1000000 ts -> 200 traj per
# if timestep > 200000 then choose 200 random numbers
#

class GenTrajWrapper(ObservationWrapper):

    def __init__(self, env, path, max_timesteps, num_traj_per_epoch = 10, name = "trajectories.npy"):
        ObservationWrapper.__init__(self, env=env)

        self.max_timesteps = max_timesteps
        self.last_e_steps = 0
        self.episodes = 1
        self.observations_list = []
        self.rewards_list = []
        self.trajectories = []

        self.path = path

        self.num_traj_per_epoch = num_traj_per_epoch


        self.name = name


    def store(self, observation, reward):
        self.observations_list.append(observation)
        self.rewards_list.append(reward)
        return observation, reward

    def step(self, action):
        self.last_e_steps += 1
        observation, reward, done, info = self.env.step(action)
        self.store(observation, reward)
        return observation, reward, done, info

    def reset(self, **kwargs):

        if self.last_e_steps > 0:

            starts = np.random.randint(0, 949, size=self.num_traj_per_epoch)
            starts.sort()

            #print(starts)
            for start in starts:
                # tuple of form ([50,27], [50])
                trajectory = np.array(self.observations_list)[start:start+50], np.sum(self.rewards_list[start:start+50])

                '''print(self.rewards_list[start:start+50])

                print(np.sum(self.rewards_list[start:start+50]))

                assert False, "test obs"'''

                self.trajectories.append(trajectory)

            self.observations_list = []
            self.rewards_list = []

            if self.last_e_steps == self.max_timesteps:

                path = self.path

                with open(os.path.join(path, self.name), 'wb') as f:
                    np.save(f, np.array(self.trajectories))
                    self.trajectories = []


        return self.env.reset(**kwargs)


class PPOAgent(object):

    def __init__(self, env):

        self.env = env

        self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name, ob_space=ob_space,
                                                                                 ac_space=ac_space,
                                                                                 hid_size=64, num_hid_layers=2
                                                                                 )
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





def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6
    parser.add_argument('--sfs', help='save_frequency_steps', type=int, default=int(5e4))  # 10000
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--log_dir', help='log directory', default='gym_mujoco_planar_snake/results/')
    parser.add_argument('--injured', type=bool, default=False)
    parser.add_argument('--clip_value', type=float, default=1.5)
    parser.add_argument('--fixed_joints', nargs='*', type=int, default=None)
    parser.add_argument('--custom_reward', type=bool, default=False)
    parser.add_argument('--reward_net_dir', default='gym_mujoco_planar_snake/log/PyTorch_Models/Thu Aug  6 20:30:12 2020/model')
    parser.add_argument('--name_dir', default=ctime())
    parser.add_argument('--save', default=True)
    parser.add_argument('--mode', default='pair')
    parser.add_argument('--monitor', default=True)

    # Step 1 args
    parser.add_argument('--num_initial_runs', type=int, default=5) # num agents
    parser.add_argument('--traj_per_episode', type=int, default=10)
    parser.add_argument('--num_initial_timesteps', type=int, default=int(3e5))
    parser.add_argument('--create_traj', type=str, default="")

    # Step 2 args
    parser.add_argument('--num_nets', type=int, default=5)
    parser.add_argument('--ranking_loss', type=int, help="Number of Trajectories to be compared at one time", default=2)
    parser.add_argument('--net_save_path', default='gym_mujoco_planar_snake/results/')
    parser.add_argument('--hparams_path', default="gym_mujoco_planar_snake/agents/hparams.json")

    # Step 3 args
    parser.add_argument('--num_improved_runs', type=int, default=5)
    #parser.add_argument('--traj_per_episode', type=int, default=10)
    parser.add_argument('--num_improved_timesteps', type=int, default=int(1e6))
    #parser.add_argument('--create_traj', type=str, default="")

    args = parser.parse_args()
    # time format time.ctime()[4:19]
    # cwd = os.path.join(os.getcwd(), "gym_mujoco_planar_snake/agents")

    # has environment name, contains, initial and improved runs
    base_log_dir = os.path.join(args.log_dir, args.env)

    ########################################################################################
    # Step 1a: create trajectories
    print("Start Data Loading/ Generation")
    template = 'python3 gym_mujoco_planar_snake/agents/run_ppo.py --env {env} --seed {seed} --num-timesteps {num_timesteps} --log_dir {log_dir}'


    initial_log_dir = os.path.join(base_log_dir, "initial_runs")

    if args.create_traj == "create":

        for seed in range(args.num_initial_runs):
            cmd = template.format(
                seed=seed,
                env=args.env,
                num_timesteps=args.num_initial_timesteps,
                log_dir=initial_log_dir
            )
            subprocess.run(cmd, cwd=".", shell=True)

    # Step 1b: load trajectories

    dirs = os.listdir(initial_log_dir)
    dirs.sort()

    trajectories = []
    true_rews = []


    for dir in dirs[-args.num_initial_runs:]:

        traj_path = os.path.join(initial_log_dir, dir, "trajectories.npy")

        with open(traj_path, 'rb') as f:
            traj = np.load(f, allow_pickle=True)


        true_rews.append(traj[:,1])
        trajectories.append(traj)

    data = np.concatenate(trajectories)

    initial_df = DataFrame(np.array(true_rews))

    ########################################################################################
    # Step 2: learn reward function
    print("Start IRL")
    # create new saving directory
    improved_log_dir = os.path.join(base_log_dir, "improved_runs")
    if not os.path.exists(improved_log_dir):
        os.mkdir(improved_log_dir)
    net_save_path = os.path.join(improved_log_dir, "ensemble" + ctime()[4:19]).replace(" ", "_")

    os.mkdir(net_save_path)
    # create Ensemble mit args

    ensemble = Ensemble(args, net_save_path)
    ensemble.fit(data)

    # print(net_save_path)

    ########################################################################################
    # Step 3: run ppo on learnt reward function

    template = 'python3 gym_mujoco_planar_snake/agents/run_ppo.py --env {env} --custom_reward {custom_reward} --seed {seed} --num-timesteps {num_timesteps} --log_dir {log_dir} --num_nets {num_nets}'


    for seed in range(args.num_improved_runs):
        cmd = template.format(
            seed=seed,
            env=args.env,
            num_timesteps=args.num_improved_timesteps,
            log_dir=str(net_save_path),
            num_nets=args.num_nets,
            custom_reward=True
        )
        subprocess.run(cmd, cwd=".", shell=True)

    dirs = os.listdir(net_save_path)
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    dirs.sort()

    preds = []
    true_rews = []

    for dir in dirs:
        traj_path = os.path.join(net_save_path, dir, "result.npy")

        with open(traj_path, 'rb') as f:
            results = np.load(f, allow_pickle=True)

        true_rews.append(results[:, 0])
        preds.append(results[:, 1])

    #improved_df = DataFrame(np.array(true_rews), np.array(preds), initial=False)

    '''print(initial_df.get_mean_true_rew())
    print(initial_df.get_max_true_rew())'''


















        #print(dirs[-args.num_initial_runs:])




    '''U.make_session(num_cpu=1, make_default=True)  # interactive

    model_dir = set_configs(args)

    env = gym.make(args.env)
    env.seed(args.seed)

    #
    env = GenTrajWrapper(env, args.log_dir + "/" + args.name_dir, args.num_timesteps)

    env = prepare_env(args=args,
                      model_dir=model_dir)

    agent = PPOAgent(env=env)

    agent.learn(args.num_timesteps)

    with open(os.path.join(args.log_dir, args.name_dir, "rewards.npy"), 'wb') as f:
        np.save(f, env.get_episode_rewards())'''



if __name__ == '__main__':
    main()
