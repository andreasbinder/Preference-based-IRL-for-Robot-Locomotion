# take trajectories from openai log dir and put it into internal dir (traj before trex)

# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from sklearn.model_selection import ParameterGrid

from gym.envs.registration import register



import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper

from gym_mujoco_planar_snake.common import my_tf_util

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import gym_mujoco_planar_snake.benchmark.plots as import_plots

from gym_mujoco_planar_snake.agents.Dataset import SubTrajectory, Dataset, Trajectory


def policy_fn(name, ob_space, ac_space):
    from baselines.ppo1 import mlp_policy
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

    # from baselines.ppo1 import cnn_policy
    # return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)


def run_environment_episode(env, pi, seed, model_file, start, horizon, max_timesteps, render, stochastic=False):

    # initial
    number_of_timestep = 0
    done = False
    l_obs = []
    l_acs = []
    l_rews = []
    cum_reward = 0

    # load model
    my_tf_util.load_state(model_file)

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    # first observation
    obs = env.reset()

    # max_timesteps is set to 2000
    # start recording at timestep start for horizon timestep long
    #while (not done) and number_of_timestep < max_timesteps:
    while (not done) and number_of_timestep < start + horizon:

        action = pi.act(stochastic, obs)
        action = action[0]

        if number_of_timestep >= start :
            l_obs.append(obs)
            l_acs.append(action)
            l_rews.append(reward)


        obs, reward, done, info = env.step(action)
        #cum_reward += reward


        if render:
            env.render()

        number_of_timestep += 1

    return done, number_of_timestep, None, cum_reward, l_obs, l_acs, l_rews


def model_file_from_trajectory(trajectory):

    model_file = str(trajectory.elements[2])
    assert model_file.endswith('.data-00000-of-00001'), "No such file"
    # delete ending
    return model_file[:-20]


def initialize_dataset(env_id, original_dataset, horizon, num_samples_per_trajectories=10,  seed=0):

    print("Initializing Dataset")

    from random import randint
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sess = U.make_session(num_cpu=1)
    sess.__enter__()

    env = gym.make(env_id)
    env.unwrapped.metadata['target_v'] = 0.15 # not sure if needed

    #########################################################################
    # TODO:                                                                 #
    # Just use first trajectory of Dataset                                                                #
    #########################################################################
    #original_dataset = original_dataset[0]

    max_timesteps = env._max_episode_steps



    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # create policy
    pi = policy_fn('pi0', env.observation_space, env.action_space)

    min_start, max_start = 0 , env._max_episode_steps -1 - horizon

    final_dataset = []


    for trajectory in original_dataset:

        model_file = model_file_from_trajectory(trajectory)

        for _ in range(num_samples_per_trajectories):

            start = randint(min_start, max_start)

            done, number_of_timesteps, info_collector, cum_reward, l_obs, l_acs, l_rews = \
                run_environment_episode(env, pi, seed, model_file, start=start, horizon=horizon,
                                        max_timesteps=2000, render=True, stochastic=False)

            # cum_reward übergeben
            subtrajectory = SubTrajectory(rewards=l_rews, actions=l_acs, observations=l_obs)

            final_dataset.append(subtrajectory)

    return Dataset(final_dataset)


def adjust_dataset(original_dataset, exclude_joints=None, min_time_step=100000, max_time_step=800000, max_num_traj=1000000):

    # create copy of dataset, list copy function
    dataset = original_dataset.copy()

    # make Dataset iterable
    for trajectory in dataset:

        if exclude_joints is not None and trajectory.fixed_joints in exclude_joints:
            dataset.remove(trajectory)

        elif min_time_step is not None and trajectory.time_step < min_time_step:
            dataset.remove(trajectory)

        elif max_time_step is not None and trajectory.time_step > max_time_step:
            dataset.remove(trajectory)

    if len(dataset) > max_num_traj:
        return original_dataset, dataset.sample(max_num_traj)
    else:
        return original_dataset, dataset


def create_dataset(log_dir):
    from pathlib import Path

    cwd = Path.cwd()
    remainder = 'models/Mujoco-planar-snake-cars-angle-line-v1/ppo/'
    log_dir = Path(cwd, log_dir)

    # eg gym_mujoco_planar_snake/log/clip_test/InjuryIndex_3/Sat Jun 27 01:09:59 2020
    joint_dirs = [x for x in log_dir.iterdir() if x.is_dir()]

    # eg adds models/Mujoco-planar-snake-cars-angle-line-v1/ppo
    dirs = [Path(i, remainder) for x in joint_dirs if x.is_dir() for i in x.iterdir()]

    # test_dir = ./InjuryIndex_3/Thu Jun 25 23:56:07 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo
    test_dir = dirs[0]
    dirs = dirs[:2]

    #print("dir",str(dirs[0]), type(str(dirs[0])))

    min_val, max_val = 10000, 1000000
    trajectories = []
    for directory in dirs:

        for index, time_step in enumerate(range(min_val, max_val + 1, 10000)):

            off_set = 9 - len(str(time_step))

            goal = "0" * off_set + str(time_step)

            trajectory = Trajectory(
                elements= [directory.glob("*" + goal + '.index').__next__(),
                           directory.glob("*" + goal + '.meta').__next__(),
                           directory.glob("*" + goal + '.data-00000-of-00001').__next__()],
                path=str(directory),
                time_step=time_step
            )


            trajectories.append(trajectory)

    dataset = Dataset(trajectories)

    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_samples_per_trajectories', type=int, default=int(10))
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--log_dir', help='log directory', default='gym_mujoco_planar_snake/log/clip_test/')
    parser.add_argument('--save_path', help='save directory for prepared subtrajectories', default='gym_mujoco_planar_snake/log/SubTrajectoryDataset')

    args = parser.parse_args()
    logger.configure()
    gym.logger.setLevel(logging.WARN)

    # CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


    #########################################################################
    # TODO:                                                                 #
    #                                                                       #
    #########################################################################

    save_path = "/home/andreas/Desktop/"
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    d = create_dataset(args.log_dir)

    d.shuffle()

    # possibility to limit data_set
    original_d, d = adjust_dataset(d, max_num_traj=1)
    print("Number of Trajectories after adjust: ", str(len(d)))

    # next initialize dataset(run episodes to fill reward,actions, observations etc)
    trainable_dataset = initialize_dataset(env_id=args.env, original_dataset=d, horizon=500, seed=args.seed)

    trainable_dataset.save(path=save_path)


if __name__ == '__main__':
    main()