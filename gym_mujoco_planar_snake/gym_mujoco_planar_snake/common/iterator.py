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

from gym_mujoco_planar_snake.common.dataset import SubTrajectory, Dataset, Trajectory


class Iterator(object):
    def __init__(self,  seed=0, max_number_trajectories=1e6,
                 num_samples_per_trajectories=10, log_dir="gym_mujoco_planar_snake/log/initial_PPO_runs/",
                 save_path="gym_mujoco_planar_snake/log/SubTrajectoryDataset"):
        self.batch_size = None
        self.env = None
        self.seed = seed
        self.max_number_trajectories = max_number_trajectories
        self.num_samples_per_trajectories = num_samples_per_trajectories
        self.log_dir = log_dir
        self.save_path = save_path
        self.size_dataset = None

        #logger.configure()
        gym.logger.setLevel(logging.WARN)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def policy_fn(self, name, ob_space, ac_space):
        from baselines.ppo1 import mlp_policy
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    def run_environment_episode(self, env, pi, seed, model_file, start, horizon, max_timesteps, render, stochastic=False):
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
        # while (not done) and number_of_timestep < max_timesteps:
        while (not done) and number_of_timestep < start + horizon:

            action = pi.act(stochastic, obs)
            action = action[0]

            if number_of_timestep >= start:
                l_obs.append(obs)
                l_acs.append(action)


            obs, reward, done, info = env.step(action)
            # cum_reward += reward

            if number_of_timestep >= start:
                l_rews.append(reward)


            if render:
                env.render()

            number_of_timestep += 1

        return done, number_of_timestep, None, cum_reward, l_obs, l_acs, l_rews

    def model_file_from_trajectory(self, trajectory):
        model_file = str(trajectory.elements[2])
        assert model_file.endswith('.data-00000-of-00001'), "No such file"
        # delete ending
        return model_file[:-20]

    def initialize_dataset_alternative(self, env_id, batch_size, original_dataset, horizon, num_samples_per_trajectories=10, seed=0, render=False):
        print("Initializing Dataset")

        from random import randint
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)
        env.unwrapped.metadata['target_v'] = 0.15  # not sure if needed

        #########################################################################
        # TODO:                                                                 #
        # Just use first trajectory of Dataset                                                                #
        #########################################################################
        # original_dataset = original_dataset[0]

        max_timesteps = env._max_episode_steps

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # create policy
        pi = self.policy_fn('pi0', env.observation_space, env.action_space)

        min_start, max_start = 0, env._max_episode_steps - 1 - horizon

        final_dataset = []

        size_dataset = len(original_dataset)
        for i, trajectory in enumerate(original_dataset):

            model_file = self.model_file_from_trajectory(trajectory)

            for j in range(num_samples_per_trajectories):
                start = randint(min_start, max_start)

                done, number_of_timesteps, info_collector, cum_reward, l_obs, l_acs, l_rews = \
                    self.run_environment_episode(env, pi, seed, model_file, start=start, horizon=horizon,
                                            max_timesteps=2000, render=render, stochastic=False)

                # cum_reward übergeben
                subtrajectory = SubTrajectory(rewards=l_rews, actions=l_acs, observations=l_obs)

                final_dataset.append(subtrajectory)

                if len(final_dataset) % batch_size == 0:
                    yield Dataset(final_dataset)
                    final_dataset.clear()

        # return rest of data
        yield Dataset(final_dataset)

                #print("Subtrajectory: ", str(j + 1), "/", str(num_samples_per_trajectories), ", Trajectory", str(i + 1),
                #      "/", str(size_dataset))

        #return Dataset(final_dataset)

    def adjust_dataset(self, original_dataset, exclude_joints=None, min_time_step=100000, max_time_step=800000,
                       max_num_traj=1000000):
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

        dataset.shuffle()

        if len(dataset) > max_num_traj:
            return original_dataset, dataset.sample(max_num_traj)
        else:
            return original_dataset, dataset

    def create_dataset(self, log_dir, max_num_dir):
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
        # dirs = dirs[:2]

        # print("dir",str(dirs[0]), type(str(dirs[0])))

        min_val, max_val = 10000, 1000000
        trajectories = []
        num_dirs = len(dirs)
        print("Total Number of Directories", str(num_dirs))
        if max_num_dir is not None:
            max_num_dir = np.clip(max_num_dir, 0, len(dirs))
            dirs = dirs[:max_num_dir]

        for i, directory in enumerate(dirs):

            for index, time_step in enumerate(range(min_val, max_val + 1, 10000)):
                off_set = 9 - len(str(time_step))

                goal = "0" * off_set + str(time_step)

                trajectory = Trajectory(
                    elements=[directory.glob("*" + goal + '.index').__next__(),
                              directory.glob("*" + goal + '.meta').__next__(),
                              directory.glob("*" + goal + '.data-00000-of-00001').__next__()],
                    path=str(directory),
                    time_step=time_step
                )

                trajectories.append(trajectory)

            print("Progress: ", str(i), "/", str(num_dirs))

        dataset = Dataset(trajectories)

        return dataset

    def prepare_data(self, log_dir, max_num_traj):
        d = create_dataset(log_dir)
        print("Number of Trajectories after create: ", str(len(d)))

        original_d, d = adjust_dataset(d, max_num_traj=max_num_traj)
        print("Number of Trajectories after adjust: ", str(len(d)))

        return original_d, d

    def flow_alternative(self, batch_size, env="Mujoco-planar-snake-cars-angle-line-v1"):
        self.batch_size = batch_size
        max_num_traj = self.max_number_trajectories
        #########################################################################
        # TODO:                                                                 #
        #                                                                       #
        #########################################################################
        max_num_traj = 10
        save_path = "/home/andreas/Desktop/"
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        d = self.create_dataset(self.log_dir)
        print("Number of Trajectories after create: ", str(len(d)))

        original_d, d = self.adjust_dataset(d, max_num_traj=max_num_traj)
        print("Number of Trajectories after adjust: ", str(len(d)))

        # returns generator
        return self.initialize_dataset_alternative(env_id=env, original_dataset=d, horizon=500, seed=self.seed, batch_size = batch_size)

    def initialize_dataset(self, env_id, batch_size, original_dataset, horizon, num_samples_per_trajectories=10, seed=0, render=False):
        print("Initializing Dataset")

        from random import randint
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)
        env.unwrapped.metadata['target_v'] = 0.15  # not sure if needed
        max_timesteps = env._max_episode_steps

        #########################################################################
        # TODO:                                                                 #
        # Just use first trajectory of Dataset                                                                #
        #########################################################################
        # original_dataset = original_dataset[0]


        max_timesteps = 2000
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # create policy
        pi = self.policy_fn('pi0', env.observation_space, env.action_space)

        min_start, max_start = 0, env._max_episode_steps - 1 - horizon

        final_dataset = []
        batch = []

        size_dataset = len(original_dataset)
        for i, trajectory in enumerate(original_dataset):

            model_file = self.model_file_from_trajectory(trajectory)

            for j in range(num_samples_per_trajectories):
                start = randint(min_start, max_start)

                done, number_of_timesteps, info_collector, cum_reward, l_obs, l_acs, l_rews = \
                    self.run_environment_episode(env, pi, seed, model_file, start=start, horizon=horizon,
                                            max_timesteps=max_timesteps, render=render, stochastic=False)

                # cum_reward übergeben
                subtrajectory = SubTrajectory(rewards=l_rews, actions=l_acs, observations=l_obs)

                #print(subtrajectory)

                batch.append(subtrajectory)



                if len(batch) == batch_size:
                    #print(batch)
                    final_dataset.append(batch)
                    batch = []

                print("Subtrajectory: ", str(j + 1), "/", str(num_samples_per_trajectories), ", Trajectory", str(i + 1),
                      "/", str(size_dataset))

        #return Dataset(final_dataset)
        return final_dataset

    def flow(self, batch_size, max_num_traj, num_samples_per_trajectories,
             max_num_dir=None, env="Mujoco-planar-snake-cars-angle-line-v1"):
        self.batch_size = batch_size
        self.max_number_trajectories = max_num_traj
        #########################################################################
        # TODO:                                                                 #
        #                                                                       #
        #########################################################################
        #max_num_traj = 10
        save_path = "/home/andreas/Desktop/"
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        d = self.create_dataset(self.log_dir, max_num_dir)
        print("Number of Trajectories after create: ", str(len(d)))

        original_d, d = self.adjust_dataset(d, max_num_traj=max_num_traj)
        print("Number of Trajectories after adjust: ", str(len(d)))

        # returns generator
        trainable_dataset = self.initialize_dataset(env_id=env, original_dataset=d, horizon=500, seed=self.seed,
                                                    batch_size = batch_size, num_samples_per_trajectories=num_samples_per_trajectories)

        self.size_dataset = len(trainable_dataset)

        #print(trainable_dataset)

        return iter(trainable_dataset), trainable_dataset


'''        if batch_size <= len(trainable_dataset):
            yield Dataset(trainable_dataset[:batch_size])
        else:
            yield Dataset(trainable_dataset[:len(trainable_dataset)])'''


'''    def get_batch(self):
        for i in range(self.size_dataset % self.batch_size):
            print("i: %d"%i)
            if batch_size <= len(trainable_dataset):
                yield Dataset(trainable_dataset[:batch_size])
            else:
                yield Dataset(trainable_dataset[:len(trainable_dataset)])'''





