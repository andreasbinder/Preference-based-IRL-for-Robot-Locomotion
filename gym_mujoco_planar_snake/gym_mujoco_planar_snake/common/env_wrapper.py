from gym.core import ObservationWrapper
# from baselines.common import tf_util as U
from gym_mujoco_planar_snake.common import my_tf_util as U
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
from gym_mujoco_planar_snake.common.ensemble import Ensemble


# np.random.seed(0)

def extract_configs(configs, learned_reward):
    if learned_reward:
        return configs.get_num_improved_timesteps()
    else:
        return configs.get_num_initial_timesteps()


def prepare_env(env, save_check_points, save_dir, sfs, id, learned_reward, configs):
    num_timesteps = extract_configs(configs=configs, learned_reward=learned_reward)

    log_dir = osp.join(save_dir, "agent_" + str(id))
    default_reward_dir = osp.join(save_dir, "default_reward")

    os.makedirs(log_dir, exist_ok=True)
    env = MyMonitor(env, log_dir)

    if save_check_points:
        env = ModelSaverWrapper(env, log_dir, sfs, id)

    # save_dir = osp.join(log_dir, ctime()[4:19])
    # save_dir = log_dir
    if learned_reward:
        # venv, nets, max_timesteps, log_dir, name_dir, ctrl_coeff
        nets = Ensemble.load(save_dir, configs.get_num_nets())

        ctrl_coeff = configs.get_ctrl_coeff()
        env = MyRewardWrapper(env, nets, num_timesteps, log_dir, ctrl_coeff, default_reward_dir, id)
    else:
        trajectory_length = configs.get_trajectory_length()
        num_traj_per_epoch = configs.get_num_traj_per_episode()
        env = GenTrajWrapper(env,
                             log_dir,
                             id,
                             num_timesteps,
                             trajectory_length,
                             num_traj_per_epoch)


    return env

class MyMonitor(Monitor):

    def __init__(self, env, log_dir):
        Monitor.__init__(self, env, log_dir)

    @staticmethod
    def load_files_from_ensemble_dir(ensemble_dir, name_run_dir="agent"):

        # get run dirs
        run_dirs = [dir for dir in os.listdir(ensemble_dir) if name_run_dir in dir]





        # get monitor files
        #monitorfiles = [get_monitor_files(os.path.join(ensemble_dir, dir)) for dir in run_dirs]
        monitorfiles = [os.path.join(ensemble_dir, dir) for dir in run_dirs]


        # load results
        dfs = [load_results(monitorfile) for monitorfile in monitorfiles]

        return dfs



    @staticmethod
    def compare_initial_and_improved_reward(initial_rew_dir, improved_rew_dir):
        initial_dfs, improved_dfs = MyMonitor.load_files_from_ensemble_dir(initial_rew_dir), MyMonitor.load_files_from_ensemble_dir(improved_rew_dir)

        get_rewards_as_numpy = lambda x: x["r"].to_numpy()

        initial_arrays = np.array([get_rewards_as_numpy(df) for df in initial_dfs])
        improved_arrays = np.array([get_rewards_as_numpy(df) for df in improved_dfs])

        print(initial_arrays.shape)
        print(improved_arrays.shape)

        max_initial = np.max(initial_arrays, axis=1)
        max_improved = np.max(improved_arrays, axis=1)

        print("Compare Initial Max to Improved Max")
        print(max_initial)
        print(max_improved)

        initial_averaged = np.average(initial_arrays, axis=0)
        improved_averaged = np.average(improved_arrays, axis=0)

        '''print(initial_averaged.shape)
        print(improved_averaged.shape)'''

        print("Compare Initial Mean to Improved Mean")
        print(initial_averaged.mean())
        print(improved_averaged.mean())

        import matplotlib.pyplot as plt

        indices1 = 1001

        plt.plot(indices1, initial_arrays[:, 1])

        '''indices2 = 801

        plt.plot(indices2, improved_arrays[1, 1])'''

        plt.show()













class ModelSaverWrapper(ObservationWrapper):

    def __init__(self, env, model_dir, save_frequency_steps, id):
        ObservationWrapper.__init__(self, env=env)

        '''self.id = str(id)
        self.sess = sess'''

        self.id = id
        self.save_frequency_steps = save_frequency_steps
        self.total_steps = 0
        self.total_steps_save_counter = 0
        self.total_episodes = 0
        self.str_time_start = time.strftime("%Y%m%d-%H%M")

        self.model_dir = model_dir

    def reset(self, **kwargs):
        self.total_episodes += 1


        # todo start saving after 100k timesteps
        if self.total_steps_save_counter == self.save_frequency_steps or self.total_steps == 1:
            


            buffer = 9

            len_total_steps = len(str(self.total_steps))

            zeros = (buffer - len_total_steps) * "0"

            file_name = osp.join(self.model_dir, zeros+str(self.total_steps))

            U.save_state(file_name)

            logger.log('Saved model to: ' + file_name)

            self.total_steps_save_counter = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps += 1
        self.total_steps_save_counter += 1

        return self.env.step(action)

    def observation(self, observation):
        return observation


class GenTrajWrapper(ObservationWrapper):

    def __init__(self, env, path, id, max_timesteps, trajectory_length, num_traj_per_epoch):
        ObservationWrapper.__init__(self, env=env)

        self.max_timesteps = max_timesteps
        # self.sfs = sfs
        self.last_e_steps = 0
        self.episodes = 1
        self.observations_list = []
        self.rewards_list = []
        self.trajectories = []
        self.trajectory_length = trajectory_length

        self.path = path
        self.num_traj_per_epoch = num_traj_per_epoch
        self.name = "trajectories_" + str(id) + ctime()[4:19].replace(" ", "_") + ".npy"

        self.saved = False

    def store(self, observation, reward):
        self.observations_list.append(observation)
        self.rewards_list.append(reward)

        return observation, reward

    def step(self, action):
        self.last_e_steps += 1
        observation, reward, done, info = self.env.step(action)
        # print(done)
        self.store(observation, reward)
        return observation, reward, done, info

    def reset(self, **kwargs):

        # TODO ignore first 100k timesteps, 100000
        if self.last_e_steps > 100000:

            # assert False, self.last_e_steps

            # TODO ignore first 200 steps
            if self.trajectory_length == 50:
                starts = np.random.randint(200, 950, size=self.num_traj_per_epoch)
                starts.sort()
            else:
                starts = [0]

            # print(starts)create trajectories
            for start in starts:
                # tuple of form ([50,27], [50])
                # TODO changing true reward for timestep
                '''trajectory = np.array(self.observations_list)[start:start+self.trajectory_length], \
                             np.sum(self.rewards_list[start:start+self.trajectory_length])'''

                trajectory = np.array(self.observations_list)[start:start + self.trajectory_length], \
                             self.last_e_steps - 1000 + start

                self.trajectories.append(trajectory)

            self.observations_list = []
            self.rewards_list = []

            # print(self.last_e_steps)

            if self.last_e_steps >= self.max_timesteps and not self.saved:
                # print("in save")mkdir
                path = self.path

                with open(os.path.join(path, self.name), 'wb') as f:
                    np.save(f, np.array(self.trajectories))
                    # self.trajectories = []

                # TODO give as parameter
                default_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset/"
                default = os.path.join(default_path, self.name)
                # TODO remove completely
                '''with open(default, 'wb') as f:
                    np.save(f, np.array(self.trajectories))'''

                self.trajectories = []
                self.saved = True

        return self.env.reset(**kwargs)


class MyRewardWrapper(RewardWrapper):

    def __init__(self, venv, nets, max_timesteps, save_dir, ctrl_coeff, default_reward_dir, id):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.counter = 0
        self.ctrl_coeff = ctrl_coeff
        self.nets = nets
        self.id = id

        self.save_dir = save_dir
        self.default_reward_dir = default_reward_dir

        self.cliprew = 10.
        self.epsilon = 1e-8

        self.rew_rms = [RunningMeanStd(shape=()) for _ in range(len(nets))]

        # TODO compare reward development
        self.max_timesteps = max_timesteps
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

        # self.reward_list.append(rews.item())

        # TODO render for debugging
        # do rendering by saving observations or actions
        '''if self.counter >= 200000:
            self.venv.render()'''



        # TODO return reward or abs_reward
        return obs, pred, news, infos

    def reset(self, **kwargs):




        if self.counter == self.max_timesteps:
            with open(os.path.join(self.save_dir, "results.npy"), 'wb') as f:
                np.save(f, np.array(self.rewards))

            with open(os.path.join(self.default_reward_dir,
                                   "results" + str(self.id) + ctime()[4:19].replace(" ", "_") + ".npy"), 'wb') as f:
                np.save(f, np.array(self.rewards))

            self.rewards = []

        return self.venv.reset(**kwargs)

    def store_rewards(self, reward, pred_reward):
        self.rewards.append((reward, pred_reward))
