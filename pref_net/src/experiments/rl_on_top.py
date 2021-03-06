from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

# from sklearn.model_selection import ParameterGrid

from gym.envs.registration import register

# TODO doesnt work
# from baselines.common.vec_env import VecVideoRecorder

import imageio

import os
import os.path as osp

from pref_net.src.utils.model_saver import ModelSaverWrapper

from pref_net.src.utils import my_tf_util

#from src.utils.info_collector import InfoCollector, InfoDictCollector

import argparse

from pref_net.src.utils.configs import Configs

#import matplotlib.pyplot as plt


def get_latest_model_file(model_dir):
    return get_model_files(model_dir)[0]


def get_model_files(model_dir):
    list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
    list.sort(key=str.lower, reverse=True)

    files = [osp.join(model_dir, ele) for ele in list]
    return files





# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, stochastic=False):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()

    cum_distance = 0

    cum_power_consumption = 0

    cum_reward = 0

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        action, predv = pi.act(stochastic, obs)

        obs, reward, done, info = env.step(action)

        cum_reward += reward

        cum_distance += info["distance_delta"]

        cum_power_consumption += info["rew_p"]

        number_of_timestep += 1

    print(cum_reward)

    return  cum_distance, cum_power_consumption




if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)
    configs = configs.data["rl_on_top"]

    logger.configure()

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    paths_with_var_scops = configs["paths_with_var_scops"]

    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'


    env = gym.make(ENV_ID)


    episode_rewards = []

    all_episodes_rew_pc = []

    for item in paths_with_var_scops:


        print(list(item.values()))

        var_scope, path = list(item.values())[0]

        with tf.variable_scope(str(var_scope)):

            sess = U.make_session(num_cpu=1)
            sess.__enter__()

            policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                              ob_space=ob_space,
                                                                              ac_space=ac_space,
                                                                              hid_size=64,
                                                                              num_hid_layers=2
                                                                              )

            pi = policy_fn('pi', env.observation_space, env.action_space)



            gym.logger.setLevel(logging.WARN)

            model_file = get_latest_model_file(path)

            distance_rew = 0

            rew_p = 0


            for s in range(configs["runs_per_model"]):
                single_episode_distance, single_episode_rew_p = run_environment_episode(env, pi,s ,#configs["seed"]
                                                             model_file,
                                                             env._max_episode_steps,
                                                             stochastic=False)


                distance_rew += single_episode_distance
                rew_p += single_episode_rew_p

            episode_rewards.append(distance_rew)
            all_episodes_rew_pc.append(rew_p)

    mean = lambda x: sum(x) / len(x)

    print("Distance")
    print("All: %s Max: %f Mean: %f"%(str(episode_rewards), max(episode_rewards), mean(episode_rewards)) )

    print("Power Consumption")
    print("All: %s Max: %f Mean: %f" % (str(all_episodes_rew_pc), max(all_episodes_rew_pc), mean(all_episodes_rew_pc)))
    '''
    print("Rewards")
    print(episode_rewards)
    print("Max")
    print(max(episode_rewards))
    print("Mean")
    print(mean(episode_rewards))'''