# TODO
# fix order of get models: sees 50K as > then 450k


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

from gym_mujoco_planar_snake.common.env_wrapper import ModelSaverWrapper

from gym_mujoco_planar_snake.common import my_tf_util

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import gym_mujoco_planar_snake.benchmark.plots as import_plots

import matplotlib.pyplot as plt


def get_latest_model_file(model_dir):
    return get_model_files(model_dir)[0]


def get_model_files(model_dir):
    list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
    list.sort(key=str.lower, reverse=True)

    files = [osp.join(model_dir, ele) for ele in list]
    return files


def get_model_dir(env_id, name):
    # '../../models'
    model_dir = osp.join(logger.get_dir(), 'models')
    os.mkdir(model_dir)
    model_dir = ModelSaverWrapper.gen_model_dir_path(model_dir, env_id, name)
    logger.log("model_dir: %s" % model_dir)
    return model_dir


# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic=False):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()

    # TODO!!!
    # obs[-1] = target_v

    # info_collector = InfoCollector(env, {'target_v': target_v})
    info_collector = InfoCollector(env, {'env': env, 'seed': seed})




    cum_reward = 0

    cum_rew_p = []
    cum_rew_v = []
    observations = []
    cum_velocity = []
    cum_rew = []
    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        # obs[-1] = target_v

        action, predv = pi.act(stochastic, obs)

        obs, reward, done, info = env.step(action)

        cum_rew_p.append(info["rew_p"])
        cum_rew_v.append(info["rew_v"])

        cum_rew.append(reward)

        cum_velocity.append(info["velocity"])

        observations.append(obs)


        reward = info["rew_p"] * info["velocity"]

        cum_reward += reward


        # render
        if render:
            env.render()

        number_of_timestep += 1



    return done, number_of_timestep, info_collector, cum_reward


def enjoy(env_id, seed, model_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        gym.logger.setLevel(logging.WARN)
        # gym.logger.setLevel(logging.DEBUG)

        model_files = get_model_files(model_dir)

        model_index = len(model_files) - 1
        model_file = model_files[model_index]


        print('available models: ', len(model_files))
        logger.log("load model_file: %s" % model_file)

        policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                          ob_space=ob_space,
                                                                          ac_space=ac_space,
                                                                          hid_size=64,
                                                                          num_hid_layers=2
                                                                          )

        sum_info = None
        reverse = True
        render = False

        if reverse:
            model_files.reverse()

        pi = policy_fn('pi', env.observation_space, env.action_space)

        episode_rewards = []

        print(model_files)



        for model_file in model_files:


            done, number_of_timesteps, info_collector, rewards = run_environment_episode(env, pi, seed, model_file,
                                                                                         env._max_episode_steps,
                                                                                         render=render,
                                                                                         stochastic=False)

            episode_rewards.append(rewards)

        # plot results
        indices = range(len(episode_rewards))

        plt.plot(indices, episode_rewards)

        plt.show()





def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')

    parser.add_argument('--model_dir', help='model to render', type=str)

    args = parser.parse_args()
    logger.configure()

    agent_id = args.model_dir[-1]

    with tf.variable_scope(agent_id):
        enjoy(args.env, seed=args.seed, model_dir=args.model_dir)




if __name__ == '__main__':
    main()
