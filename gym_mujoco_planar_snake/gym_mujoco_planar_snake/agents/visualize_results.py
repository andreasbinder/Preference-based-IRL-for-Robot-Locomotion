#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from sklearn.model_selection import ParameterGrid

from gym.envs.registration import register



# TODO doesnt work
# from baselines.common.vec_env import VecVideoRecorder

import imageio

import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper

from gym_mujoco_planar_snake.common import my_tf_util

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import gym_mujoco_planar_snake.benchmark.plots as import_plots

from gym_mujoco_planar_snake.common.reward_wrapper_pytorch import MyRewardWrapper
from gym_mujoco_planar_snake.common.reward_nets import PairNet, TripletNet

import torch


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


def policy_fn(name, ob_space, ac_space):
    from baselines.ppo1 import mlp_policy
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

    # from baselines.ppo1 import cnn_policy
    # return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)


# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic=True):
    number_of_timestep = 0
    done = False

    # timesteps test
    # max_timesteps = 5000

    # load model
    my_tf_util.load_state(model_file)

    # save_dir = '/home/andreas/Desktop/20200625-2355-001000000'

    # my_tf_util.save_state(save_dir)

    # my_tf_util.load_state(save_dir)

    images = []
    #################
    # TODO imageio test
    # img = env.render(mode='rgb_array')

    #################

    # set seed

    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()


    print("done")

    cum_reward = 0

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        action = pi.act(stochastic, obs)

        action = action[0]  # TODO check



        obs, reward, done, info = env.step(action)

        cum_reward += reward



        # render
        if render:
            env.render()

        number_of_timestep += 1

    # imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)




def enjoy(env_id, seed, model_dir, reward_net_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()



        env = gym.make(env_id)



        max_episode_steps = env._max_episode_steps

        #net = TripletNet()
        net = PairNet()
        net.load_state_dict(torch.load(reward_net_dir))

        env = MyRewardWrapper(env, net)


        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)

        model_index = 0
        model_file = model_files[model_index]

        pi = policy_fn('pi', env.observation_space, env.action_space)



        env.unwrapped.metadata['target_v'] = 0.15


        run_environment_episode(env, pi, seed, model_file, max_episode_steps, render=True, stochastic=False)



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6

    # parser.add_argument('--train', help='do training or load model', type=bool, default=True)
    parser.add_argument('--train', help='do training or load model', type=bool, default=False)

    # velocity - power test
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')

    # save_frequency_steps
    parser.add_argument('--sfs', help='save_frequency_steps', default=10000)  # for mujoco

    parser.add_argument('--model_dir', help='model to render')

    parser.add_argument('--reward_net_dir', help='model to render')

    args = parser.parse_args()
    logger.configure()

    # CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    enjoy(args.env, seed=args.seed, model_dir=args.model_dir, reward_net_dir=args.reward_net_dir)


if __name__ == '__main__':
    main()

# Notes
# in enjoy aufräumen
# es gibt 100 models (auswählbar durch modelindex) da 1 Mio/ 10K(sfs)
# constant 0.5 seems to work better than 1
# which models are displayed