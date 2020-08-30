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
#from baselines.common.vec_env import VecVideoRecorder

import imageio

import os
import os.path as osp

from gym_mujoco_planar_snake.common.env_wrapper import ModelSaverWrapper

from gym_mujoco_planar_snake.common import my_tf_util

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import gym_mujoco_planar_snake.benchmark.plots as import_plots




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

    # timesteps test
    #max_timesteps = 5000

    # load model
    my_tf_util.load_state(model_file)

    #save_dir = '/home/andreas/Desktop/20200625-2355-001000000'

    #my_tf_util.save_state(save_dir)

    #my_tf_util.load_state(save_dir)

    images = []
    #################
    # TODO imageio test
    # img = env.render(mode='rgb_array')

    #################


    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()

    # TODO!!!
    # obs[-1] = target_v

    # info_collector = InfoCollector(env, {'target_v': target_v})
    info_collector = InfoCollector(env, {'env': env, 'seed': seed})

    injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]


    #env.unwrapped.metadata['injured_joint'] = injured_joint_pos[2]
    #print(env.unwrapped.metadata['injured_joint'])
    print("done")

    cum_reward = 0

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:
    #while number_of_timestep < max_timesteps:

        #images.append(img)

        #print("Timesteps: ", str(number_of_timestep))
        # TODO!!!
        # obs[-1] = target_v

        action = pi.act(stochastic, obs)

        action = action[0]  # TODO check

        #print(action)

        """
        if number_of_timestep % int(max_timesteps / len(injured_joint_pos)) == 0:
            index = int(number_of_timestep / int(max_timesteps / (len(injured_joint_pos))))
            index = min(index, len(injured_joint_pos)-1)

            print("number_of_timestep", number_of_timestep, index)
            env.unwrapped.metadata['injured_joint'] = injured_joint_pos[index]
        """

        obs, reward, done, info = env.step(action)

        cum_reward += reward

        info['seed'] = seed
        info['env'] = env.spec.id

        #print(reward)

        # add info
        info_collector.add_info(info)

        ### TODO imagio
        #img = env.render(mode='rgb_array')
        ###



        # render
        if render:
            env.render()

        number_of_timestep += 1

    #imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)

    return done, number_of_timestep, info_collector, cum_reward


def enjoy(env_id, seed, model_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)


        # TODO
        # if I also wanted the newest model
        check_for_new_models = False

        ###################################################
        # TODO for video recording
        ###################################################



        ###################################################



        max_timesteps = 3000000

        '''modelverion_in_k_ts = 2510  # better

        model_index = int(max_timesteps / 1000 / 10 - modelverion_in_k_ts / 10)'''


        gym.logger.setLevel(logging.WARN)
        #gym.logger.setLevel(logging.DEBUG)

        #model_dir = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_ensemble_Aug_30_14:58:04/agent_0"

        # TODO best model
        #"/home/andreas/LRZ_Sync+Share/BachelorThesis//gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/single_triplet_unnormalized/models/Mujoco-planar-snake-cars-angle-line-v1/ppo"



        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)

        model_index = 0
        model_file = model_files[model_index]
        print('available models: ', len(model_files))
        logger.log("load model_file: %s" % model_file)

        sum_info = None
        pi = policy_fn('pi', env.observation_space, env.action_space)

        sum_reward = []

        while True:
            # run one episode

            # TODO specify target velocity
            # only takes effect in angle envs
            # env.unwrapped.metadata['target_v'] = 0.05
            env.unwrapped.metadata['target_v'] = 0.15
            # env.unwrapped.metadata['target_v'] = 0.25

            # env._max_episode_steps = env._max_episode_steps * 3



            #########################################################################
            # TODO:                                                                 #
            #                                                                       #
            #########################################################################
            #env._max_episode_steps = 50
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            done, number_of_timesteps, info_collector, rewards = run_environment_episode(env, pi, seed, model_file,
                                                                                env._max_episode_steps, render=True,
                                                                                stochastic=False)


            '''if True:
                break'''

            #info_collector.episode_info_print()

            sum_reward.append(rewards)

            # TODO
            # loads newest model file, not sure that I want that functionality

            check_model_file = get_latest_model_file(model_dir)
            if check_model_file != model_file and check_for_new_models:
                model_file = check_model_file
                logger.log('loading new model_file %s' % model_file)

            # TODO
            # go through saved models
            if model_index >= 0:
                model_index -= 1
                model_file = model_files[model_index]
                logger.log('loading new model_file %s' % model_file)

            # TODO
            # break if index at -1
            if model_index == -1:
                break


            print('timesteps: %d, info: %s' % (number_of_timesteps, str(sum_info)))



            



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
