# TODO
# fix order of get models: sees 50K as > then 450k

from src.common.misc_util import Configs

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

from src.common.env_wrapper import ModelSaverWrapper

from src.common import my_tf_util


# from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

# import gym_mujoco_planar_snake.benchmark.plots as import_plots


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
    # info_collector = InfoCollector(env, {'env': env, 'seed': seed})
    info_collector = None

    # injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]

    # env.unwrapped.metadata['injured_joint'] = injured_joint_pos[2]
    # print(env.unwrapped.metadata['injured_joint'])
    print("done")

    cum_reward = 0

    cum_rew_p = []
    cum_rew_v = []
    observations = []
    cum_velocity = []
    cum_rew = []
    distance_head = []
    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        # obs[-1] = target_v

        action, _ = pi.act(stochastic, obs)

        # action = action[0]  # TODO check

        # print(action)

        """
        if number_of_timestep % int(max_timesteps / len(injured_joint_pos)) == 0:
            index = int(number_of_timestep / int(max_timesteps / (len(injured_joint_pos))))
            index = min(index, len(injured_joint_pos)-1)

            print("number_of_timestep", number_of_timestep, index)
            env.unwrapped.metadata['injured_joint'] = injured_joint_pos[index]
        """

        obs, reward, done, info = env.step(action)

        # TODO add if needed
        '''cum_rew_p.append(info["rew_p"])
        cum_rew_v.append(info["rew_v"])'''

        distance_head.append(info["distance_delta"])

        cum_rew.append(reward)

        cum_velocity.append(info["velocity"])

        observations.append(obs)

        cum_reward += reward

        info['seed'] = seed
        info['env'] = env.spec.id

        # print(reward)

        # add info
        # info_collector.add_info(info)

        ### TODO imagio
        # img = env.render(mode='rgb_array')
        ###

        # render = False

        # render
        if render:
            env.render()

        number_of_timestep += 1

    print("Distance")
    print(sum(distance_head))

    # imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)
    print(sum(cum_velocity) / len(cum_velocity))
    print(sum(cum_rew_p))
    print(sum(cum_rew_v))

    '''import matplotlib.pyplot as plt

    indices = [i for i in range(len(cum_velocity))]

    #plt.plot(indices, cum_velocity)

    fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2,2)
    fig.suptitle("Metrics improved")
    ax1.plot(indices, cum_velocity)
    ax1.set_title("Velocity")
    ax4.plot(indices, cum_rew)
    ax4.set_title("Total Reward")
    ax2.plot(indices, cum_rew_p)
    ax2.set_title("Reward Power")
    ax3.plot(indices, cum_rew_v)
    ax3.set_title("Reward Velocity")


    plt.show()'''

    return done, number_of_timestep, info_collector, cum_reward


def enjoy(env_id, seed, model_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        # TODO
        # if I also wanted the newest model
        check_for_new_models = False

        max_timesteps = 3000000

        '''modelverion_in_k_ts = 2510  # better

        model_index = int(max_timesteps / 1000 / 10 - modelverion_in_k_ts / 10)'''

        gym.logger.setLevel(logging.WARN)
        # gym.logger.setLevel(logging.DEBUG)

        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)
        # model_files.sort()

        model_index = 0
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
        pi = policy_fn('pi', env.observation_space, env.action_space)

        sum_reward = []

        while True:
            # run one episode

            # TODO specify target velocity
            # only takes effect in angle envs
            # TODO
            env.unwrapped.metadata['target_v'] = 0.1
            # env.unwrapped.metadata['target_v'] = 0.15
            # env.unwrapped.metadata['target_v'] = 0.25

            # env._max_episode_steps = env._max_episode_steps * 3

            done, number_of_timesteps, info_collector, rewards = run_environment_episode(env, pi, seed, model_file,
                                                                                         env._max_episode_steps,
                                                                                         render=RENDER,
                                                                                         stochastic=False)

            print(rewards)

            # info_collector.episode_info_print()

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
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)
    configs = configs.data["visualize"]

    logger.configure()



    RENDER = True
    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'

    with tf.variable_scope(str(configs["variable_scope"])):
        enjoy(ENV_ID, configs["seed"], configs["model_dir"])




