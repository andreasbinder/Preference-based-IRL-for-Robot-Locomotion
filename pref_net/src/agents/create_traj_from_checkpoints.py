# TODO
# fix order of get models: sees 50K as > then 450k


from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from tqdm import tqdm

#from sklearn.model_selection import ParameterGrid

from gym.envs.registration import register

# TODO doesnt work
# from baselines.common.vec_env import VecVideoRecorder

import imageio

import os
import os.path as osp

from pref_net.common.misc_util import Configs

from pref_net.common.env_wrapper import ModelSaverWrapper

from pref_net.common import my_tf_util

#from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

#import gym_mujoco_planar_snake.benchmark.plots as import_plots

import torch

def set_seeds(configs):
    seed = configs.get_seed()
    # tensorflow, numpy, random(python)

    set_global_seeds(seed)

    '''tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)'''

    torch.manual_seed(seed)

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
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic, current_timestep):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # TODO get timestep from model file

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    # TODO more seeds
    import gym.spaces.prng as prng
    prng.seed(seed)
    # env.action_space.np_random.seed(seed)
    # env.action_space.seed(seed)


    obs = env.reset()

    # TODO!!!
    # obs[-1] = target_v

    # info_collector = InfoCollector(env, {'target_v': target_v})
    #info_collector = InfoCollector(env, {'env': env, 'seed': seed})

    info_collector = None

    # injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]

    # env.unwrapped.metadata['injured_joint'] = injured_joint_pos[2]
    # print(env.unwrapped.metadata['injured_joint'])
    #print("done")

    cum_reward = []

    observations = []

    distance = []

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        # obs[-1] = target_v

        action, _ = pi.act(stochastic, obs)



        obs, reward, done, info = env.step(action)

        observations.append(obs)

        cum_reward.append(reward)

        distance.append(info["distance_head"])

        info['seed'] = seed
        info['env'] = env.spec.id


        # add info
        #info_collector.add_info(info)


        # render
        if render:
            env.render()

        number_of_timestep += 1

    # imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)

    return observations, cum_reward, distance


def enjoy(env_id, seed, model_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        env.seed(seed)

        # TODO set max episode length
        env._max_episode_steps = EPISODE_MAX_LENGTH


        # TODO
        # if I also wanted the newest model
        check_for_new_models = True
        gym.logger.setLevel(logging.WARN)

        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)
        # model_files.sort()

        #  start with first file
        model_index = 0
        model_file = model_files[model_index]

        print("Model Files")
        print(model_files[0])

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

        # TODO
        time_step = TIME_STEP # 1000000 995000
        step_size = STEP_SIZE # 10000
        render = False
        from time import ctime
        start_time = ctime()

        for model_file in tqdm(model_files):

            # TODO adjust velocity
            env.unwrapped.metadata['target_v'] = 0.1


            observations, cum_reward, distance = run_environment_episode(env, pi, seed, model_file, env._max_episode_steps,
                                                   render=render, stochastic=False, current_timestep=time_step)


            # TODO use time_step or cum_reward
            save_subtrajectories(observations, time_step, distance)

            time_step -= step_size
            #sum_reward.append(rewards)

            # TODO
            # loads newest model file, not sure that I want that functionality

            '''check_model_file = get_latest_model_file(model_dir)
            if check_model_file != model_file and check_for_new_models:
                model_file = check_model_file
                logger.log('loading new model_file %s' % model_file)'''

            # TODO
            # go through saved models
            '''if model_index > 0:
                model_index -= 1
                model_file = model_files[model_index]
                logger.log('loading new model_file %s' % model_file)

            # TODO
            # break if index at -1
            if model_index == -1:
                break'''

            #print('timesteps: %d, info: %s' % (number_of_timesteps, str(sum_info)))

        print(start_time)
        print(ctime())

def save_subtrajectories(observations, time_step, distance):

    #print("Saving Timestep %i"%(time_step))

    starts = np.random.randint(0, EPISODE_MAX_LENGTH - TRAJECTORY_LENGTH, size=100)
    starts.sort()

    # TODO time_step or reward, time_step  + start
    #trajectories = np.array([(np.array(observations)[start:start + TRAJECTORY_LENGTH], sum(cum_reward[start:start + TRAJECTORY_LENGTH])) for start in starts])
    trajectories = np.array(
        [(np.array(observations)[start:start + TRAJECTORY_LENGTH], time_step + start,  sum(distance[start:start + TRAJECTORY_LENGTH])) for
         start in starts])


    #path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
    name = "trajectories_{time_step}.npy".format(time_step=time_step)

    with open(os.path.join(SAVE_PATH, name), 'wb') as f:
        np.save(f, np.array(trajectories))





def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')

    parser.add_argument('--model_dir', help='model to render', type=str)

    args = parser.parse_args()
    logger.configure()

    #set_seeds(args.seed)

    agent_id = args.model_dir[-1]

    #enjoy(args.env, seed=args.seed, model_dir=args.model_dir)

    with tf.variable_scope(agent_id):
        enjoy(args.env, seed=args.seed, model_dir=args.model_dir)


if __name__ == '__main__':
    # TODO consider longer trajectory
    TRAJECTORY_LENGTH = 50 #50 100
    EPISODE_MAX_LENGTH = 1000
    TIME_STEP = 995000
    STEP_SIZE = 5000
    #SAVE_PATH = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
    SAVE_PATH = "/tmp/"
    main()
