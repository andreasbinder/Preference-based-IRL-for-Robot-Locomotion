# TODO
# fix order of get models: sees 50K as > then 450k

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy

import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from tqdm import tqdm

import os
import os.path as osp

#from pref_net.sr.common.misc_util import Configs

#from pref_net.common import my_tf_util

from pref_net.src.utils import my_tf_util
import pref_net.src.utils.data_util as data_util
from pref_net.src.utils.configs import Configs


import torch



# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    import gym.spaces.prng as prng
    prng.seed(seed)


    obs = env.reset()

    cum_reward = []
    observations = []
    distance = []
    cum_rew_p = []

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        action, _ = pi.act(stochastic, obs)

        obs, reward, done, info = env.step(action)

        observations.append(obs)

        cum_reward.append(reward)

        distance.append(info["distance_delta"])

        cum_rew_p.append(info["rew_p"])

        # render
        if render:
            env.render()

        number_of_timestep += 1



    return observations, cum_reward, distance, cum_rew_p


def load_episodes(env_id, seed, model_files):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)
        env.seed(seed)
        # TODO set max episode length
        env._max_episode_steps = EPISODE_MAX_LENGTH

        gym.logger.setLevel(logging.WARN)

        policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                          ob_space=ob_space,
                                                                          ac_space=ac_space,
                                                                          hid_size=64,
                                                                          num_hid_layers=2
                                                                          )

        pi = policy_fn('pi', env.observation_space, env.action_space)


        render = RENDER
        from time import ctime
        start_time = ctime()


        for model_file in tqdm(model_files):

            # TODO adjust velocity
            env.unwrapped.metadata['target_v'] = 0.1

            time_step = int(model_file[-9:])


            observations, cum_reward, distance, cum_rew_p = run_environment_episode(env, pi, seed, model_file, env._max_episode_steps,
                                                   render=render, stochastic=False)



            save_full_episodes(observations, time_step, distance, cum_reward, cum_rew_p)



        print(start_time)
        print(ctime())

def save_subtrajectories(observations, time_step, distance):



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

def save_full_episodes(observations, time_step, distance, cum_reward, cum_rew_p):

    #print("Saving Timestep %i"%(time_step))



    # TODO time_step or reward, time_step  + start
    #trajectories = np.array([(np.array(observations)[start:start + TRAJECTORY_LENGTH], sum(cum_reward[start:start + TRAJECTORY_LENGTH])) for start in starts])
    trajectories = np.array([(observations, time_step, distance, cum_reward, cum_rew_p)])


    #path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
    FULL_EPISODES.append(np.array(trajectories))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')

    parser.add_argument('--model_dir', help='model to render', type=str, default='')

    parser.add_argument('--path_to_configs', type=str, default='/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml')

    parser.add_argument('--variable_scope',  default="0")

    args = parser.parse_args()
    logger.configure()

    configs_file = Configs(args.path_to_configs)
    configs = configs_file["create_dataset"]

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'

    # TODO configs

    # TODO consider longer trajectory
    #args.model_dir = "/gym_mujoco_planar_snake/prefnet/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_original_1.5Mio/agent_0"

    args.model_dir = "/home/andreas/Desktop/original_agent_1.5/agent_0"

    TRAJECTORY_LENGTH = configs["subtrajectry_length"] #50 100
    EPISODE_MAX_LENGTH = configs["episode_length"]
    RENDER = False
    #SAVE_PATH = "/gym_mujoco_planar_snake/prefnet/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_original_1.5Mio/data_500k"
    #SAVE_PATH = "/home/andreas/Desktop/original_agent_1.5/data_with_rew"
    SAVE_PATH = configs["data_dir"]

    FULL_EPISODES = []

    agent_id = args.model_dir[-1]


    list = [x[:-len(".index")] for x in os.listdir(args.model_dir) if x.endswith(".index")]
    list.sort(key=str.lower)
    model_files = [osp.join(args.model_dir, ele) for ele in list]

    num_models = len(model_files)

    with tf.variable_scope(agent_id):
        load_episodes(args.env, seed=args.seed, model_files=model_files)


    # Split in train and extrapolation data
    percentage = 1/3
    #FACTOR = int(num_models * percentage)
    FACTOR = int(configs["num_train"] / configs["save_frequency"])
    FULL_EPISODES = np.concatenate(FULL_EPISODES)

    TRAIN = FULL_EPISODES[:FACTOR]
    EXTRAPOLATE = FULL_EPISODES[FACTOR:]

    TRAIN_NAME = "train.npy"
    EXTRAPOLATE_NAME = "extrapolate.npy"

    # TODO uncomment
    with open(os.path.join(SAVE_PATH, TRAIN_NAME), 'wb') as f:
        np.save(f, np.array(TRAIN))

    with open(os.path.join(SAVE_PATH, EXTRAPOLATE_NAME), 'wb') as f:
        np.save(f, np.array(EXTRAPOLATE))



    '''from pref_net.benchmark.plot_results import return_all_episode_statistics

    return_all_episode_statistics(TRAIN)'''
    # all_episodes, trajectory_length, n
    train_set = data_util.simple_generate_dataset_from_full_episodes(all_episodes=TRAIN,
                                                                     trajectory_length=configs["subtrajectry_length"],
                                                                     n=configs["subtrajectories_per_episode"])

    name_for_default = "subtrajectories.npy"

    with open(os.path.join(SAVE_PATH, name_for_default), 'wb') as f:
        np.save(f, np.array(train_set))
