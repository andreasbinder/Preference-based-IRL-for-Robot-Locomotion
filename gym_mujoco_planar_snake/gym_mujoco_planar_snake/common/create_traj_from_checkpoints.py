# TODO
# fix order of get models: sees 50K as > then 450k


from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
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
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic, current_timestep):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # TODO get timestep from model file

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()

    # TODO!!!
    # obs[-1] = target_v

    # info_collector = InfoCollector(env, {'target_v': target_v})
    info_collector = InfoCollector(env, {'env': env, 'seed': seed})

    # injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]

    # env.unwrapped.metadata['injured_joint'] = injured_joint_pos[2]
    # print(env.unwrapped.metadata['injured_joint'])
    print("done")

    cum_reward = 0

    observations = []

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        # obs[-1] = target_v

        action, _ = pi.act(stochastic, obs)



        obs, reward, done, info = env.step(action)

        observations.append(obs)

        cum_reward += reward

        info['seed'] = seed
        info['env'] = env.spec.id

        # print(reward)

        # add info
        info_collector.add_info(info)

        ### TODO imagio
        # img = env.render(mode='rgb_array')
        ###

        # render
        if render:
            env.render()

        number_of_timestep += 1

    # imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)

    return observations


def enjoy(env_id, seed, model_dir):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)


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
        print(model_files)

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

        time_step = 995000 # 1000000
        step_size = 5000 # 10000
        render = False
        from time import ctime
        start_time = ctime()

        for model_file in model_files:


            env.unwrapped.metadata['target_v'] = 0.15


            observations = run_environment_episode(env, pi, seed, model_file, env._max_episode_steps,
                                                   render=render, stochastic=False, current_timestep=time_step)


            save_subtrajectories(observations, time_step)

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

def save_subtrajectories(observations, time_step):

    print("Saving Timestep %i"%(time_step))

    starts = np.random.randint(0, 950, size=100)
    starts.sort()

    trajectories = np.array([(np.array(observations)[start:start + 50], time_step  + start) for start in starts])

    path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
    name = "trajectories_{time_step}.npy".format(time_step=time_step)

    with open(os.path.join(path, name), 'wb') as f:
        np.save(f, np.array(trajectories))





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
