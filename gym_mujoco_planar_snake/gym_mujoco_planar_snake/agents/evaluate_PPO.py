#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from sklearn.model_selection import ParameterGrid

from gym.envs.registration import register
'''
register(
    id='Mujoco-planar-snake-cars-angle-line-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleLineEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)'''


import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper

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

    injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]


    env.unwrapped.metadata['injured_joint'] = injured_joint_pos[2]
    print(env.unwrapped.metadata['injured_joint'])
    print("done")

    while (not done) and number_of_timestep < max_timesteps:

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

        info['seed'] = seed
        info['env'] = env.spec.id

        #print(reward)

        # add info
        info_collector.add_info(info)

        # render
        if render:
            env.render()

        number_of_timestep += 1

    return done, number_of_timestep, info_collector


def enjoy(env_id, seed):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        '''
        print(env.action_space.sample())

        import numpy as np
        env.action_space.high[0] = 0.5
        env.action_space.low[0] = -0.5

        print(env.action_space.sample())

        if True:
            return
        '''



        # env = gym.make('Mujoco-planar-snake-cars-cam-v1')
        # env = gym.make('Mujoco-planar-snake-cars-cam-dist-zigzag-v1')
        # env = gym.make('Mujoco-planar-snake-cars-cam-dist-random-v1')
        # env = gym.make('Mujoco-planar-snake-cars-cam-dist-line-v1')
        # env = gym.make('Mujoco-planar-snake-cars-cam-dist-circle-v1')
        # env = gym.make('Mujoco-planar-snake-cars-cam-dist-wave-v1')

        # TODO
        # if I also wanted the newest model
        check_for_new_models = False

        # more steps
        # env._max_episode_steps = env.spec.max_episode_steps * 3
        # obs = env.reset()

        max_timesteps = 3000000

        # model_index = 254 # 251 # best

        # Select model file .....

        # check_for_new_models = False
        #
        # modelverion_in_k_ts = 2000
        modelverion_in_k_ts = 3000  # good
        modelverion_in_k_ts = 2510  # better

        model_index = int(max_timesteps / 1000 / 10 - modelverion_in_k_ts / 10)

        # TOdo last saved model
        model_index = 0

        print("actionspace", env.action_space)
        print("observationspace", env.observation_space)

        gym.logger.setLevel(logging.WARN)
        #gym.logger.setLevel(logging.DEBUG)

        run = 0
        model_dir = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/clip_test/InjuryIndex_2/Thu Jun 25 19:15:04 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/snake_test/run0_0/Thu Jun 25 13:03:08 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
            #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/snake_test/run0_-1/Thu Jun 11 14:28:53 2020_0.1/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/snake_test/run0_-1/Wed Jun 10 14:49:21 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/snake_test/run3_7_constant0,5/Wed Jun 10 12:23:51 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/run3_7/Wed Jun 10 10:50:54 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/run3_7_constant0,5/Wed Jun 10 10:50:54 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/run0_45/Mon Jun  8 19:04:37 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/run0_-1/Tue Jun  9 11:12:15 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #        '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/run0_1/Mon Jun  8 13:28:55 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
            #'/tmp/openai-2020-06-07-17-04-08-287219/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'

        # init load
        # model_dir = get_model_dir(env_id, 'ppo')
        # model_dir = '/home/andreas/LRZ Sync+Share/Training_Performances/snake_disabled/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'

        # fix range in mujoco xml file
        # model_dir = '/home/andreas/LRZ Sync+Share/Training_Performances/openai-2020-06-03-20-12-06-951977/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        # '/tmp/openai-2020-06-03-20-12-06-951977/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'

        # 8 joint disabled
        # model_dir = '/tmp/openai-2020-06-03-18-36-43-071582/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'

        # model_dir = '/home/andreas/LRZ Sync+Share/Training_Performances/openai-2020-06-01-21-57-16-490173/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        # woher kommt model_files
        # 1Mio steps: /tmp/openai-2020-05-30-14-23-41-125024/models/Mujoco-planar-snake-cars-angle-line-v1/ppo
        # 1k steps: /tmp/openai-2020-05-30-16-57-25-164249/models/Mujoco-planar-snake-cars-angle-line-v1/ppo
        # 100k steps: /tmp/openai-2020-05-30-17-06-19-529120/models/Mujoco-planar-snake-cars-angle-line-v1/ppo
        # disabled /home/andreas/LRZ Sync+Share/Training_Performances/snake_disabled/models/Mujoco-planar-snake-cars-angle-line-v1/ppo
        # model_files = get_model_files('')

        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)

        model_index = 80
        model_file = model_files[model_index]
        print('available models: ', len(model_files))
        #model_file = model_files[model_index]
        # model_file = model_files[75]
        logger.log("load model_file: %s" % model_file)

        sum_info = None
        pi = policy_fn('pi'+str(run), env.observation_space, env.action_space)

        while True:
            # run one episode

            # TODO specify target velocity
            # only takes effect in angle envs
            # env.unwrapped.metadata['target_v'] = 0.05
            env.unwrapped.metadata['target_v'] = 0.15
            # env.unwrapped.metadata['target_v'] = 0.25

            # env._max_episode_steps = env._max_episode_steps * 3

            done, number_of_timesteps, info_collector = run_environment_episode(env, pi, seed, model_file,
                                                                                env._max_episode_steps, render=True,
                                                                                stochastic=False)


            info_collector.episode_info_print()



            # TODO
            # loads newest model file, not sure that I want that functionality

            check_model_file = get_latest_model_file(model_dir)
            if check_model_file != model_file and check_for_new_models:
                model_file = check_model_file
                logger.log('loading new model_file %s' % model_file)

            # TODO
            # go through saved models
            if model_index != 0:
                model_index -= 1
                model_file = model_files[model_index]
                logger.log('loading new model_file %s' % model_file)


            print('timesteps: %d, info: %s' % (number_of_timesteps, str(sum_info)))

            """
            # print head cam
            if number_of_timestep % 100 == 0:
                img = obs[-1 * 16:] * 1
                # img = img.reshape(16, 5, 1)[::-1, :, :]
                img = img.reshape(1, 16, 1)
                # img = img[::-1, :, :]# original image is upside-down, so flip it
                img = img[:, :, 0]

                plt.imshow(img, cmap='gray', animated=True, label="aaa" + str(number_of_timestep))
                plt.title('step: ' + str(number_of_timestep))
                plt.colorbar()
                plt.savefig('aa.png')
                plt.clf()
            """



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

    args = parser.parse_args()
    logger.configure()


    # CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    enjoy(args.env, seed=args.seed)



if __name__ == '__main__':
    main()

# Notes
# in enjoy aufräumen
# es gibt 100 models (auswählbar durch modelindex) da 1 Mio/ 10K(sfs)
# constant 0.5 seems to work better than 1
# which models are displayed