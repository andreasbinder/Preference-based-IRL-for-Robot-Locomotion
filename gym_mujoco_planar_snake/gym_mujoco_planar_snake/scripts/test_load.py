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

class PPO2Agent(object):
    def __init__(self, env, env_type, path, stochastic=False, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model

        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0})

        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env,'cnn')
                elif env_type == 'mujoco':
                    policy = build_policy(env,'mlp')
                else:
                    assert False,' not supported env_type'

                make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                nsteps=1, ent_coef=0., vf_coef=0.,
                                max_grad_norm=0.)
                self.model = make_model()

                self.model_path = path
                self.model.load(path)

        if env_type == 'mujoco':
            with open(path+'.env_stat.pkl', 'rb') as f :
                import pickle
                s = pickle.load(f)
            self.ob_rms = s['ob_rms']
            self.ret_rms = s['ret_rms']
            self.clipob = 10.
            self.epsilon = 1e-8
        else:
            self.ob_rms = None

        self.stochastic = stochastic

    def act(self, obs, reward, done):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)

        with self.graph.as_default():
            with self.sess.as_default():
                if self.stochastic:
                    a,v,state,neglogp = self.model.step(obs)
                else:
                    a = self.model.act_model.act(obs)
        return a


# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic=True):
    number_of_timestep = 0
    done = False

    # timesteps test
    #max_timesteps = 5000

    # load model
    my_tf_util.load_state(model_file)

    #save_dir = '/home/andreas/Desktop/20200625-2355-001000000'

    #my_tf_util.save_state(save_dir)

    #my_tf_util.load_state(save_dir)




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

        # render
        if render:
            env.render()

        number_of_timestep += 1

    return done, number_of_timestep, info_collector, cum_reward


def enjoy(env_id, seed):
    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        # TODO
        # if I also wanted the newest model
        check_for_new_models = False



        max_timesteps = 3000000
        #max_timesteps = 1

        modelverion_in_k_ts = 2510  # better

        model_index = int(max_timesteps / 1000 / 10 - modelverion_in_k_ts / 10)

        # TOdo last saved model
        model_index = 0

        print("actionspace", env.action_space)
        print("observationspace", env.observation_space)

        gym.logger.setLevel(logging.WARN)
        #gym.logger.setLevel(logging.DEBUG)

        run = 0
        model_dir = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/InjuryIndex_6/Fri Jun 26 11:59:30 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/clip_test/InjuryIndex_7/Sat Jun 27 03:59:02 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/clip_test/InjuryIndex_7/Mon Jun 29 11:36:26 2020/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'
        #'/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/snake_test/run0_-1/Thu Jun 11 10:25:54 2020_0.2/models/Mujoco-planar-snake-cars-angle-line-v1/ppo'

        model_files = get_model_files(model_dir)

        # model_file = get_latest_model_file(model_dir)

        model_index = 40#0
        model_file = model_files[model_index]
        print('available models: ', len(model_files))
        #model_file = model_files[model_index]
        # model_file = model_files[75]
        logger.log("load model_file: %s" % model_file)

        sum_info = None
        pi = policy_fn('pi'+str(run), env.observation_space, env.action_space)

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
            env._max_episode_steps = 50
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            done, number_of_timesteps, info_collector, rewards = run_environment_episode(env, pi, seed, model_file,
                                                                                env._max_episode_steps, render=True,
                                                                                stochastic=False)


            info_collector.episode_info_print()

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

        print(len(sum_reward))
        arr = np.array(sum_reward)
        


        #np.save("rewards.npy", arr)

            



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