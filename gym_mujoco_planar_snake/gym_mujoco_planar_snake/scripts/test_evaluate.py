#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import argparse
import tensorflow as tf
from gym.envs.registration import register
import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper



from time import ctime




def get_latest_model_file(model_dir):
    return get_model_files(model_dir)[0]


def get_model_files(model_dir):
    list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
    list.sort(key=str.lower, reverse=True)

    files = [osp.join(model_dir, ele) for ele in list]
    return files


def get_model_dir(env_id, name):
    model_dir = osp.join(logger.get_dir(), 'models')
    os.mkdir(model_dir)
    model_dir = ModelSaverWrapper.gen_model_dir_path(model_dir, env_id, name)
    logger.log("model_dir: %s" % model_dir)
    return model_dir


def policy_fn(name, ob_space, ac_space):
    from baselines.ppo1 import mlp_policy
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)


def train_ppo1(env_id, num_timesteps, sfs, seed, fixed_joints, run):
    from gym_mujoco_planar_snake.agents.policies.pposgd_simple_di import learn
    # from baselines.ppo1 import pposgd_simple

    clip_value = None
    ########################################################################
    # TODO:                                                                #
    # set values for injured joints                                        #
    ########################################################################
    # v1 set Box2D actionspace, has no influence on policy

    # env.action_space.high[2] = 0.1
    # env.action_space.low[2] = -0.1

    # v2 clip action values to specific range

    import numpy as np
    clip_value = np.random.uniform(0, 1.5)
    logger.record_tabular("clip_value", clip_value)
    print(clip_value)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)


    model_dir = get_model_dir(env_id, 'ppo')

    # monitor tensotrain_ppo1(env_id, num_timesteps, sfs, seed, fixed_joints)rboard
    log_dir = osp.join(logger.get_dir(), 'log_ppo')
    logger.log("log_dir: %s" % log_dir)
    env = bench.Monitor(env, log_dir)

    env = ModelSaverWrapper(env, model_dir, sfs)






    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    # pposgd_simple.
    learn(env, policy_fn,
          max_timesteps=num_timesteps,
          timesteps_per_actorbatch=2048,
          clip_param=0.2,  # TODO 0.2
          entcoeff=0.0,
          optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
          gamma=0.99, lam=0.95,
          schedule='linear',  # TODO linear
          fixed_joints=fixed_joints,
          run=run,
          clip_value=clip_value
          )
    logger.log("End training at " + ctime())
    env.close()
    sess.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6
    parser.add_argument('--sfs', help='save_frequency_steps', type=int, default=10000)  # for mujoco
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--log_dir', help='log directory', default='gym_mujoco_planar_snake/log/initial_PPO_runs/')
    parser.add_argument('--injured', type=bool, default=False)
    parser.add_argument('--run1', nargs='*', type=int, default=None)
    parser.add_argument('--run2', nargs='*', type=int, default=None)
    parser.add_argument('--run3', nargs='*', type=int, default=None)
    parser.add_argument('--run4', nargs='*', type=int, default=None)
    parser.add_argument('--run5', nargs='*', type=int, default=None)

    args = parser.parse_args()

    # env = os.environ.copy()

    # env["OPENAI_LOGDIR"] = 'gym_mujoco_planar_snake/log/initial__PPO_runs'

    #args.num_timesteps = 1

    # logger.logkv("a", 4)

    # CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''




    runs = [args.run1, args.run2, args.run3, args.run4, args.run5]

    runs = [args.run1]

    #
    args.log_dir = 'gym_mujoco_planar_snake/log/clip_test/'

    #args.log_dir = '/media/andreas/INTENSO/initial_PPO_runs/'

    for i, run in enumerate(runs):

        if run is not None:

            joint_info = ""
            for s in run:
                joint_info += str(s)

            #logger.configure(dir=args.log_dir + 'run' + str(i) + '_' + joint_info + r"/" + ctime())

            logger.configure(dir=args.log_dir + 'InjuryIndex_' + joint_info + r"/" + ctime())

            #adjusted to array indexes
            '''
            if run[0] == -1:
                run = None
            '''
            if args.injured:
                run = None



            train_ppo1(args.env, num_timesteps=args.num_timesteps, sfs=args.sfs, seed=args.seed, fixed_joints=run, run=i)

        #tf.reset_default_graph()
        # pass

    # train_ppo1(args.env, num_timesteps=args.num_timesteps, sfs=args.sfs, seed=args.seed, fixed_joints=args.run1)


if __name__ == '__main__':
    main()
