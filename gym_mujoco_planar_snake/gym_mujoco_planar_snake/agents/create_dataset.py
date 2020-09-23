import numpy as np

import os.path as osp
import tensorflow as tf

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import learn
from baselines import logger

import gym, logging
import os

from gym_mujoco_planar_snake.common.env_wrapper import prepare_env, ModelSaverWrapper
from gym_mujoco_planar_snake.common import my_tf_util
from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector


class PPOAgent(object):

    def __init__(self, env):

        self.sess = U.make_session(num_cpu=1, make_default=False)
        self.sess.__enter__()
        self.sess.run(tf.initialize_all_variables())
        #U.initialize()

        with self.sess.as_default():

            self.env = wrapped_env

            self.env.seed(id)

            self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                     ob_space=ob_space,
                                                                                     ac_space=ac_space,
                                                                                     hid_size=64,
                                                                                     num_hid_layers=2
                                                                                     )

    def learn(self, num_timesteps):


        learn(self.env, self.policy_func,
                            max_timesteps=num_timesteps,
                            timesteps_per_actorbatch=2048,
                            clip_param=0.2,
                            entcoeff=0.0,
                            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                            gamma=0.99, lam=0.95,
                            schedule='linear',
                            )

        self.sess.close()


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


    info_collector = InfoCollector(env, {'env': env, 'seed': seed})



    cum_reward = []

    observations = []

    # max_timesteps is set to 1000
    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        # obs[-1] = target_v

        action, _ = pi.act(stochastic, obs)



        obs, reward, done, info = env.step(action)

        observations.append(obs)

        cum_reward.append(reward)

        info['seed'] = seed
        info['env'] = env.spec.id


        # add info
        info_collector.add_info(info)


        # render
        if render:
            env.render()

        number_of_timestep += 1

    # imageio.mimsave('snake.gif', [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=20)

    return observations, cum_reward


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

        # TODO
        time_step = 995000 # 1000000 995000
        step_size = 5000 # 10000
        render = False
        from time import ctime
        start_time = ctime()

        for model_file in model_files:

            observations, cum_reward = run_environment_episode(env, pi, seed, model_file, env._max_episode_steps,
                                                   render=render, stochastic=False, current_timestep=time_step)


            # TODO use time_step or cum_reward
            save_subtrajectories(observations, time_step, cum_reward)

            time_step -= step_size


        print(start_time)
        print(ctime())

def save_subtrajectories(observations, time_step, cum_reward):

    print("Saving Timestep %i"%(time_step))

    starts = np.random.randint(0, EPISODE_MAX_LENGTH - TRAJECTORY_LENGTH, size=100)
    starts.sort()

    # TODO time_step or reward, time_step  + start
    trajectories = np.array([(np.array(observations)[start:start + TRAJECTORY_LENGTH], sum(cum_reward[start:start + TRAJECTORY_LENGTH])) for start in starts])

    path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
    name = "trajectories_{time_step}.npy".format(time_step=time_step)

    with open(os.path.join(path, name), 'wb') as f:
        np.save(f, np.array(trajectories))





def get_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    parser.add_argument('--log_dir', help='save_dir', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    TRAJECTORY_LENGTH = 50 #50 100
    EPISODE_MAX_LENGTH = 1000

    with tf.variable_scope(str(args.seed)):
        enjoy(args.env, seed=args.seed, model_dir=args.model_dir)

