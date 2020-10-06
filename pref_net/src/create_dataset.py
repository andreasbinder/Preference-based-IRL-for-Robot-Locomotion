
import os.path as osp


import argparse
import os, datetime


from baselines.ppo1 import mlp_policy

import gym, logging
import os


from src.utils.configs import Configs
from src.utils.info_collector import InfoCollector, InfoDictCollector
from src.utils.model_saver import ModelSaverWrapper
from src.utils.agent import PPOAgent
from src.utils.seeds import set_seeds


from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np






if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="/home/andreas/Documents/pbirl-bachelorthesis/pref_net/configs.yml")
    args = parser.parse_args()

    configs_file = Configs(args.path_to_configs)
    configs = configs_file["create_dataset"]

    # constant hyperparameter
    SAVE_CKPT_DIR = os.path.join(configs["checkpoint_dir"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(SAVE_CKPT_DIR)

    SAVE_DATA_DIR = os.path.join(configs["data_dir"], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(SAVE_DATA_DIR)

    RENDER = False
    ENV_ID = 'Mujoco-planar-snake-cars-angle-line-v1'
    TRAIN_NAME = "train.npy"
    EXTRAPOLATE_NAME = "extrapolate.npy"

    # stores full episodes
    full_episodes = []

    # seeds
    set_seeds(configs["seed"])

    # skip warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    with tf.variable_scope(str(configs["variable_scope"])):

        env = gym.make(ENV_ID)
        env = ModelSaverWrapper(env, SAVE_CKPT_DIR, configs["save_frequency"])
        env.seed(configs["seed"])

        info_dict_collector = InfoDictCollector(env)

        policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                            ob_space=ob_space,
                                                                            ac_space=ac_space,
                                                                            hid_size=64,
                                                                            num_hid_layers=2
                                                                            )
        pi = policy_fn("pi", env.observation_space, env.action_space)

        sess = U.make_session(num_cpu=1, make_default=False)
        sess.__enter__()
        sess.run(tf.initialize_all_variables())
        with sess.as_default():

            # optimize policy
            agent = PPOAgent(env, pi, policy_fn)
            agent.learn(configs["num_timesteps"])

            # store ckpts
            list = [x[:-len(".index")] for x in os.listdir(SAVE_CKPT_DIR) if x.endswith(".index")]
            list.sort(key=str.lower)
            model_files = [osp.join(SAVE_CKPT_DIR, ele) for ele in list]
            num_models = len(model_files)
            print('available models: ', len(model_files))

            # loop over stored model files
            for model_file in model_files:

                logger.log("load model_file: %s" % model_file)


                time_step = int(model_file[-9:])
                env.unwrapped.metadata['target_v'] = 0.1

                observations, cum_reward = agent.run_environment_episode(env,
                                                                         pi,
                                                                         configs["seed"],
                                                                         model_file,
                                                                         configs["episode_length"],
                                                                         render=RENDER,
                                                                         stochastic=False
                                                                         )


                trajectories = np.array([(observations, time_step, cum_reward)])

                full_episodes.append(np.array(trajectories))


            # Split in train and extrapolation data
            full_episodes = np.concatenate(full_episodes)
            FACTOR = int(num_models * float(configs["percentage"] / configs["num_timesteps"]))


            TRAIN = full_episodes[:FACTOR]
            EXTRAPOLATE = full_episodes[FACTOR:]


            with open(os.path.join(SAVE_DATA_DIR, TRAIN_NAME), 'wb') as f:
                np.save(f, np.array(TRAIN))

            with open(os.path.join(SAVE_DATA_DIR, EXTRAPOLATE_NAME), 'wb') as f:
                np.save(f, np.array(EXTRAPOLATE))



            from src.benchmark.plot_results import return_all_episode_statistics

            # TODO
            return_all_episode_statistics(TRAIN)

            from src.common.data_util import generate_dataset_from_full_episodes

            # TODO
            train_set = generate_dataset_from_full_episodes(TRAIN, 50, 100)
            name_for_default = "subtrajectories.npy"
            with open(os.path.join(SAVE_DATA_DIR, name_for_default), 'wb') as f:
                np.save(f, np.array(train_set))






