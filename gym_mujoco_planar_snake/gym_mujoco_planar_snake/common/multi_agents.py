#!/usr/bin/env python
import tensorflow as tf
import torch


from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy
from baselines.ppo1.pposgd_simple import learn
from baselines.bench.monitor import Monitor
from baselines import logger

import gym, logging
import os


#from gym_mujoco_planar_snake.common.reward_wrapper import *
#from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.env_wrapper import prepare_env, ModelSaverWrapper
from gym_mujoco_planar_snake.common.misc_util import Configs


class PPOAgent(object):

    def __init__(self, env, id, log_dir, sfs, save_check_points, learned_reward, configs):

        self.id = str(id)


        with tf.variable_scope(self.id):

                self.sess = U.make_session(num_cpu=1, make_default=False)
                self.sess.__enter__()
                self.sess.run(tf.initialize_all_variables())
                #U.initialize()

                with self.sess.as_default():

                    wrapped_env = prepare_env(env,
                                              save_check_points,
                                              log_dir,
                                              sfs,
                                              id,
                                              learned_reward,
                                              configs)
                    self.env = wrapped_env


                    #self.env.seed(configs.get_seed())
                    self.env.seed(id)

                    # TODO check if I need to seed action_space too
                    #self.env.action_space.seed(id)


                    self.policy_func = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                                             ob_space=ob_space,
                                                                                             ac_space=ac_space,
                                                                                             hid_size=64,
                                                                                             num_hid_layers=2
                                                                                             )
        self.policy = None


    def learn(self, num_timesteps):

        with tf.variable_scope(str(self.id)):

            print("Agent %s starts learning" % self.id)

            self.policy = learn(self.env, self.policy_func,
                                max_timesteps=num_timesteps,
                                timesteps_per_actorbatch=2048,
                                clip_param=0.2,
                                entcoeff=0.0,
                                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear',
                                )

            # print(self.env.get_episode_rewards())

        self.sess.close()
        return self.policy

    def act(self, obs, stochastic=False):
        return self.policy.act(stochastic, obs)


class AgentSquad(object):

    def __init__(self, configs, learned_reward, log_dir):

        # TODO test value generation
        # load configurations from yaml file
        # configs = Configs(path_to_configs)


        # initialize env
        self.configs = configs
        self.log_dir = log_dir
        self.env_id = configs.get_env_id() # "Mujoco-planar-snake-cars-angle-line-v1"
        self.env = gym.make(self.env_id)


        self.learned_reward = learned_reward

        if learned_reward:
            self.num_agents = configs.get_num_improved_agents()
            self.num_timesteps = configs.get_num_improved_timesteps()
            self.save_check_points = configs.get_save_improved_checkpoints()
            self.sfs = configs.get_improved_sfs()
        else:
            self.num_agents = configs.get_num_initial_agents()
            self.num_timesteps = configs.get_num_initial_timesteps()
            self.save_check_points = configs.get_save_initial_checkpoints()
            self.sfs = configs.get_initial_sfs()



    def learn(self):

        num_prev_agents = self.configs.get_num_initial_agents() if self.learned_reward else 0
        agents = [PPOAgent(env=self.env,
                           id=id,
                           log_dir=self.log_dir,
                           sfs=self.sfs,
                           save_check_points=self.save_check_points,
                           learned_reward=self.learned_reward,
                           configs=self.configs)

                  for id in range(num_prev_agents, self.num_agents + num_prev_agents)]

        for agent in agents:
            agent.learn(self.num_timesteps)




from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy

import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np


import os
import os.path as osp


from gym_mujoco_planar_snake.common import my_tf_util

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector



def get_model_files(model_dir):
    list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
    list.sort(key=str.lower, reverse=True)

    files = [osp.join(model_dir, ele) for ele in list]
    return files




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

        time_step = 500000 # 1000000
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


    agent_id = args.model_dir[-1]

    with tf.variable_scope(agent_id):
        enjoy(args.env, seed=args.seed, model_dir=args.model_dir)



