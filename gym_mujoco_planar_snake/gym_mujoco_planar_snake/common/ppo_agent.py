from baselines.ppo1.pposgd_simple import learn
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy

from gym_mujoco_planar_snake.common.env_wrapper import prepare_env, ModelSaverWrapper
from gym_mujoco_planar_snake.common import my_tf_util
from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

import os
import os.path as osp


class PPOAgent(object):

    def __init__(self, id, log_dir, learned_reward, configs):

        self.id = str(id)
        self.configs = configs
        self.log_dir = log_dir
        self.env_id = configs.get_env_id() # "Mujoco-planar-snake-cars-angle-line-v1"
        self.env = gym.make(self.env_id)
        self.max_episode_steps = self.env._max_episode_steps


        self.learned_reward = learned_reward

        self.save_check_points = configs.get_save_improved_checkpoints() if learned_reward else configs.get_save_initial_checkpoints()
        self.sfs = configs.get_improved_sfs() if learned_reward else configs.get_initial_sfs()

        with tf.variable_scope(self.id):

                self.sess = U.make_session(num_cpu=1, make_default=False)
                self.sess.__enter__()
                self.sess.run(tf.initialize_all_variables())
                #U.initialize()

                with self.sess.as_default():

                    # env, save_check_points, save_dir, sfs, id, learned_reward, configs

                    wrapped_env = prepare_env(self.env,
                                              self.save_check_points,
                                              log_dir,
                                              self.sfs,
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


    def learn(self):

        num_timesteps = self.configs.get_num_improved_timesteps() if self.learned_reward else self.configs.get_num_initial_timesteps()

        with tf.variable_scope(self.id):


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
        # TODO open session
        self.sess.close()
        return self.policy

    def act(self, obs, stochastic=False):
        return self.policy.act(stochastic, obs)


    def create_trajs_from_checkpoint(self, default_dataset_dir="/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"):





        with tf.variable_scope(self.id):


            '''self.sess = U.make_session(num_cpu=1, make_default=False)
            self.sess.__enter__()
            self.sess.run(tf.initialize_all_variables())
            # U.initialize()

            with self.sess.as_default():
            '''


            gym.logger.setLevel(logging.WARN)

            model_dir = osp.join(self.log_dir, "agent_" + self.id)
            model_files = self._get_model_files(model_dir)

            model_index = 0
            model_file = model_files[model_index]


            print('available models: ', len(model_files))
            logger.log("load model_file: %s" % model_file)

            '''policy_fn = lambda name, ob_space, ac_space: mlp_policy.MlpPolicy(name=name,
                                                                              ob_space=ob_space,
                                                                              ac_space=ac_space,
                                                                              hid_size=64,
                                                                              num_hid_layers=2
                                                                              )'''


            #pi = policy_fn('pi', self.env.observation_space, self.env.action_space)


            pi = self.policy_func('pi', self.env.observation_space, self.env.action_space)



            time_step = self.configs.get_num_initial_timesteps()
            step_size = self.configs.get_initial_sfs()  # 10000
            render = False


            trajectories = []

            for model_file in model_files:
                #self.env.unwrapped.metadata['target_v'] = 0.15

                observations = self._run_environment_episode(self.env, pi, self.configs.get_seed(), model_file, self.max_episode_steps,
                                                       render=render, stochastic=False)

                episode_trajectories = self._save_subtrajectories(observations, time_step)

                trajectories.append(episode_trajectories)

                time_step -= step_size

            path = default_dataset_dir
            name = "trajectories_{id}.npy".format(id=self.id)

            with open(os.path.join(path, name), 'wb') as f:
                np.save(f, np.concatenate(trajectories, axis=0))




    def _run_environment_episode(self, env, pi, seed, model_file, max_timesteps, render, stochastic):
        number_of_timestep = 0
        done = False

        # load model
        my_tf_util.load_state(model_file)

        # TODO get timestep from model file

        #pi = self.policy_func('pi', self.env.observation_space, self.env.action_space)

        obs = env.reset()

        observations = []

        # max_timesteps is set to 1000
        while (not done) and number_of_timestep < max_timesteps:

            action, _ = pi.act(stochastic, obs)

            obs, reward, done, info = env.step(action)

            observations.append(obs)

            # render
            if render:
                env.render()

            number_of_timestep += 1


        return observations




    def _save_subtrajectories(self, observations, time_step):

        print("Saving Timestep %i"%(time_step))

        starts = np.random.randint(0, 950, size=100)
        starts.sort()

        trajectories = np.array([(np.array(observations)[start:start + 50], time_step  + start) for start in starts])

        return trajectories

    def _get_model_files(self, model_dir):
        list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
        list.sort(key=str.lower, reverse=True)

        files = [osp.join(model_dir, ele) for ele in list]
        return files









