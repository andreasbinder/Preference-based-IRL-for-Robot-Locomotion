# https://github.com/openai/gym/blob/master/gym/core.py
import tensorflow as tf
#tf.enable_eager_execution()
import gym
import numpy as np
from gym_mujoco_planar_snake.common.reward_net import SimpleNet, RewardNet
import gym_mujoco_planar_snake.common.my_tf_util as U
from baselines.common.vec_env import VecEnvWrapper
from gym.core import RewardWrapper
from tensorflow import keras


class RewardNetWrapper_v2(RewardWrapper):

    def __init__(self, venv, model_dir):
        RewardWrapper.__init__(self, venv)

        self.venv = venv
        self.model_dir = model_dir
        self.model = SimpleNet()
        self.model.load_weights(self.model_dir)
        self.true_reward = None

        self.dense = keras.layers.Dense(1350, input_shape=(27,))
        #self.modified_model = self.model(dense)

    def step(self, action):
        obs, rews, news, infos = self.venv.step(action)

        self.true_reward = rews

        # obs = [obs for _ in range(10)]
        #
        # obs = np.stack(obs, axis=0)

        #obs = tf.expand_dims(tf.convert_to_tensor(obs.reshape((1350,)).astype("float32")), axis=0)

        obs = tf.expand_dims(tf.convert_to_tensor(obs.reshape((27,)).astype("float32")), axis=0)

        obs = self.dense(obs)

        reward, abs_reward = self.model.reward(obs)

        abs_reward = tf.unstack(abs_reward)[0]

        print(abs_reward)

        # TODO return reward or abs_reward
        #return obs, abs_reward, news, infos

        # TODO test whether it always returns the constant
        #abs_reward = 0
        #print("in reward")
        return obs, abs_reward, news, infos


    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def reward(self, reward):
        return reward

    def get_true_reward(self):
        return self.true_reward

class RandomWrapper(VecEnvWrapper):

    def __init__(self, venv, model_dir):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        pass

    def reset(self, **kwargs):
        obs = self.venv.reset()

        return obs


class HorizonRewardWrapper(RewardWrapper):

    def __init__(self, venv, model_dir, sess):
        RewardWrapper.__init__(self, venv)

        # TODO need seed for consistency

        self.venv = venv
        self.model_dir = model_dir
        self.model = SimpleNet()
        self.model.load_weights(self.model_dir)
        self.true_reward = None

        self.sess = sess


        #self.dense = keras.layers.Dense(1350, input_shape=(27,))
        #self.modified_model = self.model(dense)

        self.counter = 0



        #self.history = np.random.randn(50, 27)
        # TODO for testing
        self.history = np.zeros((50, 27))

    #@tf.function
    def step(self, action):
        obs, rews, news, infos = self.venv.step(action)

        #self.counter += 1

        #print("Step", str(self.counter))

        self.true_reward = rews

        horizon = np.concatenate((self.history,
                                  np.expand_dims(obs, axis=0)), axis=0)[1:]

        observations = np.expand_dims(horizon.reshape((1350,)), axis=0)

        reward, abs_reward = self.model.reward(observations)

        abs_reward = self.sess.run(abs_reward)[0]

        self.history = horizon

        # TODO return reward or abs_reward
        return obs, abs_reward, news, infos


    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def reward(self, reward):
        return reward

    def get_true_reward(self):
        return self.true_reward

class SingleStepRewardWrapper(RewardWrapper):

    def __init__(self, venv, model_dir, sess):
        RewardWrapper.__init__(self, venv)

        # TODO need seed for consistency

        self.venv = venv
        self.model_dir = model_dir
        self.model = RewardNet()
        self.model.load_weights(self.model_dir)
        self.true_reward = None
        self.sess = sess

        self.counter = 0

        #self.history = np.random.randn(50, 27)
        # TODO for testing
        self.history = np.zeros((50, 27))

    #@tf.function
    def step(self, action):
        import time

        start = time.time()


        obs, rews, news, infos = self.venv.step(action)

        self.counter += 1
        sess = self.sess

        print("Step", str(self.counter))

        self.true_reward = rews

        '''horizon = np.concatenate((self.history,
                                  np.expand_dims(obs, axis=0)), axis=0)[1:]'''

        #observations = np.expand_dims(horizon.reshape((1350,)), axis=0)

        observations = np.expand_dims(obs, axis=0)

        reward, abs_reward = self.model.reward(observations)

        #abs_reward = sess.run(abs_reward)[0]

        abs_reward = abs_reward.eval()[0]

        #sess.close()

        #print(self.history)
        #print(self.history.shape)

        #print(observations)

        end = time.time()
        print(end - start)

        if self.counter == 500:
            print(observations)
            import sys
            sys.exit()

        #self.history = horizon

        # TODO return reward or abs_reward
        return obs, abs_reward, news, infos


    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def reward(self, reward):
        return reward

    def get_true_reward(self):
        return self.true_reward

class Dummywrapper(RewardWrapper):

    def __init__(self, venv, model_dir, sess):
        RewardWrapper.__init__(self, venv)

        # TODO need seed for consistency

        self.venv = venv
        self.model_dir = model_dir
        self.model = RewardNet()
        self.model.load_weights(self.model_dir)
        self.true_reward = None

        self.sess = sess


        #self.dense = keras.layers.Dense(1350, input_shape=(27,))
        #self.modified_model = self.model(dense)

        self.counter = 0



        #self.history = np.random.randn(50, 27)
        # TODO for testing
        self.history = np.zeros((50, 27))

    #@tf.function
    def step(self, action):
        obs, rews, news, infos = self.venv.step(action)

        self.counter += 1

        print("Step", str(self.counter))


        return obs, rews, news, infos


    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def reward(self, reward):
        return reward

    def get_true_reward(self):
        return self.true_reward