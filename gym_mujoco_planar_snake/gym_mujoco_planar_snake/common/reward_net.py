import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Dense, Reshape, Flatten, Dropout
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow.nn as nn
from tensorflow.keras.backend import sum as keras_sum
from tensorflow.keras.backend import abs as keras_abs



class RewardNet(tf.keras.Model):

    def __init__(self):
        super(RewardNet, self).__init__()
        self.dense1 = Dense(256, nn.relu)
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

    def reward(self, traj):
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.flatten(x)
        r = self.dense4(x)
        #sum_rewards += keras_sum(r)
        sum_rewards += keras_sum(r, axis=1)
        sum_abs_rewards += keras_sum(keras_abs(r))

        return sum_rewards, sum_abs_rewards



    def call(self, inputs, training=True, **kwargs):
        # traj shape (50, 27)
        # currently: 64(bs), 2, 1350 -> wie bekomme ich 2 Mal 64, 1, 1350
        #traj_i, traj_j = inputs
        traj_i = inputs[:, 0, :]
        traj_j = inputs[:, 1, :]
        cum_r_i, abs_r_i = self.reward(traj_i)
        cum_r_j, abs_r_j = self.reward(traj_j)

        return tensorflow.stack([[cum_r_i],[cum_r_j]], axis=1), abs_r_i + abs_r_j

#net = RewardNet()

