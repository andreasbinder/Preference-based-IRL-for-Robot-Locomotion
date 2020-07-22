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
        '''        self.dense1 = Dense(256, nn.relu, input_shape=(1350, ))
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()'''
        self.model = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            Dense(256, nn.relu),
            Flatten(),
            Dense(1)
        ])

    def reward(self, traj):
        '''       sum_rewards = 0
        sum_abs_rewards = 0'''

        sum_rewards = tf.Variable(0.)
        sum_abs_rewards = tf.Variable(0.)
        x = traj
        '''x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.flatten(x)
        r = self.dense4(x)'''

        r = self.model(traj)
        # sum_rewards += keras_sum(r)

        '''        sum_rewards += keras_sum(r, axis=1)
        sum_abs_rewards += keras_sum(keras_abs(r))

        return sum_rewards, sum_abs_rewards'''

        '''        sum_rewards += tf.Variable(keras_sum(r, axis=1))
        sum_abs_rewards += tf.Variable(keras_sum(keras_abs(r)))'''

        '''sum_rewards = sum_rewards + keras_sum(r, axis=1)
        sum_abs_rewards = sum_abs_rewards + keras_sum(keras_abs(r))'''
        sum_rewards = tf.add(sum_rewards, keras_sum(r, axis=1))
        sum_abs_rewards = tf.add(sum_abs_rewards, keras_sum(keras_abs(r)))

        return sum_rewards, sum_abs_rewards

    def call(self, inputs, training=True, **kwargs):
        # traj shape (50, 27)
        # currently: 64(bs), 2, 1350 -> wie bekomme ich 2 Mal 64, 1, 1350
        traj_i, traj_j = inputs

        '''traj_i = inputs[:, 0, :]
        traj_j = inputs[:, 1, :]'''

        cum_r_i, abs_r_i = self.reward(traj_i)
        cum_r_j, abs_r_j = self.reward(traj_j)

        from tensorflow import argmax as tf_argmax
        logits = tensorflow.stack([[cum_r_i], [cum_r_j]], axis=1)
        logits = tf.squeeze(tf_argmax(logits, axis=1))
        logits = tf.dtypes.cast(logits, tf.float32)

        return logits, tf.add(abs_r_i, abs_r_j)
        # return tensorflow.stack([[cum_r_i],[cum_r_j]], axis=1), abs_r_i + abs_r_j


class SimpleNet(tf.keras.Model):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.dense1 = Dense(256, nn.relu, input_shape=(1350,))
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

        self.model = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            Dense(256, nn.relu),
            Dense(1)
        ])

    def reward(self, x):
        '''
            x.shape: (batch_size, 1350)
        '''

        r = self.model(x)

        r = keras_sum(r, axis=1)
        #r_abs = keras_sum(keras_abs(r), axis=1)
        r_abs = None #dummy value

        return r, r_abs

    def call(self, inputs, training=True, **kwargs):

        traj_i, traj_j = inputs

        cum_r_i, cum_r_i_abs = self.reward(traj_i)
        cum_r_j, cum_r_j_abs = self.reward(traj_j)

        x = tf.subtract(cum_r_i, cum_r_j)

        x = tf.keras.activations.sigmoid(x)

        return x, None
