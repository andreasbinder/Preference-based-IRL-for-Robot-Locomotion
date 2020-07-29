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

        # self.optimizer = optimizer
        # self.loss = loss
        # self.accuracy = accuracy

        self.dense1 = Dense(256, nn.relu, input_shape=(27,))
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

        self.model = tf.keras.Sequential([
            Dense(128, nn.sigmoid, input_shape=(27,)),
            Dense(128, nn.sigmoid),
            #Dense(256, nn.relu),
            #Flatten(),
            Dense(1)
        ])

        self.model_v5 = tf.keras.Sequential([
            Dense(256, nn.sigmoid, input_shape=(27,)),
            BatchNormalization(),
            Dense(256, nn.sigmoid),
            BatchNormalization(),
            Dense(256, nn.sigmoid),
            BatchNormalization(),
            # Dense(256, nn.relu),
            # Flatten(),
            Dense(1)
        ])

        self.model_v3 = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(27,)),
            Dense(256, nn.relu),
            # Dense(256, nn.relu),
            # Flatten(),
            Dense(1)
        ])



        self.model_v2 = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            Dense(256, nn.relu),
            Dense(1)
        ])

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.model.summary(line_length, positions, print_fn)


    def reward(self, x):
        '''
            x.shape: (batch_size, 1350)
        '''

        pred = self.model(x)

        r = keras_sum(pred, axis=1)

        # r = keras_sum(pred, axis = 1)

        #r_abs = keras_sum(keras_abs(pred), axis=1)
        r_abs = keras_sum(keras_abs(pred), axis=1)
        #r_abs = None #dummy value

        return r, r_abs

    def call(self, inputs, training=True, **kwargs):



        traj_i, traj_j = inputs

        batch_size = traj_i.shape[0]

        ####################################
        # TODO get batchsize instead of hardcoded value
        ####################################

        traj_i, traj_j = tf.reshape(traj_i, [traj_i.shape[0] * 50, 27]), tf.reshape(traj_i, [traj_j.shape[0] * 50, 27])

        rew_obs_i, rew_obs_abs_i = self.reward(traj_i)
        rew_obs_j, rew_obs_abs_j = self.reward(traj_j)

        rew_obs_i, rew_obs_abs_i = tf.reshape(rew_obs_i, [batch_size, 50]), tf.reshape(rew_obs_abs_i, [batch_size, 50])
        rew_obs_j, rew_obs_abs_j = tf.reshape(rew_obs_j, [batch_size, 50]), tf.reshape(rew_obs_abs_j, [batch_size, 50])

        cum_r_i, cum_r_i_abs = keras_sum(rew_obs_i, axis=1), keras_sum(rew_obs_abs_i, axis=1)
        cum_r_j, cum_r_j_abs = keras_sum(rew_obs_j, axis=1), keras_sum(rew_obs_abs_j, axis=1)

        #cum_r_i, traj_j = tf.reshape(traj_i, [cum_r_i.shape[0] * 50, ]), tf.reshape(traj_i, [traj_j.shape[0] * 50, ])


        ####################################
        #cum_r_i, cum_r_i_abs = self.reward(traj_i)
        #cum_r_j, cum_r_j_abs = self.reward(traj_j)



        x = tf.subtract(cum_r_i, cum_r_j)

        x = tf.keras.activations.sigmoid(x)

        return x, cum_r_i_abs + cum_r_j_abs


class SimpleNet(tf.keras.Model):

    def __init__(self):
        super(SimpleNet, self).__init__()

        # self.optimizer = optimizer
        # self.loss = loss
        # self.accuracy = accuracy

        self.dense1 = Dense(256, nn.relu, input_shape=(1350,))
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

        self.model = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            #Dense(256, nn.relu),
            #Flatten(),
            Dense(1)
        ])



        self.model_v2 = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            Dense(256, nn.relu),
            Dense(1)
        ])


    def reward(self, x):
        '''
            x.shape: (batch_size, 1350)
        '''

        pred = self.model(x)

        r = keras_sum(pred, axis=1)
        #r_abs = keras_sum(keras_abs(r), axis=1)
        r_abs = keras_sum(keras_abs(pred), axis=1)
        #r_abs = None #dummy value

        return r, r_abs

    def call(self, inputs, training=True, **kwargs):

        traj_i, traj_j = inputs

        cum_r_i, cum_r_i_abs = self.reward(traj_i)
        cum_r_j, cum_r_j_abs = self.reward(traj_j)

        x = tf.subtract(cum_r_i, cum_r_j)

        x = tf.keras.activations.sigmoid(x)

        return x, cum_r_i_abs + cum_r_j_abs


class TransferNet(tf.keras.Model):

    def __init__(self):
        super(TransferNet, self).__init__()

        # self.optimizer = optimizer
        # self.loss = loss
        # self.accuracy = accuracy

        self.dense1 = Dense(256, nn.relu, input_shape=(1350,))
        self.dense2 = Dense(256, nn.relu)
        self.dense3 = Dense(256, nn.relu)
        self.dense4 = Dense(1)
        self.flatten = Flatten()

        self.model = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            #Dense(256, nn.relu),
            #Flatten(),
            Dense(1)
        ])



        self.model_v2 = tf.keras.Sequential([
            Dense(256, nn.relu, input_shape=(1350,)),
            Dense(256, nn.relu),
            Dense(256, nn.relu),
            Dense(1)
        ])


    def reward(self, x):
        '''
            x.shape: (batch_size, 1350)
        '''

        pred = self.model(x)

        r = keras_sum(pred, axis=1)
        #r_abs = keras_sum(keras_abs(r), axis=1)
        r_abs = keras_sum(keras_abs(pred), axis=1)
        #r_abs = None #dummy value

        return r, r_abs

    def call(self, inputs, training=True, **kwargs):

        traj_i, traj_j = inputs

        cum_r_i, cum_r_i_abs = self.reward(traj_i)
        cum_r_j, cum_r_j_abs = self.reward(traj_j)

        x = tf.subtract(cum_r_i, cum_r_j)

        x = tf.keras.activations.sigmoid(x)

        return x, cum_r_i_abs + cum_r_j_abs
