import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow import keras
from gym_mujoco_planar_snake.common.reward_net import RewardNet

from tensorflow.math import argmax as tf_argmax


class Trainer:

    def __init__(self, hparams, execute_eagerly=True):
        self.hparams = hparams

        # especially useful for debugging
        if execute_eagerly:
            pass

    def fit_pair(self, dataset):

        x_train, y_train = dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype("float32") , y_train)
        )
        dataset = dataset.shuffle(buffer_size=1024).batch(32)
        model = RewardNet()

        # TODO welche loss function sollte man verwenden
        #loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        for epoch in range(2):
            # Iterate over the batches of a dataset.
            for step, (x, y) in enumerate(dataset):

                logits = tf.Variable(0, trainable=True)

                with tf.GradientTape() as tape:
                    # TODO make tf watch my variables for computing gradients
                    #tape.watch(y)
                    #tape.watch(logits)


                    logits, regularization_constant = model(x)

                    # TODO transform
                    # I somehow thought to use label 1 and 2
                    #logits = tf.add(tf.squeeze(tf_argmax(logits, axis=1)), 1)

                    logits = tf.squeeze(tf_argmax(logits, axis=1))
                    logits = tf.dtypes.cast(logits, tf.float32)
                    y = tf.subtract(y, 1)

                    # Compute the loss value for this batch.
                    loss_value = loss_fn(y, logits)

                # Update the weights of the model to minimize the loss value.
                gradients = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                # Logging the current accuracy value so far.
                if step % 200 == 0:
                    print("Epoch:", epoch, "Step:", step)
