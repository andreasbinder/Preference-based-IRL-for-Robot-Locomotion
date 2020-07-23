import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from tensorflow import keras

import numpy as np

from gym_mujoco_planar_snake.common.reward_net import RewardNet, SimpleNet


class Trainer:

    def __init__(self, hparams,
                 save_path='/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/',
                 execute_eagerly=True):
        self.hparams = hparams
        self.save_path = save_path
        self.results = {
            "loss": 0.,
            "accuracy": 0.
        }
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.accuracy = None
        self.train_loss = None

        # initialize validation metrics
        self.test_loss = None
        self.test_accuracy = None

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @staticmethod
    def add_to_tensorboard(self):
        pass

    @tf.function  # Make it fast.
    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            # tell gradient tape to watch variables
            # tape.watch(y)
            # tape.watch(x)

            logits, regularization_constant = self.model(x)

            # Compute the loss value for this batch.
            loss_value = self.loss_fn(y, logits)

            # Update accuracy
            accuracy.update_state(y, logits)

            # Update the weights of the model to minimize the loss value.
            gradients = tape.gradient(loss_value, model.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return loss_value

    @tf.function
    def test_step(self, x_test, y_test):
        predictions, _ = self.model(x_test)

        loss = self.loss_fn(y_test, predictions)

        test_accuracy.update_state(y_test, predictions)
        # test_loss(loss)
        # test_accuracy(y_test, predictions)

        return loss



    def fit_pair_v4(self, dataset):
        # get hyperparameters and data
        batch_size = self.hparams["batch_size"]
        lr = self.hparams["lr"]
        epochs = self.hparams["epochs"]
        x_train, y_train = dataset

        # use tf iterator
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype("float32"), y_train)
        )

        # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
        size = int(x_train.shape[0])
        dataset = dataset.shuffle(size).batch(batch_size)

        # TODO make ratios injectable
        train_size = int(size * 0.8 / batch_size)
        test_size = int(size * 0.1 / batch_size)

        # initialize datasets
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(test_size)
        test_dataset = test_dataset.take(test_size)

        # initialize training metrics
        self.accuracy = tf.keras.metrics.BinaryAccuracy()
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        # initialize validation metrics
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy('test_accuracy')

        from tensorboardX import SummaryWriter
        import datetime

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'gym_mujoco_planar_snake/log/tensorboard/gradient_tape/' + current_time + '/train'
        val_log_dir = 'gym_mujoco_planar_snake/log/tensorboard/gradient_tape/' + current_time + '/val'

        train_summary_writer = SummaryWriter(train_log_dir)
        val_summary_writer = SummaryWriter(val_log_dir)


        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.model = SimpleNet()

        #model.compile(optimizer, loss_fn, [accuracy])





        for epoch in range(epochs):

            # Iterate over the batches of a dataset.
            # for step, (x, y) in enumerate(dataset):
            for step, (x, y) in enumerate(train_dataset):

                train_l = 0.

                x = x[:, 0, :], x[:, 1, :]

                loss_value = self.train_on_batch(x, y)
                train_l += loss_value
                self.train_loss(loss_value)


                train_summary_writer.add_scalar('loss', self.train_loss.result().numpy(), epoch)
                train_summary_writer.add_scalar('accuracy', self.accuracy.result().numpy(), epoch)

                # final accuracy
                final_acc = self.accuracy.result()

            for step, (x, y) in enumerate(val_dataset):


                x = x[:, 0, :], x[:, 1, :]

                loss_value = self.test_step(x, y)

                self.test_loss(loss_value)

                val_summary_writer.add_scalar('loss', self.test_loss.result().numpy(), epoch)
                val_summary_writer.add_scalar('accuracy', self.test_accuracy.result().numpy(), epoch)



                # Logging the current accuracy value so far.
                '''if step % 10 == 0:
                    print("Epoch:", epoch, "Step:", step)
                    print("Total running accuracy so far: %.3f" % test_accuracy.result())'''


            # epoch stats
            print("Epoch:", epoch)
            print("Epoch accuracy : %.3f" % self.accuracy.result())
            #print("Epoch val_loss : %.3f" % Trainer.eval_loss(val_dataset, model, loss_fn))
            print("Epoch summary_loss : %.3f" % self.train_loss.result())
            # print("Epoch val_accuracy : %.3f" % model.evaluate(x_val, y_val, batch_size=32))
            # print("Epoch val_accuracy : %.3f" % model.evaluate(test_set))



            # reset accuracy
            self.accuracy.reset_states()
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

        # print("Epoch test_accuracy : %.3f" % Trainer.eval_on_batch(test_dataset, model, size))
        # print("Epoch test_loss : %.3f" % Trainer.eval_loss(test_dataset, model, loss_fn))

        # methode update results
        self.results["loss"] = self.loss_value.numpy()
        self.results["accuracy"] = self.final_acc.numpy()

        Save = False

        if Save:
            from time import ctime
            import os
            path = os.path.join(self.save_path, ctime())
            os.mkdir(path)
            # opposite load weights
            self.model.save_weights(os.path.join(path, ctime()) + ".h5")


    def fit_triplet(self, dataset):
        pass