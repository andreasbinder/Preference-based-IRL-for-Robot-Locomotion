import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from tensorflow import keras

import numpy as np

from gym_mujoco_planar_snake.common.reward_net import RewardNet, SimpleNet


class Trainer:

    def __init__(self, hparams,
                 save_path='/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/'):
        self.hparams = hparams
        self.save_path = save_path
        self.results = {
            "loss": 0.,
            "accuracy": 0.,
            "test_loss": 0.,
            "test_accuracy": 0.
        }
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.accuracy = None
        self.train_loss = None

        # initialize validation metrics
        self.validation_loss = None
        self.validation_accuracy = None

        # initialize test metrics
        self.test_loss = None
        self.test_accuracy = None

        self.train_summary_writer = None
        self.val_summary_writer = None
        self.test_summary_writer = None

        # documentation
        self.use_tensorboard = False

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def initialize_tensorboard(self):

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            import datetime

            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'gym_mujoco_planar_snake/log/tensorboard/gradient_tape/' + current_time + '/train'
            val_log_dir = 'gym_mujoco_planar_snake/log/tensorboard/gradient_tape/' + current_time + '/val'
            test_log_dir = 'gym_mujoco_planar_snake/log/tensorboard/gradient_tape/' + current_time + '/test'

            self.train_summary_writer = SummaryWriter(train_log_dir)
            self.val_summary_writer = SummaryWriter(val_log_dir)
            self.test_summary_writer = SummaryWriter(test_log_dir)

    @tf.function
    def add_to_tensorboard(self, mode, epoch):

        if self.use_tensorboard:

            if mode == 'train':
                self.train_summary_writer.add_scalar('train_loss', self.train_loss.result().numpy(), epoch)
                self.train_summary_writer.add_scalar('train_accuracy', self.accuracy.result().numpy(), epoch)
            elif mode == 'val':
                self.val_summary_writer.add_scalar('val_loss', self.validation_loss.result().numpy(), epoch)
                self.val_summary_writer.add_scalar('val_accuracy', self.validation_accuracy.result().numpy(), epoch)
            elif mode == 'test':
                self.test_summary_writer.add_scalar('test_loss', self.test_loss.result().numpy(), 0)
                self.test_summary_writer.add_scalar('test_accuracy', self.test_accuracy.result().numpy(), 0)

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
            self.accuracy.update_state(y, logits)

            # Update the weights of the model to minimize the loss value.
            gradients = tape.gradient(loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss_value

    @tf.function
    def val_step(self, x_test, y_test):
        predictions, _ = self.model(x_test)

        loss = self.loss_fn(y_test, predictions)

        self.validation_accuracy.update_state(y_test, predictions)
        # test_loss(loss)
        # test_accuracy(y_test, predictions)

        return loss

    @tf.function
    def test_step(self, x_test, y_test):
        predictions, _ = self.model(x_test)

        loss = self.loss_fn(y_test, predictions)

        self.test_accuracy.update_state(y_test, predictions)
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
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.accuracy = tf.keras.metrics.BinaryAccuracy('train_accuracy')

        # initialize validation metrics
        self.validation_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.BinaryAccuracy('val_accuracy')

        # initialize test metrics
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy('test_accuracy')

        # initialize tensorboard
        self.initialize_tensorboard()

        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.model = SimpleNet()

        # model.compile(optimizer, loss_fn, [accuracy])

        final_train_accuracy = None
        final_train_loss = None

        for epoch in range(epochs):

            # training
            print("Start Training")
            for step, (x, y) in enumerate(train_dataset):
                x = x[:, 0, :], x[:, 1, :]

                loss_value = self.train_on_batch(x, y)

                self.train_loss(loss_value)

                self.add_to_tensorboard('train', epoch)

            # validation
            print("Start Validating")
            for step, (x, y) in enumerate(val_dataset):
                x = x[:, 0, :], x[:, 1, :]

                loss_value = self.val_step(x, y)

                self.validation_loss(loss_value)

                self.add_to_tensorboard('val', epoch)

            # print results
            template = 'Epoch {}, Loss: {:10.4f}, Accuracy: {:10.4f}, Val Loss: {:10.4f}, Val Accuracy: {:10.4f}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.accuracy.result(),
                                  self.validation_loss.result(),
                                  self.validation_accuracy.result()))

            final_train_accuracy = self.accuracy.result().numpy()
            final_train_loss = self.train_loss.result().numpy()

            # reset accuracy
            self.accuracy.reset_states()
            self.train_loss.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()

        # testing
        for step, (x, y) in enumerate(test_dataset):
            x = x[:, 0, :], x[:, 1, :]

            loss_value = self.test_step(x, y)

            self.test_loss(loss_value)

            self.add_to_tensorboard('test', None)

        print("Final Test Loss: ", str(self.test_loss.result().numpy()), " Test Accuracy: ",
              str(self.test_accuracy.result().numpy()))

        # print("Epoch test_accuracy : %.3f" % Trainer.eval_on_batch(test_dataset, model, size))
        # print("Epoch test_loss : %.3f" % Trainer.eval_loss(test_dataset, model, loss_fn))

        # TODO excel update
        # methode update results
        self.results["loss"] = final_train_loss
        self.results["accuracy"] = final_train_accuracy
        self.results["test_loss"] = self.test_loss.result().numpy()
        self.results["test_accuracy"] = self.test_accuracy.result().numpy()

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
