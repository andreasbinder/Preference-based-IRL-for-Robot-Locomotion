import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from tensorflow import keras
from gym_mujoco_planar_snake.common.reward_net import RewardNet, SimpleNet




class Trainer:

    def __init__(self, hparams,  save_path='/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/', execute_eagerly=True):
        self.hparams = hparams
        self.save_path = save_path
        self.results = {
            "loss": 0.,
            "accuracy": 0.
        }

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value


    def fit_pair(self, dataset):

        x_train, y_train = dataset

        '''dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype("float32") , y_train)
        )
        dataset = dataset.shuffle(buffer_size=1024).batch(32)'''

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], -1))
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        dataset = x_train, y_train

        model = RewardNet()

        # TODO welche loss function sollte man verwenden
        # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        for epoch in range(2):
            # Iterate over the batches of a dataset.
            # for step, (x, y) in enumerate(dataset):
            for step in range(250):
                x = tf.expand_dims(x_train[step], 0)
                y = y_train[step]

                # logits = tf.Variable(0, trainable=True)
                test = 0
                # x = tf.Variable(x,trainable=True)
                # x = tf.expand_dims(tf.Variable(x[0]), 0)
                y = tf.subtract(y, 1)

                x = x[:, 0, :], x[:, 1, :]

                with tf.GradientTape() as tape:
                    # TODO make tf watch my variables for computing gradients
                    tape.watch(y)
                    tape.watch(x)

                    # tape.watch(logits)

                    # model.build((1350,))

                    logits, regularization_constant = model(x)

                    # gradient test
                    # logits = net(x)

                    model.summary()

                    # TODO transform
                    # I somehow thought to use label 1 and 2
                    # logits = tf.add(tf.squeeze(tf_argmax(logits, axis=1)), 1)

                    '''                    logits = tf.squeeze(tf_argmax(logits, axis=1))
                    logits = tf.dtypes.cast(logits, tf.float32)'''

                    y = tf.expand_dims(y, 0)
                    logits = tf.expand_dims(logits, 0)

                    # Compute the loss value for this batch.

                    loss_value = loss_fn(y, logits)
                    # loss_value = tf.nn.softmax_cross_entropy_with_logits_v2(y, logits)

                    w_var = tape.watched_variables()

                    # Update the weights of the model to minimize the loss value.
                    gradients = tape.gradient(loss_value, model.trainable_weights)
                    # gradients = tape.gradient(loss_value, model.inputs)

                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                # Logging the current accuracy value so far.
                if step % 200 == 0:
                    print("Epoch:", epoch, "Step:", step)



    def fit_pair_v4(self, dataset):
        batch_size = self.hparams["batch_size"]
        lr = self.hparams["lr"]
        epochs = self.hparams["epochs"]



        x_train, y_train = dataset

        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype("float32"), y_train)
        )
        dataset = dataset.shuffle(buffer_size=2048).batch(batch_size)

        model = SimpleNet()

        accuracy = tf.keras.metrics.BinaryAccuracy()
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        @tf.function  # Make it fast.
        def train_on_batch(x, y):
            with tf.GradientTape() as tape:
                # tell gradient tape to watch variables
                # tape.watch(y)
                # tape.watch(x)

                logits, regularization_constant = model(x)

                # Compute the loss value for this batch.
                loss_value = loss_fn(y, logits)

                # Update accuracy
                accuracy.update_state(y, logits)

                # Update the weights of the model to minimize the loss value.
                gradients = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            return loss_value

        for epoch in range(epochs):

            # Iterate over the batches of a dataset.
            for step, (x, y) in enumerate(dataset):

                x = x[:, 0, :], x[:, 1, :]

                # with tf.GradientTape() as tape:
                #     # tell gradient tape to watch variables
                #     #tape.watch(y)
                #     #tape.watch(x)
                #
                #     logits, regularization_constant = model(x)
                #
                #     # Compute the loss value for this batch.
                #     loss_value = loss_fn(y, logits)
                #
                #     # Update accuracy
                #     accuracy.update_state(y, logits)
                #
                #
                #     # Update the weights of the model to minimize the loss value.
                #     gradients = tape.gradient(loss_value, model.trainable_weights)
                #
                # optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                loss_value = train_on_batch(x,y)

                # final accuracy
                final_acc = accuracy.result()

                # Logging the current accuracy value so far.
                if step % 10 == 0:
                    print("Epoch:", epoch, "Step:", step)
                    print("Total running accuracy so far: %.3f" % accuracy.result())


            # reset accuracy
            accuracy.reset_states()

        # methode update results
        self.results["loss"] = loss_value.numpy()
        self.results["accuracy"] = final_acc.numpy()


        from time import ctime
        import os
        path = os.path.join(self.save_path, ctime())
        os.mkdir(path)
        # opposite load weights
        model.save_weights(os.path.join(path, ctime())+".h5")


    def fit_triplet(self, dataset):
        batch_size = self.hparams["batch_size"]
        lr = self.hparams["lr"]
        epochs = self.hparams["epochs"]



        x_train, y_train = dataset

        dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(x_train.shape[0], x_train.shape[1], -1).astype("float32"), y_train)
        )
        dataset = dataset.shuffle(buffer_size=2048).batch(batch_size)

        model = SimpleNet()

        accuracy = tf.keras.metrics.BinaryAccuracy()
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        @tf.function  # Make it fast.
        def train_on_batch(x, y):
            with tf.GradientTape() as tape:
                # tell gradient tape to watch variables
                # tape.watch(y)
                # tape.watch(x)

                logits, regularization_constant = model(x)

                # Compute the loss value for this batch.
                loss_value = loss_fn(y, logits)

                # Update accuracy
                accuracy.update_state(y, logits)

                # Update the weights of the model to minimize the loss value.
                gradients = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            return loss_value

        for epoch in range(epochs):

            # Iterate over the batches of a dataset.
            for step, (x, y) in enumerate(dataset):

                x = x[:, 0, :], x[:, 1, :]

                # with tf.GradientTape() as tape:
                #     # tell gradient tape to watch variables
                #     #tape.watch(y)
                #     #tape.watch(x)
                #
                #     logits, regularization_constant = model(x)
                #
                #     # Compute the loss value for this batch.
                #     loss_value = loss_fn(y, logits)
                #
                #     # Update accuracy
                #     accuracy.update_state(y, logits)
                #
                #
                #     # Update the weights of the model to minimize the loss value.
                #     gradients = tape.gradient(loss_value, model.trainable_weights)
                #
                # optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                loss_value = train_on_batch(x,y)

                # final accuracy
                final_acc = accuracy.result()

                # Logging the current accuracy value so far.
                if step % 10 == 0:
                    print("Epoch:", epoch, "Step:", step)
                    print("Total running accuracy so far: %.3f" % accuracy.result())


            # reset accuracy
            accuracy.reset_states()

        # methode update results
        self.results["loss"] = loss_value.numpy()
        self.results["accuracy"] = final_acc.numpy()


        from time import ctime
        import os
        path = os.path.join(self.save_path, ctime())
        os.mkdir(path)
        # opposite load weights
        model.save_weights(os.path.join(path, ctime()))


