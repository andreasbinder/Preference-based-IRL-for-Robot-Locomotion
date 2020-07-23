def train(data, input_shape=(2,50,27)):
    # paper hyperparameters
    # 3 fully connected layers of 256 units with ReLUnonlinearities.
    # Adam with a learning rate of 1e-4
    # minibatch size of 64 for 10,000 timesteps.

    pairs, labels = data
    #model.layers[index].output for output of each layer
    model = Sequential([
        Input(input_shape),
        Dense(256),
        ReLU(),
        Dense(256),
        ReLU(),
        Dense(256),
        ReLU(),
        Flatten(),
        Dense(1), #, activation='sigmoid'
    ], name='reward_net')

    model.summary()

    optimizer = Adam()
    optimizer.apply_gradients()

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.5), metrics=['accuracy'])

    #model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-6), metrics=['accuracy'])

    model.fit(pairs, labels, batch_size=64, epochs=500, shuffle=True)

    #model.save(filepath='/home/andreas/Desktop/model')

    return model

######################################################################################

@staticmethod
def eval_loss(data_set, model, loss_fn):

    total_loss = 0

    for step, (x, y) in enumerate(data_set):
        batch_size = x.shape[0].value

        x = x[:, 0, :], x[:, 1, :]

        predictions, _ = model(x)

        loss_value = loss_fn(y, predictions)

        total_loss += loss_value / batch_size

        # predictions_int = tf.cast(predictions, tf.int64)
        #
        # count = tf.math.equal(predictions_int, y)
        #
        # c += tf.count_nonzero(count)

    return total_loss

######################################################################################

@staticmethod
def eval_on_batch(test_set, model, size):

    total = 0

    c = 0

    for step, (x, y) in enumerate(test_set):
        total += x.shape[0].value

        x = x[:, 0, :], x[:, 1, :]

        predictions, _ = model(x)

        predictions_int = tf.cast(predictions, tf.int64)

        count = tf.math.equal(predictions_int, y)

        c += tf.count_nonzero(count)

    return c.numpy() / total

######################################################################################