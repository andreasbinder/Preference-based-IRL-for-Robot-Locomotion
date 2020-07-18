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

