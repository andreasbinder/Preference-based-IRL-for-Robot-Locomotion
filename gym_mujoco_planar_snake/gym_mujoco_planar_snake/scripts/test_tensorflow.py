import tensorflow as tf

def variable():
    a = tf.Variable(0)
    b = tf.constant(1)

    mid = tf.add(a, b)

    final = tf.assign(a, mid)

    var = tf.initialize_all_variables()

    with tf.Session() as sess:
        # sess.__enter__()
        sess.run(var)

        for i in range(3):
            sess.run(final)
            print(sess.run(a))

def placeholder():
    x = tf.placeholder("float", shape=[None, 3])
    y = x**2

    with tf.Session() as sess:
        x_data = [[1,2,3],[4,5,6]]

        result = sess.run(y, feed_dict={x : x_data})
        print(type(result))
        print(result)

def constant():
    a =  tf.constant([1,2])
    b = tf.constant([1, 2])

    tf.InteractiveSession()

    c = a + b

    c = c.eval()

    print(c)

def save():
    matrix_1 = tf.Variable([[1, 2], [2, 3]], name="v1")
    matrix_2 = tf.Variable([[3, 4], [5, 6]], name="v2")
    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)
    save_path = saver.save(sess, "model.ckpt")
    sess.close()

def assign():
    a = tf.Variable(0)
    b = tf.constant(1)

    mid = tf.add(a, b)

    final = tf.assign(a, mid)

    var = tf.initialize_all_variables()

    with tf.Session() as sess:
        # sess.__enter__()
        sess.run(var)

        for i in range(3):
            sess.run(final)
            final.eval()
            print(sess.run(a))

def loss():
    import numpy as np

    y = tf.placeholder("float")
    y_hat = tf.placeholder("float")
    l2 = tf.reduce_sum(tf.abs(tf.subtract(y, y_hat)))

    var = tf.Variable(0)

    path ='/home/andreas/Desktop/Checkpoints'

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        y_data = np.arange(5)
        y_hat_data = np.arange(5, 10)
        result = session.run(l2, feed_dict={y:y_data, y_hat:y_hat_data })
        print(result)
        saver.save(session, path+"/model.ckpt", write_meta_graph=False, max_to_keep=0)

        saver.restore(session, path+"/model.ckpt")

def tf_train():
    w = tf.Variable(0)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, "./net.ckpt")

def tf_restore():
    w = tf.Variable(tf.zeros(1))

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, "./net.ckpt")

def tf_group():

    var1 = tf.Variable(5)
    var2 = tf.Variable(5)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.group( var2, var1))
        print(var2)

def tf_keras():
    #import tensorflow.keras as keras
    from tensorflow.keras.layers import Dense
    import tensorflow as tf
    model = Dense(10, input_shape=(3,3))
    print(tf.__version__)

@tf.function
def function():
    print("TF function")


def output():
    from tensorflow import keras
    import tensorflow as tf
    import numpy as np
    m = tf.Variable(0)
    inp = np.arange(-5,5)
    model = keras.Sequential()
    model.add(keras.layers.ReLU())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m = model(inp)
    #print(model.output)
    #final = tf.assign(m, model(inp))
    print(m)
    #var = tf.initialize_all_variables()
    print(m)
    with tf.Session() as sess:
        #sess.run(var)
        sess.run(m)
        m = sess.run(model(inp))
        print(m)

def eager():
    tf.compat.v1.enable_eager_execution()
    var = tf.ones((3,4))
    result = tf.add(var, 1)
    result = result

def cross_entropy():
    tf.compat.v1.enable_eager_execution()
    y_true = [[0., 1.], [0., 0.]]
    y_pred = [[0.6, 0.4], [0.4, 0.6]]
    #y_pred = [1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2]
    #y_true = [2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1]
    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = tf.keras.losses.BinaryCrossentropy()
    res = bce(y_true, y_pred).numpy()
    rest = res



if __name__ == "__main__":
    cross_entropy()
