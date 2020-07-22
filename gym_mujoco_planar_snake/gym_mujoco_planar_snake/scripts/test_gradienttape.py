import tensorflow as tf

tf.compat.v1.enable_eager_execution()

a = tf.keras.layers.Dense(32)
layer = tf.keras.layers.ReLU()

'''net = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1)
])'''

from tensorflow.keras.layers import Dense
import tensorflow as tf

class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__()

        self.dense3 = Dense(256)
        self.dense4 = Dense(1)

    def reward(self, x):
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def call(self, x, **kwargs):
        x1 = self.reward(x)
        x2 = self.reward(x)

        return tf.add(x1,x2)


net = Net()

# x = tf.constant(3.0)
x_old = tf.constant([-3.0, -1.0, 1.0, 3.0])
x = tf.expand_dims(x_old, 0)

y = tf.constant([-3.0, -1.0, 5.0, 2.0])

with tf.GradientTape() as g:
    #g.watch(x)

    logit = net(x)

    loss = tf.keras.losses.MSE(logit, y)

dy_dx = g.gradient(loss, net.trainable_weights)  # Will compute to 6.0

x = 3

'''opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
loss_fn = lambda: tf.keras.losses.mse(model(input), output)
var_list_fn = lambda: model.trainable_weights
for input, output in data:
  opt.minimize(loss_fn, var_list_fn)'''

# a.build()
