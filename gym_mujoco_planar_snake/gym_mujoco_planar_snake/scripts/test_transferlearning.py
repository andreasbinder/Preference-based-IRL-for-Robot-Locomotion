import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow import keras
from gym_mujoco_planar_snake.common.reward_net import SimpleNet
from gym_mujoco_planar_snake.agents.run_reward_learning import get_data_from_file
from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory


#/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/

file = '/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/TrajectoryDataset'
name = 'Dataset500_Sun Jul 12 17:21:36 2020'
files = get_data_from_file(file, name)

file1 = files[0]

import numpy as np


obs1 = np.array(file1.observations).reshape((1350,))

obs1 = tf.convert_to_tensor(obs1)

obs1 = tf.expand_dims(obs1, axis=0)

file2 = files[1]


def change_model(model, new_input_shape=(None, 27)):
    # replace input shape of first layer
    model._layers[1].batch_input_shape = new_input_shape

    # feel free to modify additional parameters of other layers, for example...


    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    X = np.random.rand(10, 27)
    y_pred = new_model.predict(X, )
    print(y_pred)

    return new_model




obs2 = np.array(file2.observations).reshape((1350,))

obs2 = tf.convert_to_tensor(obs2)

obs2 = tf.expand_dims(obs2, axis=0)

#path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Tue Jul 21 12:47:05 2020/Tue Jul 21 12:47:05 2020'

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Sun Jul 26 20:36:01 2020/Sun Jul 26 20:36:01 2020.h5'

#model = keras.Model()

model = SimpleNet()

model.load_weights(path)

#model._layers[1].batch_input_shape = (None, 27)

#model = change_model(model)

#first_layer = keras.layers.InputLayer(input_shape=(27, ))

#model.layers[1].batch_input_shape = (None, 27)

dense = keras.layers.Dense(1350, input_shape=(27, ))

#inp = tf.random.normal([1,1350])

inp = tf.random.normal([1,27])

#inp = first_layer(inp)

inp = dense(inp)

res1, res_abs = model.reward(inp)

print(res1)
print(res_abs)


'''
res1, _ = model.reward(obs1)

res2, _ = model.reward(obs2)

x, _ = model((obs1, obs2))

test = res1.numpy()'''



