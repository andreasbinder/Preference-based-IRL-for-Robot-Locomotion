import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow import keras
from gym_mujoco_planar_snake.common.reward_net import SimpleNet
from gym_mujoco_planar_snake.agents.run_reward_learning import get_data_from_file
from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory


#/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/

file = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset'
name = 'Dataset500_Sun Jul 12 17:21:36 2020'
files = get_data_from_file(file, name)

file = files[0]

import numpy as np


obs = np.array(file.observations).reshape((1350,))

obs = tf.convert_to_tensor(obs)

obs = tf.expand_dims(obs, axis=0)

#path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Tue Jul 21 12:47:05 2020/Tue Jul 21 12:47:05 2020'

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Tue Jul 21 14:22:38 2020/Tue Jul 21 14:22:38 2020.h5'

#model = keras.Model()

model = SimpleNet()

model.load_weights(path)

inp = tf.random.normal([1,1350])

res, _ = model.reward(obs)

test = res.numpy()

file.cum_reward

