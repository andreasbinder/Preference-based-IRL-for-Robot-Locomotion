import tensorflow as tf

#tf.compat.v1.enable_eager_execution()
from tensorflow import keras
#from gym_mujoco_planar_snake.common.reward_net import SimpleNet
#from gym_mujoco_planar_snake.agents.run_reward_learning import get_data_from_file
from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
import numpy as np
import torch
from baselines.common import set_global_seeds
import gym


def evaluate_policy(env, pi, max_timesteps=5000, custom=True, render=True, seed=0):

    env = gym.make(env)

    number_of_timestep = 0
    done = False


    # set seed
    set_global_seeds(seed)
    env.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset()

    rewards = []

    while (not done) and number_of_timestep < max_timesteps:



        action, v_pred = pi.act( True, obs)

        #action = action[0]  # TODO check

        # print(action)



        obs, reward, done, info = env.step(action)

        rewards.append(reward)


        # render
        if render:
            env.render()

        number_of_timestep += 1

    import os
    rewards = np.array(rewards)
    dir = "/home/andreas/Desktop"
    name = 'custom.npy' if custom else 'original.npy'

    with open(os.path.join(dir, name), 'wb') as f:
        np.save(f, rewards)





def reward_prediction(model, trajectory):

    obs = tf.expand_dims(tf.convert_to_tensor(np.array(trajectory.observations).reshape((1350,)).astype("float32")), axis=0)

    reward, abs_reward = model.reward(obs)

    print("True reward: ", str(trajectory.cum_reward), "Predicted reward: ", str(reward.numpy()), "Predicted  absolute reward: ",
          str(abs_reward.numpy()))

    print()




def test():

    # /home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/

    file = '/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/TrajectoryDataset'
    name = 'Dataset500_Sun Jul 12 17:21:36 2020'
    files = get_data_from_file(file, name)

    file1 = files[0]

    import numpy as np

    obs1 = np.array(file1.observations).reshape((1350,))

    obs1 = tf.convert_to_tensor(obs1)

    obs1 = tf.expand_dims(obs1, axis=0)

    file2 = files[1]

    obs2 = np.array(file2.observations).reshape((1350,))

    obs2 = tf.convert_to_tensor(obs2)

    obs2 = tf.expand_dims(obs2, axis=0)

    # path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Tue Jul 21 12:47:05 2020/Tue Jul 21 12:47:05 2020'

    path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/models/Wed Jul 22 17:37:45 2020/Wed Jul 22 17:37:45 2020.h5'

    # model = keras.Model()

    model = SimpleNet()

    model.load_weights(path)

    inp = tf.random.normal([1, 1350])

    res1, _ = model.reward(obs1)

    res2, _ = model.reward(obs2)

    x, _ = model((obs1, obs2))

    test = res1.numpy()





