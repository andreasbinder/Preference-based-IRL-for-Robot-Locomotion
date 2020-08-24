# visualize

import numpy as np

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/'

rews = np.zeros((1, 1000))

for arr in range(1000, 259000, 1000):

    with open(path + str(arr) + ".npy" , 'rb') as f:
        d = np.load(f, allow_pickle=True)
        d = np.expand_dims(d, axis=0)
        rews = np.concatenate((rews, d), axis=0)


'''with open(path + str(2000) + ".npy" , 'rb') as f:
    test = np.load(f, allow_pickle=True)'''

#test = test.sum(axis=0)
rews = rews.sum(axis=1)


import matplotlib.pyplot as plt

indices = list(range(rews.size))
plt.plot(indices, rews)

plt.show()


#data = np.load(path)