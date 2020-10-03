import numpy as np

#path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/test.npy'


#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/1000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/Mon Aug 17 00:02:18 2020test.npy"

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 18:33:08 2020/trajectories.npy"

with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)

pass
print(d.shape)
'''rew = d[:,0].reshape((500,1000)).sum(axis=1)
pred = d[:,1].reshape((500,1000)).sum(axis=1)'''

#d = d[-500:]



rew = d[:, 1]

rew = rew.reshape((300,10)).sum(axis=1)

import matplotlib.pyplot as plt

indices = np.arange(300)

print(d[0].shape)

plt.plot(indices, rew, color='b')
#plt.plot(indices, pred, color='r')

plt.show()



#data = np.load(path)