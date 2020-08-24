import numpy as np

#path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/test.npy'


#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/1000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Sun Aug 16 10:39:39 2020500000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Mon Aug 17 18:12:38 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/unnormalized.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/ensemble_normalized.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Wed Aug 19 01:05:15 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Wed Aug 19 01:11:56 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Wed Aug 19 01:05:15 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Wed Aug 19 01:11:56 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/hinge_ensemble_unnormalized.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/crossentropy_ensemble_unnormalized.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Fri Aug 21 12:34:51 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Sat Aug 22 13:44:36 20201000000.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/Sun Aug 16 10:39:39 2020500000.npy"

# path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/improved_runs/crossentropy/Sun Aug 23 13:45:24 20201000000.npy"



path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Hopper-v2/Aug 23 21:21:03/trajectories.npy"

with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)



num_episodes = int(d[:,0].size / 1000)

rew = d[:,0].reshape((num_episodes,1000)).sum(axis=1)
pred = d[:,1].reshape((num_episodes,1000)).sum(axis=1)

#rew.sort()

rew = rew[:300]

#rew.sort()
num_episodes  = 300
print(rew.mean(), rew.max())
'''rew = rew[200:300,]
pred = pred[200:300,]
num_episodes = 100'''

import matplotlib.pyplot as plt

indices = np.arange(num_episodes)



plt.plot(indices, rew, color='b')
#äplt.plot(indices, pred, color='r')

plt.show()



#data = np.load(path)"

