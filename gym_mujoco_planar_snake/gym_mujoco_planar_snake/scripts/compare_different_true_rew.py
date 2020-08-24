import numpy as np

# hinge
path1 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/hinge_ensemble_unnormalized.npy"

# cross
path2 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/improved_PPO_runs/Test_True_Reward/crossentropy_ensemble_unnormalized.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)

with open(path2, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

#num_episodes = int(d1[:,0].size / 1000)

#rew = d[:,0].reshape((num_episodes,1000)).sum(axis=1)

#rew = d[:,0].reshape((num_episodes,1000)).sum(axis=1)



rew1 = d1[:, 0].sum()
rew2 = d2[:, 0].sum()
print(int(d1[:,0].size / 1000), int(d2[:,0].size / 1000))
print(rew1, rew2)

