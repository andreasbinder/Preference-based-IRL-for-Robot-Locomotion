import numpy as np

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/6000.npy'

with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)

print(d.shape)

#data = np.load(path)