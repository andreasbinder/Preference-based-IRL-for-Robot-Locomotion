import numpy as np

path1 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset/trajectories_0Aug_27_16:23:15.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)


path1 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset/trajectories_1Aug_27_16:23:15.npy"

with open(path1, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

pass