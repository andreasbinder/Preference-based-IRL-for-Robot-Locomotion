import numpy as np

path1 = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_original_1.5Mio/data_500k/train.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)


path1 = "/home/andreas/Desktop/create_test2/trajectories_970000.npy"

with open(path1, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

pass