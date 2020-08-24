import numpy as np

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Hopper-v2/Aug 23 19:29:24/trajectories.npy"


with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)

pass