import numpy as np

path1 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble2_triplet_good_one/default_reward/results1Sep__4_11:01:43.npy"

with open(path1, 'rb') as f:
    d1 = np.load(f, allow_pickle=True)


path1 = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/ppo_ensemble_1_1Mio_good_run/agent_0/trajectories_0Aug_31_09:27:15.npy"

with open(path1, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

pass