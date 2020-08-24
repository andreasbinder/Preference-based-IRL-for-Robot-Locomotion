from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
from gym_mujoco_planar_snake.common.misc_util import get_all_files_from_dir

import numpy as np

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/Sat Aug 22 19:13:53 2020/trajectories.npy"

#path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/Mon Aug 17 00:02:18 2020test.npy"

with open(path, 'rb') as f:
    d = np.load(f, allow_pickle=True)

trajectories = []

for obs, cum_reward in d:
    trajectories.append(SubTrajectory(cum_reward=cum_reward, observations=obs, time_step=None))

dataset = Dataset(trajectories)

save_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/initial_runs/default_dataset/"

dataset.save(save_path, name="t5")
