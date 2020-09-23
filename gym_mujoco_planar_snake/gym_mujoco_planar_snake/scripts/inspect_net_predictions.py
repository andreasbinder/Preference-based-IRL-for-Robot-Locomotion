import torch

from gym_mujoco_planar_snake.common.ensemble import Ensemble, Net

path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble5_Sep_13_16:04:26/model_0"

#Ensemble.load(path, 1)

inp = torch.ones(27)

net = Net(27)
net.load_state_dict(torch.load(path))

traj_path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset/trajectories_335000.npy"
import numpy as np



with open(traj_path, 'rb') as f:
    d2 = np.load(f, allow_pickle=True)

traj_1 = d2[0,0]
traj_1 = torch.from_numpy(traj_1).float()
rew = net.model(traj_1)

import matplotlib.pyplot as plt

indices = np.arange(100)



plt.plot(indices, rew.view(100, ).detach().numpy(), color='b')
#plt.plot(indices, pred, color='r')

plt.show()

print(rew)
print(rew.sum())