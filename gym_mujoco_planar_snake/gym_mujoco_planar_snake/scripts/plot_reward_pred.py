import torch

from gym_mujoco_planar_snake.common.ensemble import Ensemble, Net

path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble5_epochs25/model_0"

net = Net(27)
net.load_state_dict(torch.load(path))


import numpy as np
import os

data_path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"

trajs = os.listdir(data_path)
subset = np.random.choice(trajs, 100)

inp = []

for traj in subset:

    with open(os.path.join(data_path,traj), 'rb') as f:
        data = np.load(f, allow_pickle=True)

    np.random.shuffle(data)

    inp.append(data[:10])

inp = np.concatenate(inp)

#print(inp.shape)

#print(np.array(list(zip(*inp))).shape)

trajs, timesteps = list(zip(*inp))


#print(trajs)

ind = np.argsort(timesteps)

trajs = np.array(trajs)[ind]

trajs = trajs[:,0,:]

#print(zip(*inp))

'''import sys
sys.exit()'''

# inp = np.array(inp)

#inp.sort(key=lambda x: x[:, 1])



#print([inp[0, 1], inp[1, 1], inp[2, 1]])




traj_1 = torch.tensor(trajs).float()

traj_1.view(1000, 27)

#print(traj_1.shape)

#print()


rew = net.model(traj_1)

import matplotlib.pyplot as plt

indices = range(1000)

plt.plot(indices, rew.detach().numpy())

plt.show()

