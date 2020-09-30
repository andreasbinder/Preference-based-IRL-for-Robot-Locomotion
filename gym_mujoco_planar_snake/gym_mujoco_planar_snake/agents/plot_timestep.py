import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from gym_mujoco_planar_snake.common.ensemble import Net
from gym_mujoco_planar_snake.common.misc_util import merge_saved_numpy_arrays

log_dir = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"
train_data = merge_saved_numpy_arrays(log_dir, None)

log_dir = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/test"
test_data = merge_saved_numpy_arrays(log_dir, None)


rewards = train_data


'''obs, rew = rewards[:, 0], rewards[:, 1]

sort = np.argsort(rew)

rewards = rewards[sort]'''

import torch

torch.manual_seed(0)

triplet_path = "/home/andreas/Documents/pbirl-bachelorthesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble2_triplet_good_one/model_0"


triplet_net = Net(27)
triplet_net.load_state_dict(torch.load(triplet_path))

#print(rewards[:, 0].shape)

#assert False, "Test"
len1 = rewards.shape[0]
len2 = test_data.shape[0]

obs = np.array([rewards[index, 0] for index in range(len1)]) #19900
rews = np.array([rewards[index, 2] for index in range(len1)]) #1

obs2 = np.array([test_data[index, 0] for index in range(len2)]) #19900
rews2 = np.array([test_data[index, 2] for index in range(len2)]) #1

#print(obs[0].shape)

#assert False, "Test"
starts = np.random.randint(0, len1, size=200) # 500
starts.sort()

starts2 = np.random.randint(0, len2, size=200) # 500
starts2.sort()
'''obs = obs[-500:]
rews = rews[-500:]
'''

obs = np.array([obs[start] for start in starts])
rews = np.array([rews[start] for start in starts])

obs2 = np.array([obs2[start] for start in starts2])
rews2 = np.array([rews2[start] for start in starts2])

inps = torch.from_numpy(obs).float()

inps2 = torch.from_numpy(obs2).float()

predictions = np.array([triplet_net.cum_return(inp) for inp in inps])
predictions2 = np.array([triplet_net.cum_return(inp) for inp in inps2])

print(predictions.shape)

print(rews)
print(predictions)


#plt.plot(rews, predictions)
plt.scatter(rews, predictions*20, color='b')
plt.scatter(rews2, predictions2*20, color="r")

plt.show()




