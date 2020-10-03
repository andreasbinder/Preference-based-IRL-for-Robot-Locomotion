import numpy as np
import matplotlib.pyplot as plt
import os

data_path = "/gym_mujoco_planar_snake/prefnet/results/Mujoco-planar-snake-cars-angle-line-v1/initial_runs/default_dataset"

paths = os.listdir(data_path)

paths.sort()

#print(paths)

#assert False, "Test"

rewards = []

for path in paths:

    with open(os.path.join(data_path,path), 'rb') as f:
        data = np.load(f, allow_pickle=True)

    #rewards.append(data[0][1])
    rewards.append(data)

rewards = np.concatenate(rewards)

obs, rew = rewards[:, 0], rewards[:, 1]

sort = np.argsort(rew)

rewards = rewards[sort]



print(obs.shape)
indices = range(len(rewards[:,1]))

#rewards.sort()

plt.plot(indices, rewards[:, 1])

plt.show()


with open(os.path.join(data_path, "train.npy"), 'wb') as f:
    np.save(f, np.array(rewards[:10000]))

with open(os.path.join(data_path, "test.npy"), 'wb') as f:
    np.save(f, np.array(rewards[10000:]))



