from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
from gym_mujoco_planar_snake.common.misc_util import get_all_files_from_dir

import numpy as np

results = get_all_files_from_dir(None)

list_rewards = [ i.cum_reward for i in results]

list_episodes = [i for i in range(len(list_rewards))]

import matplotlib.pyplot as plt

plt.plot(list_episodes, list_rewards, color='green')

print(np.array(list_rewards).mean())
print(np.array(list_rewards).min())
print(np.array(list_rewards).max())

