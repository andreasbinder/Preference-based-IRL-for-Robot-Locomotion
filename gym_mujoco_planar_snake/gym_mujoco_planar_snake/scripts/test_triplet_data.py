import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import os

# take before TREX trajectories and put results into after TREX traj
from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
from gym_mujoco_planar_snake.common.iterator import Iterator
from gym_mujoco_planar_snake.common.trainer import Trainer
from gym_mujoco_planar_snake.common.documentation import to_excel
from gym_mujoco_planar_snake.common.performance_checker import reward_prediction
import tensorflow.keras as keras


#######################################################################


# load data
def get_data_from_file(path, name="Dataset"):
    return Dataset.load(path=path, name=name)


def get_all_files_from_dir():
    import os
    from itertools import chain
    path = '/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/TrajectoryDataset'

    files = os.listdir(path)
    print("Source Files: ", files)

    files = [get_data_from_file(path=path, name=name) for name in files if hasattr(get_data_from_file(path=path, name=name)[0], "time_step") ]
    result = list(chain.from_iterable(files))
    print("Total number of files", len(result))

    # assertion here
    assert all([isinstance(i, SubTrajectory) for i in result]), "Not all elements from expected source type"

    return result

data = get_all_files_from_dir()
data = Dataset(data)
#data = Dataset.load("/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/",
#                    name="Dataset1500_Wed Jul 29 13:01:44 2020")

#pass

print("Trajectories: ", str(len(data)))

import sys
sys.exit(0)

triplets = []

counter = 0

while True:
    print(counter)
    anchor = data.sample(1)[0]

    trajs = data.sample(2)

    cum_reward_i = abs(anchor.cum_reward - trajs[0].cum_reward)
    time_step_i = abs(anchor.time_step - trajs[0].time_step)

    cum_reward_j = abs(anchor.cum_reward - trajs[1].cum_reward)
    time_step_j = abs(anchor.time_step - trajs[1].time_step)


    while True:
        if cum_reward_i < cum_reward_j and time_step_i < time_step_j:
            triplets.append([anchor, trajs[0], trajs[1]])
            data.remove(anchor)
            data.remove(trajs[0])
            data.remove(trajs[1])

            break
        elif cum_reward_i > cum_reward_j and time_step_i > time_step_j:
            triplets.append([anchor, trajs[1], trajs[0]])
            data.remove(anchor)
            data.remove(trajs[0])
            data.remove(trajs[1])
            break
        else:
            trajs = data.sample(2)
            cum_reward_i = abs(anchor.cum_reward - trajs[0].cum_reward)
            time_step_i = abs(anchor.time_step - trajs[0].time_step)

            cum_reward_j = abs(anchor.cum_reward - trajs[1].cum_reward)
            time_step_j = abs(anchor.time_step - trajs[1].time_step)

    if len(data) < 10:
        break

    counter += 1

    if counter > 10000:
        break


import numpy as np
import os
data = np.array(triplets)

print("Triplets: ", str(data.shape[0]))

path = '/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/triplets/'



with open(path+'test.npy', 'wb') as f:

    np.save(f, data)













# maybe constraint satisfaction for triplets
