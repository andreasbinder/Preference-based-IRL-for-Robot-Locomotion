from gym_mujoco_planar_snake.common.dataset import Dataset, SubTrajectory
import pickle

data = Dataset.load('/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/SubTrajectoryDataset/','Dataset500_Sun Jul 12 17:21:36 2020')

d = data[0]

dic = {
    "cum_reward": d.cum_reward,
    "observations": d.observations
}

with open("dict_test", 'wb') as f:
    pickle.dump(dic, f)

with open("subtrajectory_test", 'wb') as f:
    pickle.dump(d, f)
