import torch
import numpy as np
from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.misc_util import *

files = get_all_files_from_dir(None)[30:60]

# files sort by
files.sort(key=lambda x: x.cum_reward)

model_path = "gym_mujoco_planar_snake/log/PyTorch_Models/Wed Aug 12 21:06:21 2020/model"

#print(len(files))

net = SingleStepPairNet()
net.load_state_dict(torch.load(model_path))

#pred = net.cum_return(torch.from_numpy(np.array(files[0].observations)).float())

#print(pred)

for file in files:
    obs = torch.from_numpy(np.array(file.observations)).float()
    true_rew = file.cum_reward
    with torch.no_grad():
        pred, pred_abs = net.cum_return(obs)

    print("True: ", true_rew, "Pred: ", pred, "Pred_Abs: ", pred_abs)
