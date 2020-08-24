import torch
import numpy as np
from gym_mujoco_planar_snake.common.reward_nets import *
from gym_mujoco_planar_snake.common.misc_util import *

def pred_from_dataset():
    files = get_all_files_from_dir()[-30:]

    # files sort by
    files.sort(key=lambda x: x.cum_reward)

    # model_path = "gym_mujoco_planar_snake/log/PyTorch_Models/Wed Aug 12 21:06:21 2020/model"
    # model_path = "gym_mujoco_planar_snake/log/PyTorch_Models/Wed Aug 12 22:50:36 2020/model"

    #model_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Sun Aug 16 23:57:51 2020/model"

    model_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Mon Aug 17 16:55:25 2020/model"

    # print(len(files))

    net = SingleStepPairNet()
    net.load_state_dict(torch.load(model_path))

    # pred = net.cum_return(torch.from_numpy(np.array(files[0].observations)).float())

    # print(pred)

    for file in files:
        obs = torch.from_numpy(np.array(file.observations)).float()
        true_rew = file.cum_reward
        with torch.no_grad():
            pred, pred_abs = net.cum_return(obs)

        print("True: ", true_rew, "Pred: ", pred, "Pred_Abs: ", pred_abs)


def pred_from_array():

    path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/initial_PPO_runs/trajectories_on_the_fly/test.npy"

    with open(path, 'rb') as f:
        d = np.load(f, allow_pickle=True)



    data = d[500:550]

    #data = np.random.choice(d[], 30)

    #data.reshape((30, 1, 1))

    #assert False, data.shape

    #model_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Sun Aug 16 23:57:51 2020/model"

    #model_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/lastest_one_step_test/model"

    model_path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/log/PyTorch_Models/Mon Aug 17 16:55:25 2020/model"

    # print(len(files))

    net = SingleStepPairNet()
    net.load_state_dict(torch.load(model_path))


    rews = []

    for trajectory in data:
        obs = torch.from_numpy(np.array(trajectory[0])).float()
        true_rew = trajectory[1]
        with torch.no_grad():
            pred, pred_abs = net.cum_return(obs)


        rews.append((true_rew, pred, pred_abs))


    rews.sort(key=lambda x: x[0])

    for true_rew, pred, pred_abs in rews:

        print("True: ", true_rew, "Pred: ", pred, "Pred_Abs: ", pred_abs)




# pred_from_array()

pred_from_dataset()
