import argparse
from time import ctime
import os

from gym_mujoco_planar_snake.common.multi_agents import AgentSquad
from gym_mujoco_planar_snake.common.misc_util import Configs
from gym_mujoco_planar_snake.common.results import DataFrame
from gym_mujoco_planar_snake.common.ensemble import Ensemble

from baselines.common import set_global_seeds

import torch
import numpy as np

def set_seeds(configs):
    seed = configs.get_seed()
    # tensorflow, numpy, random(python)
    set_global_seeds(seed)

    torch.manual_seed(seed)



def create_base_log_dir():
    base_log_dir = os.path.join(configs.get_log_dir(), configs.get_env_id())

    return base_log_dir

def create_new_vf_ensemble(base_log_dir):
    improved_log_dir = os.path.join(base_log_dir, "improved_runs")
    if not os.path.exists(improved_log_dir):
        os.mkdir(improved_log_dir)
    net_save_path = os.path.join(improved_log_dir, "vf_ensemble" + str(configs.get_ranking_approach()) + "_" + ctime()[4:19]).replace(
        " ", "_")

    os.mkdir(net_save_path)
    # TODO write reward to one specific dir
    default_reward = os.path.join(net_save_path, "default_reward")
    os.mkdir(default_reward)

    return net_save_path, default_reward

def create_new_ensemble_dir(base_log_dir):
    base_log_dir = os.path.join(configs.get_log_dir(), configs.get_env_id())

    initial_log_dir = os.path.join(base_log_dir, "initial_runs")

    default_dataset_dir = os.path.join(initial_log_dir, "default_dataset")

    ensemble_dir = os.path.join(initial_log_dir, "ppo_ensemble_"+ctime()[4:19].replace(" ", "_"))

    if not os.path.exists(default_dataset_dir):
        os.mkdir(default_dataset_dir)

    return ensemble_dir

def get_dataset_and_true_rewards_from_default_dir(default_dataset_dir):

    true_rews = []
    trajectories = []
    for file in os.listdir(default_dataset_dir):

            traj_path = os.path.join(default_dataset_dir, file)

            with open(traj_path, 'rb') as f:
                traj = np.load(f, allow_pickle=True)


            true_rews.append(traj[:,1])
            trajectories.append(traj)


    return np.concatenate(trajectories), np.array(true_rews)

def get_true_rewards_and_predictions_from_default_dir(default_reward_dir):

    true_rews = []
    preds = []
    for file in os.listdir(default_reward_dir):

            results_path = os.path.join(default_reward_dir, file)

            with open(results_path, 'rb') as f:
                results = np.load(f, allow_pickle=True)


            true_rews.append(results[:,0])
            preds.append(results[:,1])


    return np.concatenate(true_rews), np.concatenate(preds)






def main():

    set_seeds(configs)

    ########################################################################################
    # Step 1a: if new trajectories should be created
    base_log_dir = create_base_log_dir()

    if configs.get_create_initial_trajectories():

        ensemble_dir = create_new_ensemble_dir(base_log_dir)


        agents = AgentSquad(configs = configs,
                            learned_reward = False,
                            log_dir = ensemble_dir)
        agents.learn()

    ########################################################################################
    # Step 1b: scan default_dataset directory for training data
    default_dataset_dir = os.path.join(base_log_dir, "initial_runs", "default_dataset")

    trajectories, true_initial_rewards = get_dataset_and_true_rewards_from_default_dir(default_dataset_dir)

    # first reproducability, then dataframe, then reward learning

    # store initial to compare them to final result
    initial_dataframe = DataFrame(true_initial_rewards, initial=True)
    print("Initial Mean Reward: %f"%initial_dataframe.get_mean_true_rew())
    print("Initial Max Reward: %f" % initial_dataframe.get_max_true_rew())

    ########################################################################################
    # Step 2: Learn Reward Function
    print("Start IRL")

    net_save_path, default_reward_dir = create_new_vf_ensemble(base_log_dir)
    # create Ensemble mit args

    ensemble = Ensemble(configs, net_save_path)
    ensemble.fit(trajectories)

    ########################################################################################
    # Step 3: Validate IRL through RL

    if configs.get_validate_learned_reward():

        #ensemble_dir = create_new_ensemble_dir(base_log_dir)


        agents = AgentSquad(configs = configs,
                            learned_reward = True,
                            log_dir = net_save_path)
        agents.learn()

        true_rews, preds = get_true_rewards_and_predictions_from_default_dir(default_reward_dir)

        improved_df = DataFrame(true_rew = true_rews, preds = preds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_to_configs', type=str,
                        default="gym_mujoco_planar_snake/agents/configurations/configs.yml")
    args = parser.parse_args()

    configs = Configs(args.path_to_configs)

    # main process
    main()

