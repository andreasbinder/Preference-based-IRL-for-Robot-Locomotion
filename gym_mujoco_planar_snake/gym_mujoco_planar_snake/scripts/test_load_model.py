import torch

from gym_mujoco_planar_snake.common.ensemble import Ensemble

path = "/home/andreas/LRZ_Sync+Share/BachelorThesis/gym_mujoco_planar_snake/gym_mujoco_planar_snake/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble2_Aug_31_12:55:26/"

Ensemble.load(path, 1)

inp = torch.ones(27)