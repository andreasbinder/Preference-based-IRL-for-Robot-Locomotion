import mujoco_py
import gym
import torch
import torch.nn as nn


m = nn.Sigmoid()
loss = nn.BCELoss()


input = torch.randn(3, requires_grad=True)
target = torch.tensor([0., 0.5, 1.])
print(target)
output = loss(m(input), target)
output.backward()
