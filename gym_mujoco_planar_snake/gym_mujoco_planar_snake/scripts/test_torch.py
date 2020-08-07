import torch
import torch.nn as nn

from gym_mujoco_planar_snake.common.trainer import Net

'''loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
print(input.shape)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
print(target.shape)
output = loss(input, target)
output.backward()'''


triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(100, 128, requires_grad=True)
print(anchor.shape)
positive = torch.randn(100, 128, requires_grad=True)
print(positive.shape)
negative = torch.randn(100, 128, requires_grad=True)
print(negative.shape)
output = triplet_loss(anchor, positive, negative)
print(output)
output.backward()