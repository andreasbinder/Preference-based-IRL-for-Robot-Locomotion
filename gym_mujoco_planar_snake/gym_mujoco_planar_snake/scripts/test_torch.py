import torch
import torch.nn as nn

#from gym_mujoco_planar_snake.common.trainer import Net

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

loss = nn.MarginRankingLoss()
'''input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()'''
input1 = torch.tensor([1.,1.,1.], requires_grad=True)
input2 = torch.tensor([1.1,0.,0.], requires_grad=True)
target = torch.tensor([0,1,1])

print(input1, input2, target)
output = loss(input1, input2, target)
print(output)
output.backward()

'''from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()'''

'''linear = nn.Linear(27, 1)

anchor = torch.randn(100, 27, requires_grad=True)

print(anchor.shape)
out = linear(anchor)
print(out.shape)

net = nn.Sequential(
    nn.Linear(27, 1)
)

out = net(anchor)
print(out.shape)'''


'''m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
print(input)
print(m(input))
print(target)
output = loss(m(input), target)
print(output)
output.backward()'''
'''triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(100, 128, requires_grad=True)
print(anchor.shape)
positive = torch.randn(100, 128, requires_grad=True)
print(positive.shape)
negative = torch.randn(100, 128, requires_grad=True)
print(negative.shape)
output = triplet_loss(anchor, positive, negative)
print(output)
output.backward()'''