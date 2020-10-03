import torch
import torch.nn as nn


class MyEnsemble(nn.Module):
    def __init__(self):
        super(MyEnsemble, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1,2)
        )
        make_model = lambda : self.model
        self.modelA = make_model()
        self.modelB = make_model()
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()

        # Create new classifier
        #self.classifier = nn.Linear(2048 + 512, nb_classes)

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        x = self.classifier(F.relu(x))
        return x


inp =  torch.tensor([3.])

ensemble = MyEnsemble()


print(ensemble.modelA(inp))

print(ensemble.modelB(inp))

import sys
sys.exit()

path_today = "/gym_mujoco_planar_snake/prefnet/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble5_Sep_29_21:10:59/model_0"

path_original = "/gym_mujoco_planar_snake/prefnet/results/Mujoco-planar-snake-cars-angle-line-v1/improved_runs/vf_ensemble2_triplet_good_one/model_0"

net = Net(27)
net.load_state_dict(torch.load(path_today))

net_original = Net(27)
net_original.load_state_dict(torch.load(path_original))

inp = torch.ones(27)

print(net.cum_return(inp))

#print(net_original.cum_return(inp))



'''loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)'''

torch.manual_seed(0)

'''input1 = torch.tensor([[3, 4]]).float() #torch.randn(1, 2)
input2 = torch.tensor([[4, 3]]).float()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(input1)
print(input2)
print(output)'''

'''def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

#net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))


linear = nn.Linear(1, 1)
linear.apply(init_weights)


nets = [linear, linear]

inp = torch.tensor([2.])

print([net(inp) for net in nets])'''


#print('Loss_Bce: {:.3f}'.format(loss_bce.item()))


'''torch.manual_seed(0)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)'''


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