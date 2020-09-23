import torch
import torch.nn as nn

'''loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)'''

torch.manual_seed(0)

input1 = torch.tensor([[3, 4]]).float() #torch.randn(1, 2)
input2 = torch.tensor([[4, 3]]).float()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(input1)
print(input2)
print(output)

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