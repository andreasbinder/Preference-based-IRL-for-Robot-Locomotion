import torch
import torch.nn as nn

'''loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)'''


model = nn.Linear(20, 3) # predict logits for 5 classes
x = torch.randn(2, 20)
y = torch.tensor([[1., 0., 1.],
                  [1., 0., 1.]]) # get classA and classC as active
print(x)
print(y)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print('Loss: {:.3f}'.format(loss.item()))


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