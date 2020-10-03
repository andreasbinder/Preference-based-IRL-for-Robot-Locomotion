class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=10):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()

        # Create new classifier
        self.classifier = nn.Linear(2048 + 512, nb_classes)

    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        x = self.classifier(F.relu(x))
        return x


# Train your separate models
# ...
# We use pretrained torchvision models here
modelA = models.resnet50(pretrained=True)
modelB = models.resnet18(pretrained=True)

# Freeze these models
for param in modelA.parameters():
    param.requires_grad_(False)

for param in modelB.parameters():
    param.requires_grad_(False)

# Create ensemble model
model = MyEnsemble(modelA, modelB)
x = torch.randn(1, 3, 224, 224)
output = model(x)