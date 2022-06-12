import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import pickle
from define_model import ConvNet

# Meta-parameters
epochs = 5
batch_size = 16
learning_rate = 0.001

# Loading data
train_data = datasets.ImageFolder("../ml/train/", transform=ToTensor())
test_data = datasets.ImageFolder("../ml/test/", transform=ToTensor())
print(train_data)
print(test_data)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
# with open("idx_to_class", "wb") as file:
#     pickle.dump(idx_to_class, file)

model = ConvNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print("Finished Training")

torch.save(model.state_dict(), './model.pth')

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')