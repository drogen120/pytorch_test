# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.block_1 = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=3, stride=2, padding=2),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(96)
            )
        self.block_2 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(32)
            )
        self.block_3 = nn.Sequential(
                nn.Conv2d(32, 10, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(10)
            )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.softmax(x.reshape((-1,10)))

        return x

train_mnist_data = MNIST('./mnist', train=True, download=False, transform = transforms.ToTensor())
test_mnist_data = MNIST('./mnist', train=False, download=True, transform = transforms.ToTensor())
train_data_loader = DataLoader(train_mnist_data, batch_size=4, shuffle=True)
test_data_loader = DataLoader(test_mnist_data, batch_size=4, shuffle=False)
model = MnistNet()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)
model.load_state_dict(torch.load("./mnist_model.pt"))
for epoch_num in range(40):
    model.train()
    for iter_num, input_data in enumerate(train_data_loader):
        input_image, gt = input_data
        output = model(input_image)
        optimizer.zero_grad()

        loss = loss_fn(output, gt)
        # print("iterate {}: loss={}".format(iter_num, loss.item()))

        # model.zero_grad()
        loss.backward()
    print("=============================Finished Epoch {}".format(epoch_num))
    model.eval()
    test_loss = 0
    correct = 0
    for iter_num, input_data in enumerate(test_data_loader):
        input_image, gt = input_data
        output = model(input_image)

        test_loss += loss_fn(output, gt)
        pred = output.argmax(dim=-1)
        correct += pred.eq(gt.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_data_loader.dataset)
    precision = correct / len(test_data_loader.dataset)
    print("test loss: {}, correct: {} in {}, precition: {}".format(test_loss, 
        correct, len(test_data_loader.dataset), precision))

torch.save(model.state_dict(), "./mnist_model_new.pt")