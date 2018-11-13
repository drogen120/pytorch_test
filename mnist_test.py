import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

mnist_data = MNIST('./mnist', train=True, download=False, transform = transforms.ToTensor())
data_loader = DataLoader(mnist_data, batch_size=4, shuffle=True)
for num_batch, images in enumerate(data_loader):
    print(num_batch, images)
    if num_batch == 4:
        print("finished")
        break