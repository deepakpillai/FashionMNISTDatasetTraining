import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


class Datapack():
    def __init__(self):
        pass

    def download_fasion_minst(self, data_for_train=True):
        return datasets.FashionMNIST(root='data', train=data_for_train, download=True, transform=transforms.ToTensor())

    def load_fasion_mnist(self, data):
        return DataLoader(dataset=data, batch_size=32, shuffle=True)
