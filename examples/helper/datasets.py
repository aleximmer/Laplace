import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import torch

import os

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = '/'.join(PACKAGE_DIR.split('/')[:-1])
DATA_DIR = ROOT + '/data'

CIFAR10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                         (0.24703223, 0.24348513, 0.26158784))
])

MNIST_transform = transforms.ToTensor()


class QuickDS(VisionDataset):

    def __init__(self, ds, device):
        self.D = [(ds[i][0].to(device).requires_grad_(), torch.tensor(ds[i][1]).to(device).requires_grad_())
                  for i in range(len(ds))]
        self.K = ds.K
        self.channels = ds.channels
        self.pixels = ds.pixels

    def __getitem__(self, index):
        return self.D[index]

    def __len__(self):
        return len(self.D)


def get_dataset(dataset, double, device=None):
    if dataset == 'MNIST':
        ds_train = MNIST(train=True, double=double)
        ds_test = MNIST(train=False, double=double)
    elif dataset == 'FMNIST':
        ds_train = FMNIST(train=True, double=double)
        ds_test = FMNIST(train=False, double=double)
    elif dataset == 'CIFAR10':
        ds_train = CIFAR10(train=True, double=double)
        ds_test = CIFAR10(train=False, double=double)
    else:
        raise ValueError('Invalid dataset argument')
    if device is not None:
        return QuickDS(ds_train, device), QuickDS(ds_test, device)
    else:
        return ds_train, ds_test


class CIFAR10(dset.CIFAR10):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=CIFAR10_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.K = 10
        self.pixels = 32
        self.channels = 3
        if double:
            self.data = self.data.astype(np.double)


class MNIST(dset.MNIST):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=MNIST_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.K = 10
        self.pixels = 28
        self.channels = 1
        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()


class FMNIST(dset.FashionMNIST):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=MNIST_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.K = 10
        self.pixels = 28
        self.channels = 1
        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()