import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


train_batch_size = 128
test_batch_size = 100

path = './temp/'


def MNIST(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.MNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


def FMNIST(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def CIFAR10(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR10(path, train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


def CIFAR100(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR100(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def SVHN(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    if train:
        split = 'train'
    else:
        split = 'test'

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.SVHN(path, split=split, transform=transform, download=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader
