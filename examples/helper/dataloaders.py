# Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


train_batch_size = 128
test_batch_size = 100

path = './temp/'


def CIFAR10(train=True, batch_size=None, augm_flag=True):
    if batch_size == None:
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


def get_sinusoid_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, y_train), 
        batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test
    