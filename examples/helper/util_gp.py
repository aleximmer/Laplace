import os
import urllib

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from deepobs.pytorch.testproblems.testproblems_utils import tfconv2d, tfmaxpool2d
from torch import nn
from torchvision.datasets import VisionDataset

PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = "/".join(PACKAGE_DIR.split("/")[:-1])
DATA_DIR = ROOT + "/data"

MNIST_transform = transforms.ToTensor()


def download_pretrained_model():
    # Download pre-trained model if necessary
    if not os.path.isfile("FMNIST_CNN_10_2.2e+02.pt"):
        if not os.path.exists("./temp"):
            os.makedirs("./temp")

        urllib.request.urlretrieve(
            "https://drive.usercontent.google.com/download?id=1jPDKrykvU2viKisXT-8Q7ANMOPoTnPyO&export=download",
            "./temp/FMNIST_CNN_10_2.2e+02.pt",
        )


class QuickDS(VisionDataset):
    def __init__(self, ds, device):
        self.D = [
            (ds[i][0].to(device).requires_grad_(), torch.tensor(ds[i][1]).to(device))
            for i in range(len(ds))
        ]
        self.K = ds.K
        self.channels = ds.channels
        self.pixels = ds.pixels

    def __getitem__(self, index):
        return self.D[index]

    def __len__(self):
        return len(self.D)


def get_dataset(dataset, double, device=None):
    if dataset == "FMNIST":
        ds_train = FMNIST(train=True, double=double)
        ds_test = FMNIST(train=False, double=double)
    else:
        raise ValueError("Invalid dataset argument")
    if device is not None:
        return QuickDS(ds_train, device), QuickDS(ds_test, device)
    else:
        return ds_train, ds_test


class FMNIST(dset.FashionMNIST):
    def __init__(
        self,
        root=DATA_DIR,
        train=True,
        download=True,
        transform=MNIST_transform,
        double=False,
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.K = 10
        self.pixels = 28
        self.channels = 1
        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()


class CIFAR10Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    In Deepobs called `net_cifar10_3c3d`
    """

    def __init__(self, in_channels=3, n_out=10, use_tanh=False):
        super(CIFAR10Net, self).__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.add_module(
            "conv1", tfconv2d(in_channels=in_channels, out_channels=64, kernel_size=5)
        )
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "maxpool1", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same")
        )

        self.add_module(
            "conv2", tfconv2d(in_channels=64, out_channels=96, kernel_size=3)
        )
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "maxpool2", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same")
        )

        self.add_module(
            "conv3",
            tfconv2d(
                in_channels=96, out_channels=128, kernel_size=3, tf_padding_type="same"
            ),
        )
        self.add_module("relu3", nn.ReLU())
        self.add_module(
            "maxpool3", tfmaxpool2d(kernel_size=3, stride=2, tf_padding_type="same")
        )

        self.add_module("flatten", nn.Flatten())

        self.add_module("dense1", nn.Linear(in_features=3 * 3 * 128, out_features=512))
        self.add_module("relu4", activ())
        self.add_module("dense2", nn.Linear(in_features=512, out_features=256))
        self.add_module("relu5", activ())
        self.add_module("dense3", nn.Linear(in_features=256, out_features=n_out))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
