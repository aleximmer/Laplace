import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision.models import resnet18


def load_cifar10_resnet18() -> nn.Module:
    """ResNet-18 trained on CIFAR-10, from https://huggingface.co/edadaltocg/resnet18_cifar10.

    It uses a 3x3 stem without max pooling, the usual change for 32x32 inputs, and expects
    inputs normalized with `helper.dataloaders.CIFAR10_MEAN` and `CIFAR10_STD`.
    """
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    path = hf_hub_download("edadaltocg/resnet18_cifar10", "pytorch_model.bin")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    return model
