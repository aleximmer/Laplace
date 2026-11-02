# CIFAR-10 transforms taken from https://github.com/AlexMeinke/certified-certain-uncertainty
from collections.abc import Callable

import torch
import torch.utils.data as data_utils
from torchvision import transforms

train_batch_size = 128
test_batch_size = 100

# Normalization the pretrained CIFAR-10 ResNet-18 was trained with
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class HFImageDataset(data_utils.Dataset):
    """Yields `(image, label)` pairs from an image dataset on the Hugging Face Hub."""

    def __init__(
        self,
        name: str,
        split: str,
        image_key: str,
        transform: Callable,
        n_data: int | None = None,
    ) -> None:
        # Imported here so that the sinusoid examples do not need `datasets`
        from datasets import load_dataset

        self.ds = load_dataset(name, split=split)

        if n_data is not None:
            self.ds = self.ds.select(range(min(n_data, len(self.ds))))

        self.image_key = image_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.ds[int(idx)]
        return self.transform(row[self.image_key]), row["label"]


def CIFAR10(
    train: bool = True,
    batch_size: int | None = None,
    augm_flag: bool = True,
    n_data: int | None = None,
) -> data_utils.DataLoader:
    if batch_size is None:
        if train:
            batch_size = train_batch_size
        else:
            batch_size = test_batch_size

    transform_base = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        ]
        + transform_base
    )
    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    split = "train" if train else "test"
    dataset = HFImageDataset("uoft-cs/cifar10", split, "img", transform, n_data)

    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=train)


def FashionMNIST(
    train: bool = True, batch_size: int = 128, n_data: int | None = None
) -> data_utils.DataLoader:
    split = "train" if train else "test"
    dataset = HFImageDataset(
        "zalando-datasets/fashion_mnist", split, "image", transforms.ToTensor(), n_data
    )

    return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_sinusoid_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise
    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, y_train), batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test
