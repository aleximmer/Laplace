import numpy as np
from sklearn.datasets import make_blobs

import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


train_batch_size = 128
test_batch_size = 100

path = './temp/'


# Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
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
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=train, num_workers=4)

    return loader


# Taken from https://github.com/team-approx-bayes/fromp/ (modified)
class ToydataGenerator:
    def __init__(self, max_iter=5, num_samples=2000, option=0, batch_size=20):
        self.batch_size=batch_size
        self.offset = 5  # Offset when loading data in next_task()

        # Generate data
        if option == 0:
            # Standard settings
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 1:
            # Six tasks
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3], [1.65, 0.1],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1], [0.7, 0.25]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16], [0.14, 0.14],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22], [0.14, 0.14]]

        elif option == 2:
            # All std devs increased
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.12, 0.22], [0.24, 0.12], [0.07, 0.2], [0.16, 0.08], [0.08, 0.16],
                [0.12, 0.16], [0.16, 0.12], [0.08, 0.16], [0.24, 0.08], [0.08, 0.22]]

        elif option == 3:
            # Tougher to separate
            centers = [[0, 0.2], [0.6, 0.65], [1.3, 0.4], [1.6, -0.22], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.55], [1., 0.1], [1.7, -0.3], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 4:
            # Two tasks, of same two gaussians
            centers = [[0, 0.2], [0, 0.2],
                       [0.45, 0], [0.45, 0]]
            std = [[0.08, 0.22], [0.08, 0.22],
                   [0.08, 0.16], [0.08, 0.16]]

        else:
            # If new / unknown option
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        if option != 1 and max_iter > 5:
            raise Exception("Current toydatagenerator only supports up to 5 tasks.")

        self.X, self.y = make_blobs(num_samples*2*max_iter, centers=centers, cluster_std=std)
        self.X = self.X.astype('float32')
        h = 0.01
        self.x_min, self.x_max = self.X[:, 0].min() - 0.2, self.X[:, 0].max() + 0.2
        self.y_min, self.y_max = self.X[:, 1].min() - 0.2, self.X[:, 1].max() + 0.2
        self.data_min = np.array([self.x_min, self.y_min], dtype='float32')
        self.data_max = np.array([self.x_max, self.y_max], dtype='float32')
        self.data_min = np.expand_dims(self.data_min, axis=0)
        self.data_max = np.expand_dims(self.data_max, axis=0)
        xx, yy = np.meshgrid(np.arange(self.x_min, self.x_max, h),
                             np.arange(self.y_min, self.y_max, h))
        xx = xx.astype('float32')
        yy = yy.astype('float32')
        self.test_shape = xx.shape
        X_test = np.c_[xx.ravel(), yy.ravel()]
        self.X_test = torch.from_numpy(X_test)
        self.y_test = torch.zeros((len(self.X_test)), dtype=self.X_test.dtype)
        self.max_iter = max_iter
        self.num_samples = num_samples  # number of samples per task

        if option == 1:
            self.offset = 6
        elif option == 4:
            self.offset = 2

        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            x_train_0 = self.X[self.y == self.cur_iter]
            x_train_1 = self.X[self.y == self.cur_iter + self.offset]
            y_train_0 = np.zeros_like(self.y[self.y == self.cur_iter])
            y_train_1 = np.ones_like(self.y[self.y == self.cur_iter + self.offset])
            x_train = np.concatenate([x_train_0, x_train_1], axis=0)
            y_train = np.concatenate([y_train_0, y_train_1], axis=0)
            y_train = y_train.astype('int64')
            self.cur_iter += 1
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True)
            
            test_dataset = TensorDataset(self.X_test, self.y_test)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False)
            
            return train_loader, test_loader
