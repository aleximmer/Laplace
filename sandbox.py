import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace.baselaplace import FullLaplace

n_epochs = 1000
batch_size = 150  # full batch
true_sigma_noise = 0.3
torch.manual_seed(711)

# create simple sinusoid data set
X_train = (torch.rand(150) * 8).unsqueeze(-1)
y_train = torch.sin(X_train) + torch.randn_like(X_train) * true_sigma_noise
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)  # +-5 on top of the training X-range

# create and train MAP model
model = torch.nn.Sequential(torch.nn.Linear(1, 50),
                            torch.nn.Tanh(),
                            torch.nn.Linear(50, 1))

a = FullLaplace(model, 'regression')
a.fit(train_loader)

