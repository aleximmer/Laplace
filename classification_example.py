import numpy as np
import matplotlib.pyplot as plt
import torch
from tests.utils import toy_classification_dataset, toy_model

from laplace import Laplace

n_epochs = 200
batch_size = 100
c = 3
torch.manual_seed(711)

X_train, y_train, train_loader, X_test = toy_classification_dataset(batch_size=batch_size, in_dim=4, out_dim=c)
model = toy_model(train_loader, in_dim=4, out_dim=c, regression=False)


# la = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='GP',
#              diagonal_L=True, diagonal_kernel=True)
la = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='diag')
la.fit(train_loader)
log_prior = torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
print('prior precision:', log_prior.exp().item())
