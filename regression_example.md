## Full example: *post-hoc* optimization of the marginal likelihood and prediction

#### Sinusoidal toy data
We show how the marginal likelihood can be used after training a MAP network on a simple sinusoidal regression task. 
Subsequently, we use the optimized LA to predict which provides uncertainty on top of the MAP prediction.
First, we set up the training data for the problem with observation noise \\(\\sigma=0.3\\):
```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace

n_epochs = 1000
batch_size = 150  # full batch
true_sigma_noise = 0.3

# create simple sinusoid data set
X_train = (torch.rand(150) * 8).unsqueeze(-1)
y_train = torch.sin(X_train) + torch.randn_like(X_train) * true_sigma_noise
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)  # +-5 on top of the training X-range
```

#### Training a MAP
We now use `pytorch` to train a neural network with single hidden layer and Tanh activation.
This is standard so nothing new here, yet:
```python
# create and train MAP model
model = torch.nn.Sequential(torch.nn.Linear(1, 50),
                            torch.nn.Tanh(),
                            torch.nn.Linear(50, 1))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
for i in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
```

#### Fitting and optimizing the Laplace approximation using empirical Bayes
With the MAP-trained model at hand, we can estimate the prior precision and observation noise
using empirical Bayes after training.
The `Laplace` method is called to construct a LA for `'regression'` with `'all'` weights.
As default `Laplace` returns a Kronecker factored LA.
We fit the LA to the training data and initialize `log_prior` and `log_sigma`.
Using Adam, we minimize the negative log marginal likelihood for `n_epochs`.

```python
la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='full')
la.fit(train_loader)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()
```

The obtained observation noise is close to the ground truth with a value of \\(\\sigma \\approx 0.28\\)
without the need for any validation data.
The resulting prior precision is \\(\\delta \\approx 0.18\\).

#### Bayesian predictive
Lastly, we compare the MAP prediction to the obtained LA prediction.
For LA, we have a closed-form predictive distribution on the output \\(f\\) which is a Gaussian
\\(\\mathcal{N}(f(x;\\theta_{MAP}), \\mathbb{V}[f] + \\sigma^2)\\):

```python
x = X_test.flatten().cpu().numpy()
f_mu, f_var = la(X_test)
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)
```

In comparison to the MAP, the predictive shows useful uncertainties:

```python
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
                               figsize=(4.5, 2.8))
ax1.set_title('MAP')
ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax1.plot(x, f_mu, color='black', label='$f_{MAP}$')
ax1.legend()

ax2.set_title('LA')
ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax2.plot(x, f_mu, label='$\mathbb{E}[f]$')
ax2.fill_between(x, f_mu-pred_std*2, f_mu+pred_std*2, 
                 alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[f]}$')
ax2.legend()
ax1.set_ylim([-4, 6])
ax1.set_xlim([x.min(), x.max()])
ax2.set_xlim([x.min(), x.max()])
ax1.set_ylabel('$y$')
ax1.set_xlabel('$x$')
ax2.set_xlabel('$x$')
plt.tight_layout()
plt.show()
```

.. image:: regression_example.png
   :align: center
