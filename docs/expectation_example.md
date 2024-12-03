## Motivation

Often, we want to compute non-analytic expectation \(\mathbb{E}_{p(f(x) \mid \mathcal{D})} [g(f(x))]\) of a function \(g(f(x))\) given a Laplace posterior over functions \(p(f(x) \mid \mathcal{D})\) at input \(x\).
This naturally arises in decision-making:
Given a posterior belief about an unknown function \(f\) we would like to compute the expected utility of \(x\).
In Bayesian optimization, this is called _acquisition function_.
For some utility function \(g\) and posterior belief \(p(f(x) \mid \mathcal{D})\), the resulting acquisition function can be computed analytically, e.g. if \(g\) is linear and the posterior is Gaussian (process).
But, in general, closed-form solutions don't exist. 

In this example, we will see how easy it is to compute a _differentiable_, Monte-Carlo approximated acquisition function under the posterior distribution over neural network functions implied by a Laplace approximation.

## Laplace approximations

As always, the first step of a Laplace approximation is MAP estimation.

```python
import torch
import torch.utils.data as data_utils
from torch import autograd, nn, optim

from laplace import Laplace
from laplace.utils.enums import HessianStructure, Likelihood, PredType, SubsetOfWeights

torch.manual_seed(123)

model = nn.Sequential(nn.Linear(2, 10), nn.GELU(), nn.Linear(10, 1))
X, Y = torch.randn(5, 2), torch.randn(5, 1)
train_loader = data_utils.DataLoader(data_utils.TensorDataset(X, Y), batch_size=3)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()

    for x, y in train_loader:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
```

Then, we are ready to obtain the Laplace approximation. 
In this example, we focus on the weight-space approximation, but the same can be done in the function space directly by simply specifying `hessian_structure=HessianStructure.GP`.

```python
la = Laplace(
    model,
    Likelihood.REGRESSION,
    subset_of_weights=SubsetOfWeights.ALL,
    hessian_structure=HessianStructure.KRON,
    enable_backprop=True,
)
la.fit(train_loader)
la.optimize_prior_precision(PredType.GLM)
```

## First Acquisition Function: Thompson Sampling

