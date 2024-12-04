## Motivation

Often, we want to compute non-analytic expectation \(\mathbb{E}_{p(f(x) \mid \mathcal{D})} [g(f(x))]\) of a function \(g(f(x))\) given a Laplace posterior over functions \(p(f(x) \mid \mathcal{D})\) at input \(x\).
This naturally arises in decision-making:
Given a posterior belief about an unknown function \(f\), we would like to compute the expected utility of \(x\).
In Bayesian optimization, this is called \_acquisition function_.
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

!!! tip

    If you need the gradient of quantities that depend on Laplace's predictive
    distribution/samples, then be sure to specify `enable_backprop=True`. This is in
    fact necessary for continuous Bayesian optimization.

## Thompson Sampling

The simplest acquisition function that can be obtained from the Laplace posterior is Thompson sampling.
This is defined as \(a \sim p(f(x) \mid \mathcal{D})\).
Thus, given `la`, it can be obtained very simply:

```python
f_sample = la.functional_samples(x_test, n_samples=1)
```

Note that `f_sample` can be obtained through the `"nn"` and `"glm"` predictives.
The `la.functional_samples` function supports both options.

```python
for pred_type in [PredType.GLM, PredType.NN]:
    print(f"Thompson sampling, {pred_type}")

    x_test = torch.randn(10, 2)
    x_test.requires_grad = True

    f_sample = la.functional_samples(x_test, pred_type=pred_type, n_samples=1)
    f_sample = f_sample.squeeze(0)  # We only use a single sample
    print(f"TS shape: {f_sample.shape}, TS requires grad: {f_sample.requires_grad}")

    grad_x = autograd.grad(f_sample.sum(), x_test)[0]

    print(
        f"Grad x_test shape: {grad_x.shape}, Grad x_test vanishing: {torch.allclose(grad_x, torch.tensor(0.0))}"
    )
    print()
```

The snippet above will output:

```
Thompson sampling, glm
TS shape: torch.Size([10, 1]), TS requires grad: True
Grad x_test shape: torch.Size([10, 2]), Grad x_test vanishing: False

Thompson sampling, nn
TS shape: torch.Size([10, 1]), TS requires grad: True
Grad x_test shape: torch.Size([10, 2]), Grad x_test vanishing: False
```

As we can see, the gradient can be computed through Laplace's predictive and its non-vanishing.

## Monte-Carlo EI

In general, given a choice of utility function \(u(f(x))\), any acquisition function can be obtained w.r.t. the Laplace posterior.
For example, to compute Monte-Carlo-approximated EI, we can do so via:

```python
f_samples = la.functional_samples(x_test, pred_type=pred_type, n_samples=10)
ei = (f_samples - f_best.reshape(1, 1, 1)).clamp(0.0).mean(0)
```

Again, if \(u\) is differentiable, then we can obtain the gradient w.r.t. the input, and we can do continuous Bayesian optimization.
