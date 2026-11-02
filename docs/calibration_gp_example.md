!!! caution

    To follow this tutorial, you need to install the `netcal` dependency.

Applying the General-Gauss-Newton (GGN) approximation to the Hessian in the Laplace approximation (LA) of the BNN posterior
turns the underlying probabilistic model from a BNN into a generalized linear model (GLM).
This GLM is equivalent to a Gaussian Process (GP) with a particular kernel [1, 2].

In this notebook, we will show how to use `laplace` library to perform GP inference on top of a trained neural network.

The code runs on the CPU, but a GPU makes it much faster.

## Data loading

First, let us load the FMNIST dataset from the Hugging Face Hub. The helper script for the data is in the `examples/helper` directory of the main repository.

```python
import numpy as np
import torch
import torch.distributions as dists
from netcal.metrics import ECE
from torch import nn

import helper.dataloaders as dl
from laplace import Laplace

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = dl.FashionMNIST(train=True)
test_loader = dl.FashionMNIST(train=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
```

## Train a small CNN

Next, we train a small CNN for a few epochs. Training with L2 regularization finds the MAP estimate under a Gaussian prior. The loss is averaged over the data, so the weight decay is the prior precision divided by the number of training points. We pass the same prior precision to Laplace below.

```python
PRIOR_PRECISION = 100.0

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=PRIOR_PRECISION / len(train_loader.dataset),
)

for epoch in range(5):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()

model.eval()
```

To simplify the downstream tasks, we will use the following helper function to make predictions. It simply iterates through all minibatches and obtains the predictive probabilities of the FMNIST classes.

```python
@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.to(device)))
        else:
            py.append(torch.softmax(model(x.to(device)), dim=-1))

    return torch.cat(py).cpu()
```

## The calibration of MAP

We are now ready to see how calibrated is the model. The metrics we use are the expected calibration error (ECE, Naeni et al., AAAI 2015) and the negative (Categorical) log-likelihood. Note that lower values are better for both these metrics.

First, let us inspect the MAP model. We shall use the [`netcal`](https://github.com/fabiankueppers/calibration-framework) library to easily compute the ECE.

```python
probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')
```

Running this snippet, we would get:

```
[MAP] Acc.: 89.1%; ECE: 1.5%; NLL: 0.318
```

## The calibration of Laplace

Next, we run Laplace-GP inference to calibrate neural network's predictions. Since running exact GP inference is computationally infeasible, we perform Subset-of-Datapoints (SoD) [3] approximation here. In the code below, `m`denotes the number of datapoints used in the SoD posterior.

Execution of the cell below can take up to 5min (depending on the exact hardware used).

```python
for m in [50, 200, 800, 1600]:
    print(f'Fitting Laplace-GP for m={m}')
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='gp',
                 independent_outputs=True,
                 n_subset=m,
                 prior_precision=PRIOR_PRECISION)
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(f'[Laplace-GP, m={m}] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
```

```
Fitting Laplace-GP for m=50
[Laplace] Acc.: 89.1%; ECE: 1.6%; NLL: 0.318
Fitting Laplace-GP for m=200
[Laplace] Acc.: 89.1%; ECE: 1.6%; NLL: 0.318
Fitting Laplace-GP for m=800
[Laplace] Acc.: 89.0%; ECE: 1.7%; NLL: 0.318
Fitting Laplace-GP for m=1600
[Laplace] Acc.: 89.0%; ECE: 1.8%; NLL: 0.318
```

Notice that the post-hoc Laplace-GP inference does not have a significant impact on the accuracy, yet it improves the calibration (in terms of ECE) of the MAP model substantially.

## References

[1] Khan, Mohammad Emtiyaz E., et al. "Approximate inference turns deep networks into gaussian processes." Advances in neural information processing systems 32 (2019)

[2] Immer, Alexander, Maciej Korzepa, and Matthias Bauer. "Improving predictions of Bayesian neural nets via local linearization." International Conference on Artificial Intelligence and Statistics. PMLR, 2021

[3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Springer, 2004
