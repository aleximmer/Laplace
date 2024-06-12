## Full example: Functional Laplace (GP) on FMNIST image classifier
Applying the General-Gauss-Newton (GGN) approximation to the Hessian in the Laplace approximation (LA) of the BNN posterior
turns the underlying probabilistic model from a BNN into a generalized linear model (GLM).
This GLM is equivalent to a Gaussian Process (GP) with a particular kernel [1, 2]. 

In this notebook, we will show how to use `laplace` library to perform GP inference on top of a *pre-trained* neural network.

Note that a GPU with CUDA support is needed for this example. We recommend using a GPU with at least 24 GB of memory. If less memory is available, we suggest reducing `BATCH_SIZE` below.

#### Data loading

First, let us load the FMIST dataset. The helper scripts for FMNIST and pre-trained CNN are available in the `examples/helper` directory in the main repository.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributions as dists
from netcal.metrics import ECE

from helper.util_gp import get_dataset, CIFAR10Net
from laplace import Laplace

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available()

DATASET = 'FMNIST'
BATCH_SIZE = 256
ds_train, ds_test = get_dataset(DATASET, False, 'cuda')
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
```

#### Load a pre-trained model

Next, we load a pre-trained CNN model. The code to train the model can be found in [BNN-predictions repo](https://github.com/AlexImmer/BNN-predictions).

``` python
MODEL_NAME = 'FMNIST_CNN_10_2.2e+02.pt'
model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True).to('cuda')
state = torch.load(f'helper/models/{MODEL_NAME}')
model.load_state_dict(state['model'])
model = model.cuda()
prior_precision = state['delta']
```

To simplify the downstream tasks, we will use the following helper function to make predictions. It simply iterates through all minibatches and obtains the predictive probabilities of the FMNIST classes.

``` python
@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu().numpy()
```

#### The calibration of MAP

We are now ready to see how calibrated is the model. The metrics we use are the expected calibration error (ECE, Naeni et al., AAAI 2015) and the negative (Categorical) log-likelihood. Note that lower values are better for both these metrics.

First, let us inspect the MAP model. We shall use the [`netcal`](https://github.com/fabiankueppers/calibration-framework) library to easily compute the ECE.

``` python
probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')
```

Running this snippet, we would get:

```
[MAP] Acc.: 91.7%; ECE: 1.6%; NLL: 0.253
```

### The calibration of Laplace

Next, we run Laplace-GP inference to calibrate neural network's predictions. Since running exact GP inference is computationally infeasible, we perform Subset-of-Datapoints (SoD) [3] approximation here. In the code below, `m`denotes the number of datapoints used in the SoD posterior. 

Execution of the cell below can take up to 5min (depending on the exact hardware used).

``` python
for m in [50, 200, 800, 1600]:
    print(f'Fitting Laplace-GP for m={m}')
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='gp',
                 diagonal_kernel=True, 
                 num_data=m,
                 prior_precision=prior_precision)
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(f'[Laplace-GP, m={m}] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
```

```
Fitting Laplace-GP for m=50
[Laplace] Acc.: 91.6%; ECE: 1.5%; NLL: 0.252
Fitting Laplace-GP for m=200
[Laplace] Acc.: 91.5%; ECE: 1.1%; NLL: 0.252
Fitting Laplace-GP for m=800
[Laplace] Acc.: 91.4%; ECE: 0.8%; NLL: 0.254
Fitting Laplace-GP for m=1600
[Laplace] Acc.: 91.3%; ECE: 0.7%; NLL: 0.257
```

Notice that the post-hoc Laplace-GP inference does not have a significant impact on the accuracy, yet it improves the calibration (in terms of ECE) of the MAP model substantially.
<br />
<br />
<br />
<br />

### References
[1] Khan, Mohammad Emtiyaz E., et al. "Approximate inference turns deep networks into gaussian processes." Advances in neural information processing systems 32 (2019)

[2] Immer, Alexander, Maciej Korzepa, and Matthias Bauer. "Improving predictions of Bayesian neural nets via local linearization." International Conference on Artificial Intelligence and Statistics. PMLR, 2021

[3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Springer, 2004

