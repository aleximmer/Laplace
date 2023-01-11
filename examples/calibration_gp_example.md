## Full example: Functional Laplace (GP) on FMNIST image classifier
Usign GGN approximation of the Hessian matrix in Laplace approximation of a BNN gives rise to a generalized linear model which is equivalent to a GP model with a particular kernel [1, 2]. 

In this notebook, we will show how to use `laplace` library to perform GP inference on top of a *pre-trained* neural network.

Note that a GPU with CUDA support is needed for this example. We recommend using a GPU with at least 24 GB of memory. If less memory is available, we suggest reducing `BATCH_SIZE` below.
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from helper.util_gp import get_dataset, CIFAR10Net
from helper.util import predict, get_metrics
from laplace import Laplace

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
```

``` python
assert torch.cuda.is_available()
```
In the first step, we load the `FMNIST` dataset.
``` python
DATASET = 'FMNIST'
BATCH_SIZE = 128
ds_train, ds_test = get_dataset(DATASET, False, 'cuda')
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
```
Next, we load a pre-trained CNN model. The code to train the model can be found in [BNN-predictions repo](https://github.com/AlexImmer/BNN-predictions).
``` python
MODEL_NAME = 'FMNIST_CNN_10_2.2e+02.pt'
model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True).to('cuda')
state = torch.load(f'helper/models/{MODEL_NAME}')
model.load_state_dict(state['model'])
model = model.cuda()
prior_precision = state['delta']
```
To get the performance of the MAP model, run:
``` python
probs_map = predict(test_loader, model, laplace=False)
acc_map, ece_map, nll_map = get_metrics(probs_map, targets)
print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')
```
```
TODO: MAP results print output
```

Next, we run Laplace-GP inference to calibrate neural network's predictions. Since running exact GP inference is computationally infeasible, we perform Subset-of-Datapoints (SoD) [3] approximation here. In the code below, `m`denotes the number of datapoints used in the SoD posterior.

``` python
metrics_df = pd.DataFrame()
for m in [50, 200, 800, 1600]:
    print(f'Fitting Laplace-GP for m={m}')
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='gp',
                 diagonal_kernel=True, M=m,
                 prior_precision=prior_precision)
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True, la_type='gp')
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean().cpu().item()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean().cpu().item()

    print(f'[Laplace-GP, m={m}] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

    metrics_df = metrics_df.append({'M': m, 'acc_laplace': acc_laplace, 'ece_laplace':ece_laplace, 'nll_laplace': nll_laplace},
                                   ignore_index=True)
```

```
TODO: Laplace-GP results print output
```

``` python
# TODO: plot
print(metrics_df)
```
Notice that the post-hoc Laplace-GP inference does not have a significant impact on the accuracy, yet it improves the calibration of the MAP model substantially.

[1] Khan, Mohammad Emtiyaz E., et al. "Approximate inference turns deep networks into gaussian processes." Advances in neural information processing systems 32 (2019)

[2] Immer, Alexander, Maciej Korzepa, and Matthias Bauer. "Improving predictions of Bayesian neural nets via local linearization." International Conference on Artificial Intelligence and Statistics. PMLR, 2021

[3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Springer, 2004

