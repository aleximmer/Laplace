## Full example: *post-hoc* Laplace on a large image classifier

An advantage of the Laplace approximation over variational Bayes and Markov Chain Monte Carlo methods is its *post-hoc* nature. That means we can apply LA on (almost) any *pre-trained* neural network. In this example, we will see how we can apply the last-layer LA on a deep WideResNet model, trained on CIFAR-10.

#### Data loading

First, let us load the CIFAR-10 dataset. The helper scripts for CIFAR-10 and WideResNet are available in the `examples/helper` directory in the main repository.

```python
import torch
import torch.distributions as dists
import numpy as np
import helper.wideresnet as wrn
import helper.dataloaders as dl
from helper import util
from netcal.metrics import ECE

from laplace import Laplace


np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

train_loader = dl.CIFAR10(train=True)
test_loader = dl.CIFAR10(train=False)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
```

#### Load a pre-trained model

Next, we will load a pre-trained WideResNet-16-4 model. Note that a GPU with CUDA support is needed for this example.

``` python
# The model is a standard WideResNet 16-4
# Taken as is from https://github.com/hendrycks/outlier-exposure
model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()

util.download_pretrained_model()
model.load_state_dict(torch.load('./temp/CIFAR10_plain.pt'))
```

To simplify the downstream tasks, we will use the following helper function to make predictions. It simply iterates through all minibatches and obtains the predictive probabilities of the CIFAR-10 classes.

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
[MAP] Acc.: 94.8%; ECE: 2.0%; NLL: 0.172
```

### The calibration of Laplace

Now we inspect the benefit of the LA. Let us apply the simple last-layer LA model, and optimize the prior precision hyperparameter using a *post-hoc* marginal likelihood maximization.

``` python
# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')
la.fit(train_loader)
la.optimize_prior_precision(method='marglik')
```

Then, we are ready to see how well does LA improves the calibration of the MAP model:

``` python
probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
```

Running this snippet, we obtain:

```
[Laplace] Acc.: 94.8%; ECE: 0.8%; NLL: 0.157
```

Notice that the last-layer LA does not do any harm to the accuracy, yet it improves the calibration of the MAP model substantially.
