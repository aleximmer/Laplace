# %% [markdown]
# # Calibration
#
# An advantage of the Laplace approximation over variational Bayes and Markov Chain Monte Carlo methods is its _post-hoc_ nature. That means we can apply LA on (almost) any _pre-trained_ neural network. In this example, we will see how we can apply the last-layer LA on a ResNet-18 trained on CIFAR-10.
#
# ## Data loading
#
# First, let us load the CIFAR-10 dataset from the Hugging Face Hub. The helper scripts for the data and the model are in the `examples/helper` directory of the main repository.

# %%
import argparse
import warnings

import helper.dataloaders as dl
import numpy as np
import torch
import torch.distributions as dists
from helper.models import load_cifar10_resnet18, tiny_cifar10_cnn
from netcal.metrics import ECE

from laplace import CategoricalLikelihood, Laplace

warnings.simplefilter("ignore", UserWarning)

# A slice of CIFAR-10 is enough to see the effect. Smaller slices and a tiny random
# model are for quick test runs.
parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=5000)
parser.add_argument("--n_test", type=int, default=2000)
parser.add_argument("--tiny_model", action="store_true")
args, _ = parser.parse_known_args()

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = dl.CIFAR10(train=True, n_data=args.n_train)
test_loader = dl.CIFAR10(train=False, n_data=args.n_test)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# %% [markdown]
# ## Load a pre-trained model
#
# Next, we load a ResNet-18 pretrained on CIFAR-10 from the [Hugging Face Hub](https://huggingface.co/edadaltocg/resnet18_cifar10). It runs on the CPU too, but a GPU is much faster.

# %%
model = tiny_cifar10_cnn() if args.tiny_model else load_cifar10_resnet18()
model = model.to(device).eval()

# %% [markdown]
# To simplify the downstream tasks, we will use the following helper function to make predictions. It simply iterates through all minibatches and obtains the predictive probabilities of the CIFAR-10 classes.


# %%
@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.to(device)))
        else:
            py.append(torch.softmax(model(x.to(device)), dim=-1))

    return torch.cat(py).cpu()


# %% [markdown]
# ## The calibration of MAP
#
# We are now ready to see how calibrated is the model. The metrics we use are the expected calibration error (ECE, Naeni et al., AAAI 2015) and the negative (Categorical) log-likelihood. Note that lower values are better for both these metrics.
#
# First, let us inspect the MAP model. We shall use the [`netcal`](https://github.com/fabiankueppers/calibration-framework) library to easily compute the ECE.

# %%
probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

# %% [markdown]
# Running this snippet, we get something like:
#
# ```
# [MAP] Acc.: 95.1%; ECE: 3.5%; NLL: 0.218
# ```
#
# ## The calibration of Laplace
#
# Now we inspect the benefit of the LA. Let us apply the simple last-layer LA model, and optimize the prior precision hyperparameter using a _post-hoc_ marginal likelihood maximization.

# %%
# Laplace
la = Laplace(
    model,
    CategoricalLikelihood(),
    subset_of_weights="last_layer",
    hessian_structure="kron",
)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

# %% [markdown]
# Then, we are ready to see how well does LA improves the calibration of the MAP model:

# %%
probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(
    f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
)

# %% [markdown]
# Running this snippet, we get something like:
#
# ```
# [Laplace] Acc.: 95.1%; ECE: 1.8%; NLL: 0.179
# ```
#
# Notice that the last-layer LA does not do any harm to the accuracy, yet it improves the calibration of the MAP model substantially.
