# %% [markdown]
# # GP Inference
#
# > **Caution:** To follow this tutorial, you need to install the `netcal` dependency.
#
# Applying the General-Gauss-Newton (GGN) approximation to the Hessian in the Laplace approximation (LA) of the BNN posterior
# turns the underlying probabilistic model from a BNN into a generalized linear model (GLM).
# This GLM is equivalent to a Gaussian Process (GP) with a particular kernel [1, 2].
#
# In this notebook, we will show how to use `laplace` library to perform GP inference on top of a _pre-trained_ neural network. We use the last layer only: its GP kernel is the inner product of the network's features, which keeps the computation cheap even for a deep network.
#
# The code runs on the CPU, but a GPU makes it much faster.
#
# ## Data loading
#
# First, let us load a slice of the CIFAR-10 dataset from the Hugging Face Hub, which is enough to see the effect. The helper scripts for the data and the model are in the `examples/helper` directory of the main repository.

# %%
import argparse
import warnings

import helper.dataloaders as dl
import numpy as np
import torch
import torch.distributions as dists
from helper.models import load_cifar10_resnet18, tiny_cifar10_cnn
from netcal.metrics import ECE

from laplace import Laplace

warnings.simplefilter("ignore", UserWarning)

# Smaller slices, GP subset sizes, and a tiny random model are for quick test runs
parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=5000)
parser.add_argument("--n_test", type=int, default=2000)
parser.add_argument("--subset_sizes", type=int, nargs="+", default=[200, 800])
parser.add_argument("--tiny_model", action="store_true")
args, _ = parser.parse_known_args()

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# No augmentation: the GP fit passes over the data several times and needs the same
# inputs each time, see https://github.com/aleximmer/Laplace/issues/330
train_loader = dl.CIFAR10(train=True, augm_flag=False, n_data=args.n_train)
test_loader = dl.CIFAR10(train=False, n_data=args.n_test)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# %% [markdown]
# ## Load a pre-trained model
#
# Next, we load a ResNet-18 pretrained on CIFAR-10 from the [Hugging Face Hub](https://huggingface.co/edadaltocg/resnet18_cifar10).

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
# Next, we run Laplace-GP inference to calibrate neural network's predictions. Since running exact GP inference is computationally infeasible, we perform Subset-of-Datapoints (SoD) [3] approximation here. In the code below, `m` denotes the number of datapoints used in the SoD posterior. The prior precision sets the scale of the GP kernel: a smaller value means a wider prior over the last-layer weights.

# %%
for m in args.subset_sizes:
    print(f"Fitting Laplace-GP for m={m}")
    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="gp",
        independent_outputs=True,
        n_subset=m,
        prior_precision=1.0,
    )
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )

# %% [markdown]
# ```
# Fitting Laplace-GP for m=200
# [Laplace] Acc.: 95.1%; ECE: 2.5%; NLL: 0.182
# Fitting Laplace-GP for m=800
# [Laplace] Acc.: 95.0%; ECE: 2.2%; NLL: 0.18
# ```
#
# Notice that the post-hoc Laplace-GP inference does not have a significant impact on the accuracy, yet it improves the calibration (in terms of ECE) of the MAP model substantially.
#
# ## References
#
# [1] Khan, Mohammad Emtiyaz E., et al. "Approximate inference turns deep networks into gaussian processes." Advances in neural information processing systems 32 (2019)
#
# [2] Immer, Alexander, Maciej Korzepa, and Matthias Bauer. "Improving predictions of Bayesian neural nets via local linearization." International Conference on Artificial Intelligence and Statistics. PMLR, 2021
#
# [3] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Springer, 2004
