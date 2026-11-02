import argparse
import warnings

warnings.simplefilter("ignore", UserWarning)

import helper.dataloaders as dl
import numpy as np
import torch
import torch.distributions as dists
from helper.models import load_cifar10_resnet18, tiny_cifar10_cnn
from netcal.metrics import ECE  # noqa: E402

from laplace import Laplace  # noqa: E402

# A slice of the data is enough to show the effect. Smaller slices and a tiny random
# model are for quick test runs.
parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=5000)
parser.add_argument("--n_test", type=int, default=2000)
parser.add_argument("--tiny_model", action="store_true")
args = parser.parse_args()

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = dl.CIFAR10(train=True, n_data=args.n_train)
test_loader = dl.CIFAR10(train=False, n_data=args.n_test)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# A ResNet-18 pretrained on CIFAR-10, from the Hugging Face Hub
model = tiny_cifar10_cnn() if args.tiny_model else load_cifar10_resnet18()
model = model.to(device).eval()


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.to(device)))
        else:
            py.append(torch.softmax(model(x.to(device)), dim=-1))

    return torch.cat(py).cpu()


probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

# Laplace
la = Laplace(
    model, "classification", subset_of_weights="last_layer", hessian_structure="kron"
)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(
    f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
)
