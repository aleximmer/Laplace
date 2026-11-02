import argparse
import warnings

import helper.dataloaders as dl
import numpy as np
import torch
import torch.distributions as dists
from netcal.metrics import ECE
from torch import nn

from laplace import Laplace

# Data subsets, training length, and GP subset sizes, e.g. for a quick test run
parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=None)
parser.add_argument("--n_test", type=int, default=None)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--subset_sizes", type=int, nargs="+", default=[50, 200, 800, 1600])
args = parser.parse_args()

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

warnings.simplefilter("ignore", UserWarning)


device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = dl.FashionMNIST(train=True, n_data=args.n_train)
test_loader = dl.FashionMNIST(train=False, n_data=args.n_test)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# A small CNN. With L2 regularization, MAP training matches a Gaussian prior with
# precision `PRIOR_PRECISION`; the mean loss divides it by the number of data points.
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

for epoch in range(args.epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()

model.eval()


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

for m in args.subset_sizes:
    print(f"Fitting Laplace-GP for m={m}")
    la = Laplace(
        model,
        "classification",
        subset_of_weights="all",
        hessian_structure="gp",
        independent_outputs=True,
        n_subset=m,
        prior_precision=PRIOR_PRECISION,
    )
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )
