import warnings
<<<<<<< HEAD

warnings.simplefilter("ignore", UserWarning)

import helper.dataloaders as dl
import helper.wideresnet as wrn
import numpy as np
import torch
import torch.distributions as dists
from helper import util
from netcal.metrics import ECE
=======

warnings.simplefilter('ignore', UserWarning)
>>>>>>> main

import torch  # noqa: E402
import torch.distributions as dists  # noqa: E402
import numpy as np  # noqa: E402
import helper.wideresnet as wrn  # noqa: E402
import helper.dataloaders as dl  # noqa: E402
from helper import util  # noqa: E402
from netcal.metrics import ECE  # noqa: E402

from laplace import Laplace  # noqa: E402

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

train_loader = dl.CIFAR10(train=True)
test_loader = dl.CIFAR10(train=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

# The model is a standard WideResNet 16-4
# Taken as is from https://github.com/hendrycks/outlier-exposure
model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()

util.download_pretrained_model()
model.load_state_dict(torch.load("./temp/CIFAR10_plain.pt"))


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).float().mean()
ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

# Laplace
la = Laplace(
<<<<<<< HEAD
    model, "classification", subset_of_weights="last_layer", hessian_structure="kron"
=======
    model, 'classification', subset_of_weights='last_layer', hessian_structure='kron'
>>>>>>> main
)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

print(
<<<<<<< HEAD
    f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
=======
    f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}'
>>>>>>> main
)
