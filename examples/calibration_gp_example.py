import warnings

import numpy as np
import torch
import torch.distributions as dists
from helper.util_gp import CIFAR10Net, download_pretrained_model, get_dataset
from netcal.metrics import ECE
from torch.utils.data import DataLoader

from laplace import Laplace

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

warnings.simplefilter("ignore", UserWarning)


assert torch.cuda.is_available()

DATASET = "FMNIST"
BATCH_SIZE = 25
ds_train, ds_test = get_dataset(DATASET, False, "cuda")
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

MODEL_NAME = "FMNIST_CNN_10_2.2e+02.pt"
model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True).to("cuda")
download_pretrained_model()
state = torch.load(f"./temp/{MODEL_NAME}")
model.load_state_dict(state["model"])
model = model.cuda()
prior_precision = state["delta"]


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

for m in [50, 200, 800, 1600]:
    print(f"Fitting Laplace-GP for m={m}")
    la = Laplace(
        model,
        "classification",
        subset_of_weights="all",
        hessian_structure="gp",
        diagonal_kernel=True,
        num_data=m,
        prior_precision=prior_precision,
    )
    la.fit(train_loader)

    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )
