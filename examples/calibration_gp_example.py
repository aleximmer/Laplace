import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader
from netcal.metrics import ECE
from helper.util_gp import get_dataset, CIFAR10Net
from laplace import Laplace
import tqdm
import torch.distributions as dists

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available()

DATASET = 'FMNIST'
BATCH_SIZE = 25
ds_train, ds_test = get_dataset(DATASET, False, 'cuda')
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

MODEL_NAME = 'FMNIST_CNN_10_2.2e+02.pt'
model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True)#.to('cuda')
state = torch.load(f'./examples/helper/models/{MODEL_NAME}', map_location=torch.device('cpu'))
model.load_state_dict(state['model'])
model = model.cuda()
prior_precision = state['delta']


@torch.no_grad()
def predict(dataloader, model, laplace=False, progress_bar = False):
    py = []

    if progress_bar:
        loader = tqdm.tqdm(dataloader, desc="Evaluating")
    else:
        loader = dataloader

    for x, _ in loader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


probs_map = predict(test_loader, model, laplace=False)
acc = (probs_map.argmax(-1) == targets).float().mean().cpu().item()
ece = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
nll = -dists.Categorical(probs_map).log_prob(targets).mean().cpu().item()
print(f'[MAP] Acc.: {acc:.1%}; ECE: {ece:.1%}; NLL: {nll:.3}')

for m in [50, 200, 800, 1600]:
    print(f'Fitting Laplace-GP for m={m}')
    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='gp',
                 diagonal_kernel=True, M=m,
                 prior_precision=prior_precision)
    la.fit(train_loader, progress_bar = True)

    probs_laplace = predict(test_loader, la, laplace=True, progress_bar=True)
    acc = (probs_laplace.argmax(-1) == targets).float().mean().cpu().item()
    ece = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll = -dists.Categorical(probs_laplace).log_prob(targets).mean().cpu().item()
    print(f'[Laplace] Acc.: {acc:.1%}; ECE: {ece:.1%}; NLL: {nll:.3}')
