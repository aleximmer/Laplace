import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netcal.metrics import ECE
import torch
from torch.utils.data import DataLoader
import torch.distributions as dists

from helper.util_gp import get_dataset, CIFAR10Net
from helper.util import predict
from laplace import Laplace

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


assert torch.cuda.is_available()

DATASET = 'FMNIST'
MODEL_NAME = 'FMNIST_CNN_10_2.2e+02.pt'

ds_train, ds_test = get_dataset(DATASET, False, 'cuda')
train_loader = DataLoader(ds_train, batch_size=48, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=24, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True).to('cuda')
state = torch.load(f'helper/models/{MODEL_NAME}')
model.load_state_dict(state['model'])
model = model.cuda()
prior_precision = state['delta']

metrics_df = pd.DataFrame()
for m in [50, 200, 800, 1600]:
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

        metrics_df = metrics_df.append({'M': m, 'acc_laplace': acc_laplace, 'ece_laplace':ece_laplace, 'nll_laplace': nll_laplace}, ignore_index=True)

# TODO: plot
print(metrics_df)


