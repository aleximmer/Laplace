import warnings
warnings.simplefilter("ignore", UserWarning)

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

assert torch.cuda.is_available()

DATASET = 'FMNIST'
BATCH_SIZE = 128
ds_train, ds_test = get_dataset(DATASET, False, 'cuda')
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
targets = torch.cat([y for x, y in test_loader], dim=0).cpu()

MODEL_NAME = 'FMNIST_CNN_10_2.2e+02.pt'
model = CIFAR10Net(ds_train.channels, ds_train.K, use_tanh=True).to('cuda')
state = torch.load(f'helper/models/{MODEL_NAME}')
model.load_state_dict(state['model'])
model = model.cuda()
prior_precision = state['delta']

probs_map = predict(test_loader, model, laplace=False)
acc_map, ece_map, nll_map = get_metrics(probs_map, targets)
print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

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
    acc_laplace, ece_laplace, nll_laplace = get_metrics(probs_laplace, targets)
    print(f'[Laplace-GP, m={m}] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

    metrics_df = metrics_df.append({'M': m, 'acc_laplace': acc_laplace, 'ece_laplace':ece_laplace, 'nll_laplace': nll_laplace},
                                   ignore_index=True)

# TODO: plot
print(metrics_df)


