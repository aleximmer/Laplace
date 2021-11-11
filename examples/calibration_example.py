import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import numpy as np
import helper.wideresnet as wrn
import helper.dataloaders as dl
from netcal.metrics import ECE
import urllib.request
import os.path

from laplace import Laplace


np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

train_loader = dl.CIFAR10(train=True)
test_loader = dl.CIFAR10(train=False)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()

# Download pre-trained model if necessary
if not os.path.isfile('CIFAR10_plain.pt'):
    if not os.path.exists('./temp'):
        os.makedirs('./temp')

    urllib.request.urlretrieve('https://nc.mlcloud.uni-tuebingen.de/index.php/s/2PBDYDsiotN76mq/download', './temp/CIFAR10_plain.pt')

model.load_state_dict(torch.load('./temp/CIFAR10_plain.pt'))


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu().numpy()


probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).mean()
ece_map = ECE(bins=15).measure(probs_map, targets)

print(f'[MAP] Acc.(🠕): {acc_map:.1%}; ECE(🠗): {ece_map:.1%}')

# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')
la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).mean()
ece_laplace = ECE(bins=15).measure(probs_laplace, targets)

print(f'[Laplace] Acc.(🠕): {acc_laplace:.1%}; ECE(🠗): {ece_laplace:.1%}')
