import torch
import wideresnet
import dataloaders as dl
from pycalib.scoring import expected_calibration_error

from laplace import Laplace


train_loader = dl.CIFAR10(train=True)
test_loader = dl.CIFAR10(train=False)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

model = wideresnet.WideResNet(16, 4, num_classes=10)


def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.stack(py)


conf_map = predict(test_loader, model, laplace=False).max(dim=-1)
ece_map = expected_calibration_error(targets, conf_map)

print(f'ECE of the MAP model (lower is better): {ece_map}')

# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')
la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

conf_laplace = predict(test_loader, la, laplace=True).max(dim=-1)
ece_laplace = expected_calibration_error(targets, conf_laplace)

print(f'ECE of the Laplace model (lower is better): {ece_laplace}')
