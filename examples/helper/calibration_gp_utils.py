import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import torch
import torch.distributions as dists
from netcal.metrics import ECE

from laplace import Laplace


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


def gp_calibration_eval(model, train_loader, test_loader, subset_of_weights='last_layer', M_arr=[10, 50, 100, 300, 500, 1000]) -> pd.DataFrame:
    targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
    metrics_df = pd.DataFrame()
    for m in M_arr:
        for seed in range(5):
            la = Laplace(model, 'classification',
                         subset_of_weights=subset_of_weights,
                         hessian_structure='gp',
                         diagonal_kernel=True, M=m, seed=seed)
            la.fit(train_loader)
            la.optimize_prior_precision(method='marglik')

            probs_laplace = predict(test_loader, la, laplace=True)
            acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
            ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
            nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

            print(m ,seed)
            print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

            metrics_df = metrics_df.append({'M': m, 'seed': seed, 'acc_laplace': acc_laplace,
                                            'ece_laplace':ece_laplace, 'nll_laplace': nll_laplace}, ignore_index=True)
    return metrics_df