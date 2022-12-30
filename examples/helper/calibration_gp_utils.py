import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import torch
import torch.distributions as dists
from netcal.metrics import ECE
import wandb
from laplace import Laplace
import matplotlib.pyplot as plt
import helper.wideresnet as wrn
from  helper import util
from helper.datasets import get_dataset
from helper.models import get_model
from torch.utils.data import DataLoader
import helper.dataloaders as dl


# @torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cuda()))
        else:
            py.append(torch.softmax(model(x.cuda()), dim=-1))

    return torch.cat(py).cpu()


def gp_calibration_eval(model, train_loader, test_loader, subset_of_weights='last_layer',
                        M_arr=[10, 50, 100, 300, 500, 1000], prior_precision=1.0, optimize_prior_precision=True) -> pd.DataFrame:
    targets = torch.cat([y for x, y in test_loader], dim=0).cpu()
    metrics_df = pd.DataFrame()
    for m in M_arr:
        for seed in range(5):
            la = Laplace(model, 'classification',
                         subset_of_weights=subset_of_weights,
                         hessian_structure='gp',
                         diagonal_kernel=True, M=m,
                         seed=seed, prior_precision=prior_precision)
            la.fit(train_loader)
            if optimize_prior_precision:
                la.optimize_prior_precision(method='marglik')

            probs_laplace = predict(test_loader, la, laplace=True)
            acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
            ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
            nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

            print(m ,seed)
            print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')

            metrics_df = metrics_df.append({'M': m, 'seed': seed, 'acc_laplace': acc_laplace,
                                            'ece_laplace':ece_laplace, 'nll_laplace': nll_laplace}, ignore_index=True)
    metrics_df["acc_laplace"] = metrics_df["acc_laplace"].apply(lambda x: x.cpu().item())
    metrics_df["nll_laplace"] = metrics_df["nll_laplace"].apply(lambda x: x.cpu().item())

    return metrics_df


def gp_calibration_eval_wandb(model, train_loader, test_loader, wandb_kwargs,
                              subset_of_weights='last_layer', M_arr=[10, 50, 100, 300, 500, 1000],
                              prior_precision=1.0, optimize_prior_precision=True) -> pd.DataFrame:
    with wandb.init(**wandb_kwargs) as run:
        metrics_gp = gp_calibration_eval(model=model, train_loader=train_loader,
                                         test_loader=test_loader, subset_of_weights=subset_of_weights,
                                         M_arr=M_arr, prior_precision=prior_precision,
                                         optimize_prior_precision=optimize_prior_precision)
        run.log({'metrics': wandb.Table(dataframe=metrics_gp)})

        for metric in ["acc_laplace", "ece_laplace", "nll_laplace"]:
            fig, ax = plt.subplots()
            metrics_gp.groupby(by="M")[[metric]].mean().rename({metric: "GPLaplace"}, axis=1).plot(style="-bo", ax=ax)
            ax.set_title(metric[:3])
            run.log({metric: wandb.Image(fig)})


def load_model(repo: str, dataset: str, train_data):
    if repo == "BNN-preds":
        if dataset == "FMNIST":
            model = get_model('CNN', train_data).to('cuda')
            # state = torch.load("helper/models/FMNIST_CNN_117_4.6e-01.pt")
            state = torch.load("helper/models/FMNIST_CNN_117_1.0e+01.pt")
            prior_precision = 10.

        elif dataset == "CIFAR10":
            model = get_model('AllCNN', train_data).to('cuda')
            state = torch.load("helper/models/CIFAR10_AllCNN_117_1.0e+01.pt")
            prior_precision = 1.
        else:
            raise ValueError()
        model.load_state_dict(state['model'])
        model = model.cuda()
    else:
        assert dataset == "CIFAR10"
        # The model is a standard WideResNet 16-4
        # Taken as is from https://github.com/hendrycks/outlier-exposure
        model = wrn.WideResNet(16, 4, num_classes=10).cuda().eval()
        # print( sum(p.numel() for p in model.parameters()))

        util.download_pretrained_model()
        model.load_state_dict(torch.load('./temp/CIFAR10_plain.pt'))
        prior_precision = 1.
    return model, prior_precision


def load_data(repo: str, dataset: str):
    if repo == "BNN-preds":
        ds_train, ds_test = get_dataset(dataset, False, 'cuda')
        train_loader = DataLoader(ds_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(ds_test, batch_size=100, shuffle=False)
    elif repo == "Laplace":
        assert dataset == "CIFAR10"
        train_loader = dl.CIFAR10(train=True)
        test_loader = dl.CIFAR10(train=False)
        ds_train = None
    else:
        raise ValueError()
    return train_loader, test_loader, ds_train