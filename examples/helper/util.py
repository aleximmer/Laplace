import urllib.request
import os.path
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
import torch.distributions as dists
from netcal.metrics import ECE


def download_pretrained_model():
    # Download pre-trained model if necessary
    if not os.path.isfile('CIFAR10_plain.pt'):
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        urllib.request.urlretrieve('https://nc.mlcloud.uni-tuebingen.de/index.php/s/2PBDYDsiotN76mq/download', './temp/CIFAR10_plain.pt')


def plot_regression(X_train, y_train, X_test, f_test, y_std, plot=True, 
                    file_name='regression_example', la_type='full'):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
                                figsize=(4.5, 2.8))
    ax1.set_title('MAP')
    ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
    ax1.plot(X_test, f_test, color='black', label='$f_{MAP}$')
    ax1.legend()

    ax2.set_title(f'LA-{la_type}')
    ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.3, color='tab:orange')
    ax2.plot(X_test, f_test, label='$\mathbb{E}[f]$')
    ax2.fill_between(X_test, f_test-y_std*2, f_test+y_std*2, 
                     alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[y]}$')
    ax2.legend()
    ax1.set_ylim([-4, 6])
    ax1.set_xlim([X_test.min(), X_test.max()])
    ax2.set_xlim([X_test.min(), X_test.max()])
    ax1.set_ylabel('$y$')
    ax1.set_xlabel('$x$')
    ax2.set_xlabel('$x$')
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(f'docs/{file_name}.png')


def predict(dataloader, model, laplace=False, la_type='kron'):
    with torch.no_grad() if (la_type != 'gp') else nullcontext():
        py = []

        for x, _ in dataloader:
            if laplace:
                py.append(model(x.cuda()))
            else:
                py.append(torch.softmax(model(x.cuda()), dim=-1))

        return torch.cat(py).cpu()


def get_metrics(probs, targets):
    acc = (probs.argmax(-1) == targets).float().mean().cpu().item()
    ece = ECE(bins=15).measure(probs.numpy(), targets.numpy())
    nll = -dists.Categorical(probs).log_prob(targets).mean().cpu().item()
    return acc, ece, nll
