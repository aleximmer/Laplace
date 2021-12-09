import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils import parameters_to_vector

from laplace import Laplace
from laplace.curvature import AsdlGGN
from helper.dataloaders import ToydataGenerator
from helper.mlp import MLP

plt.rcParams['text.usetex'] = True


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    # Generate toy data data loaders
    num_tasks = 5
    datagen = ToydataGenerator(max_iter=num_tasks, num_samples=2000)

    # Create model
    model = MLP([2, 20, 20, 2], act='sigmoid').to(DEVICE)

    # Initialize Laplace approximation
    la = Laplace(
        model, 'classification',
        subset_of_weights='all',
        hessian_structure='diag',
        prior_precision=1e-4,
        backend=AsdlGGN)

    # iterate over all tasks
    for task_id in range(num_tasks):
        # Get data loaders for current task
        train_loader, test_loader = datagen.next_task()

        # Train on current task
        train(task_id, model, la, train_loader)

        # Fit Laplace approximation on current task
        la.fit(train_loader, override=False)

    # Predict on test data (a 2D grid of points for plotting)
    cl_outputs = list()
    for X, _ in test_loader:
        cl_outputs.append(la(X.to(DEVICE)))
    cl_outputs = torch.cat(cl_outputs, dim=0)

    # Plot visualisation (2D figure)
    cl_outputs, _ = torch.max(cl_outputs, dim=-1)
    cl_show = 2*cl_outputs - 1
    cl_show = cl_show.detach().cpu().numpy().reshape(datagen.test_shape)

    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    plt.figure()
    plt.imshow(cl_show, cmap='gray', origin='lower')
    for t in range(task_id+1):
        idx = np.where(datagen.y == t)
        plt.scatter(
            datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[t], s=0.03)
        idx = np.where(datagen.y == t+datagen.offset)
        plt.scatter(
            datagen.X[idx][:, 0], datagen.X[idx][:, 1], c=color[t+datagen.offset], s=0.03)
    plt.xlim(datagen.x_min, datagen.x_max)
    plt.ylim(datagen.y_min, datagen.y_max)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.tight_layout()
    plt.savefig('docs/continual_learning_example.png', dpi=300)


def train(task_id, model, la, train_loader):
    # Create loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Set up differentiable inital prior precision
    log_prior_prec = np.log(1e-4) * torch.ones(1, device=DEVICE)
    log_prior_prec.requires_grad = True
    # Set up optimizer for marginal likelihood optimization
    hyper_optimizer = torch.optim.Adam([log_prior_prec], lr=0.01)
    # Accumulated Hessian approximations over all tasks up to current one
    prior_prec_offset = la.H.clone()

    N = len(train_loader.dataset)
    model.train()
    for epoch in range(50):
        train_loss = 0.
        for X, y in train_loader:
            f = model(X.to(DEVICE))

            # Subtract log prior from loss function
            mean = parameters_to_vector(model.parameters())
            loss = loss_fn(f, y.to(DEVICE)) - la.log_prob(mean) / N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X)

        train_loss /= N

        if (epoch + 1) % 5 != 0:
            continue

        # Fit Laplace approximation for marginal likelihood optimization
        hyper_la = Laplace(
            model, 'classification',
            subset_of_weights='all',
            hessian_structure='diag',
            prior_mean=la.mean,
            prior_precision=la.posterior_precision,
            backend=AsdlGGN)
        hyper_la.fit(train_loader)
        
        # Optimize the initial prior precision via marginal likelihood optimization
        for _ in range(100):
            hyper_optimizer.zero_grad()
            prior_prec_init = torch.exp(log_prior_prec)
            prior_prec = prior_prec_init + prior_prec_offset
            marglik = -hyper_la.log_marginal_likelihood(prior_prec)
            marglik.backward()
            hyper_optimizer.step()
        marglik = marglik.item() / N

        print(f'Task {task_id+1} epoch {epoch+1} - train loss: {train_loss:.3f}, neg. log marglik: {marglik:.3f}')


if __name__ == "__main__":
    main()
