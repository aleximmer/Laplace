import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from laplace import Laplace
from laplace.curvature import BackPackEF

n_epochs = 1000
batch_size = 150  # full batch
true_sigma_noise = 0.3
torch.manual_seed(711)

# create simple sinusoid data set
X_train = (torch.rand(150) * 8).unsqueeze(-1)
y_train = torch.sin(X_train) + torch.randn_like(X_train) * true_sigma_noise
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)  # +-5 on top of the training X-range

# Val-loader
X_val = (torch.rand(50) * 8).unsqueeze(-1)
y_val = torch.sin(X_val) + torch.randn_like(X_val) * true_sigma_noise
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# create and train MAP model
model = torch.nn.Sequential(torch.nn.Linear(1, 50),
                            torch.nn.Tanh(),
                            torch.nn.Linear(50, 1))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-2)
for i in range(n_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

la = Laplace(model, 'regression', subset_of_weights='all', hessian_structure='full', backend=BackPackEF)
la.fit(train_loader)

# Cross validation
la.optimize_prior_precision(
    'CV', val_loader=val_loader, loss=F.mse_loss,
    pred_type='nn', link_approx='mc',
    log_prior_prec_min=-6, log_prior_prec_max=6, grid_size=1000
)
la.sigma_noise = true_sigma_noise

x = X_test.flatten().cpu().numpy()
f_mu, f_var = la(X_test, pred_type='nn', link_approx='mc')
f_mu = f_mu.squeeze().detach().cpu().numpy()
f_sigma = f_var.squeeze().sqrt().cpu().numpy()
pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,
                               figsize=(4.5, 2.8))
ax1.set_title('MAP')
ax1.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax1.plot(x, f_mu, color='black', label='$f_{MAP}$')
ax1.legend()

ax2.set_title('LA')
ax2.scatter(X_train.flatten(), y_train.flatten(), alpha=0.7, color='tab:orange')
ax2.plot(x, f_mu, label='$\mathbb{E}[f]$')
ax2.fill_between(x, f_mu-pred_std*2, f_mu+pred_std*2,
                 alpha=0.3, color='tab:blue', label='$2\sqrt{\mathbb{V}\,[f]}$')
ax2.legend()
ax1.set_ylim([-4, 6])
ax1.set_xlim([x.min(), x.max()])
ax2.set_xlim([x.min(), x.max()])
ax1.set_ylabel('$y$')
ax1.set_xlabel('$x$')
ax2.set_xlabel('$x$')
plt.tight_layout()
plt.savefig('docs/regression_example_mc.png', dpi=300)
plt.show()
