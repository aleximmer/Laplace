import torch
import torch.utils.data as data_utils
from torch import autograd, nn, optim

from laplace import Laplace
from laplace.utils.enums import HessianStructure, Likelihood, PredType, SubsetOfWeights

torch.manual_seed(123)

model = nn.Sequential(nn.Linear(2, 10), nn.GELU(), nn.Linear(10, 1))
X, Y = torch.randn(5, 2), torch.randn(5, 1)
train_loader = data_utils.DataLoader(data_utils.TensorDataset(X, Y), batch_size=3)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()

    for x, y in train_loader:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

la = Laplace(
    model,
    Likelihood.REGRESSION,
    subset_of_weights=SubsetOfWeights.ALL,
    hessian_structure=HessianStructure.KRON,
    enable_backprop=True,
)
la.fit(train_loader)
la.optimize_prior_precision(PredType.GLM)

# Thompson sampling
for pred_type in [PredType.GLM, PredType.NN]:
    print(f"Thompson sampling, {pred_type}")

    x_test = torch.randn(10, 2)
    x_test.requires_grad = True

    f_sample = la.functional_samples(x_test, pred_type=pred_type, n_samples=1)
    f_sample = f_sample.squeeze(0)  # We only use a single sample
    print(f"TS shape: {f_sample.shape}, TS requires grad: {f_sample.requires_grad}")

    # Get the gradient of the Thompson sample w.r.t. input x.
    # Summed since it doesn't change the grad and autograd requires a scalar function.
    grad_x = autograd.grad(f_sample.sum(), x_test)[0]

    print(
        f"Grad x_test shape: {grad_x.shape}, Grad x_test vanishing: {torch.allclose(grad_x, torch.tensor(0.0))}"
    )
    print()

print()

# Monte-Carlo expected improvement (EI): E_{f(x) ~ p(f(x) | D)} [max(f(x) - best_f, 0)]
f_best = torch.tensor(0.123)  # Arbitrary in this example

for pred_type in [PredType.GLM, PredType.NN]:
    print(f"MC-EI, {pred_type}")

    x_test = torch.randn(10, 2)
    x_test.requires_grad = True

    f_samples = la.functional_samples(x_test, pred_type=pred_type, n_samples=10)
    ei = (f_samples - f_best.reshape(1, 1, 1)).clamp(0.0).mean(0)
    print(f"EI shape: {ei.shape}, EI requires grad: {ei.requires_grad}")

    # Get the gradient of the EI w.r.t. input x.
    # Summed since it doesn't change the grad and autograd requires a scalar function.
    grad_x = autograd.grad(ei.sum(), x_test)[0]

    print(
        f"Grad x_test shape: {grad_x.shape}, Grad x_test vanishing: {torch.allclose(grad_x, torch.tensor(0.0))}"
    )
    print()
