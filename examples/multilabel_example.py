# %% [markdown]
# # Multi-Label Classification
#
# In multi-label classification, each input has several binary labels that need not
# exclude each other, e.g. the objects in an image or the tags of a document. The
# network outputs one logit per label, and a sigmoid turns each logit into the
# probability of its label. The matching loss is `nn.BCEWithLogitsLoss`, which is the
# negative log likelihood of a `BernoulliLikelihood`.
#
# We train a MAP network on synthetic multi-label data and put a Laplace
# approximation (LA) on top of it. `Laplace` accepts the criterion you trained with.
# We tune the prior precision with the marginal likelihood and compare two ways to
# get the predictive.
#
# First, we create the data: points in the plane with 3 labels, each label a noisy
# function of the position.

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace, marglik_training
from laplace.utils import RunningBernoulliNLLMetric

torch.manual_seed(711)


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty(n, 2).uniform_(-2, 2)

    logits = torch.stack(
        [
            3 * X[:, 0],  # a vertical boundary
            3 * X[:, 1] - 1,  # a horizontal boundary
            4 * (X.norm(dim=1) - 1.5),  # a circle
        ],
        dim=1,
    )
    y = torch.bernoulli(torch.sigmoid(logits))

    return X, y


X_train, y_train = make_data(300)
X_val, y_val = make_data(200)
X_test, y_test = make_data(500)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=100)

# %% [markdown]
# ## Training a MAP
#
# The network returns logits, so it has no sigmoid at the end. We train it with plain
# PyTorch and `nn.BCEWithLogitsLoss`:


# %%
def get_model() -> nn.Module:
    torch.manual_seed(711)
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 3))


model = get_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(300):
    for X, y in train_loader:
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()

# %% [markdown]
# We measure how well the predicted probabilities fit the test labels with the
# negative log likelihood (NLL), summed over the 3 labels and averaged over the test
# points:


# %%
def nll(probs: torch.Tensor, y: torch.Tensor) -> float:
    return nn.functional.binary_cross_entropy(probs, y, reduction="sum").item() / len(y)


with torch.no_grad():
    nll_map = nll(torch.sigmoid(model(X_test)), y_test)

print(f"MAP test NLL: {nll_map:.3f}")

# %% [markdown]
# ## Fitting the LA
#
# We pass the criterion to `Laplace`, which turns it into a `BernoulliLikelihood`
# with the same options. We use the Kronecker-factored Hessian, the library default,
# computed with the generalized Gauss-Newton of Curvlinops.
#
# The marginal likelihood tunes the prior precision without validation data:

# %%
la = Laplace(model, criterion, subset_of_weights="all", hessian_structure="kron")
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

print(f"prior precision: {la.prior_precision.item():.2f}")

# %% [markdown]
# ## Predictive
#
# The LA predicts the probability of each label. For an input $x$, the GLM predictive
# is a Gaussian over the logits, $\mathcal{N}(f_\mu, \Sigma_f)$, and the predicted
# probability of label $k$ is $\mathbb{E}[\sigma(f_k)]$ under it.
#
# The default `link_approx="probit"` is a closed-form approximation. It scales each
# logit by its marginal variance,
#
# $$
#     \sigma\!\left(\frac{f_{\mu,k}}{\sqrt{1 + \frac{\pi}{8} (\Sigma_f)_{kk}}}\right),
# $$
#
# so there is no sampling. With `link_approx="mc"`, the LA draws samples from the
# Gaussian and averages the sigmoid over them. The Laplace bridge only exists for
# the softmax, so `"bridge"` raises an error here.

# %%
probs_probit = la(X_test)
probs_mc = la(X_test, link_approx="mc", n_samples=1000)

print(f"LA test NLL (probit): {nll(probs_probit, y_test):.3f}")
print(f"LA test NLL (MC): {nll(probs_mc, y_test):.3f}")
print(f"largest difference: {(probs_probit - probs_mc).abs().max().item():.3f}")

# %% [markdown]
# The two approximations agree closely, and the probit one needs no samples. The 3
# probabilities of an input don't sum to one, since the labels are independent.
#
# ## Tuning on validation data
#
# If you have validation data, grid search tunes the prior precision directly for the
# predictive. For the Bernoulli likelihood, the default loss is the NLL from
# `RunningBernoulliNLLMetric`, which sums over the labels and averages over the
# points:

# %%
la.optimize_prior_precision(method="gridsearch", val_loader=val_loader, grid_size=20)

print(f"prior precision (grid search): {la.prior_precision.item():.2f}")
print(f"LA test NLL (probit): {nll(la(X_test), y_test):.3f}")

# %% [markdown]
# `optimize_prior_precision` also takes another `loss`, e.g. a `torchmetrics` metric.
# The metric above works on its own too:

# %%
metric = RunningBernoulliNLLMetric()
metric.update(la(X_val), y_val)

print(f"LA validation NLL: {metric.compute().item():.3f}")

# %% [markdown]
# ## Options and soft labels
#
# `nn.BCEWithLogitsLoss` has two options, `weight` and `pos_weight`, e.g. to
# upweight the rare positives of a label. Pass the criterion with its options, or
# build `BernoulliLikelihood(weight=..., pos_weight=...)`. They work with the `diag`
# and `full` structures. The Kronecker factors of the generalized Gauss-Newton can't
# see them yet, so `kron` raises an error instead of dropping them silently, see the
# [Curvlinops issue](https://github.com/f-dangel/curvlinops/issues/311). The
# empirical Fisher backend `CurvlinopsEF` supports them with `kron`.
#
# As in PyTorch, the targets can be soft labels in $[0, 1]$, e.g. from label
# smoothing or from several annotators. Pass them like any other target. Here we
# shrink the training labels towards 0.5:

# %%
y_soft = 0.9 * y_train + 0.05
soft_loader = DataLoader(TensorDataset(X_train, y_soft), batch_size=100)

la_soft = Laplace(model, criterion, subset_of_weights="all", hessian_structure="kron")
la_soft.fit(soft_loader)

print(f"LA test NLL with soft labels: {nll(la_soft(X_test), y_test):.3f}")

# %% [markdown]
# ## Jointly optimizing the MAP and hyperparameters
#
# The `marglik_training` utility alternates between training the network and tuning
# the prior precision with the marginal likelihood. It returns the trained model
# and the LA. The default prior is layerwise, so we get one precision per layer
# parameter:

# %%
la_joint, model_joint, margliks, losses = marglik_training(
    get_model(),
    train_loader,
    likelihood=nn.BCEWithLogitsLoss(),
    hessian_structure="kron",
    n_epochs=100,
    optimizer_kwargs={"lr": 1e-2},
    n_epochs_burnin=50,
)

print(
    "joint prior precisions:", [f"{p:.2f}" for p in la_joint.prior_precision.tolist()]
)
print(f"joint LA test NLL: {nll(la_joint(X_test), y_test):.3f}")

# %% [markdown]
# ## Sampling the network weights
#
# `pred_type="nn"` samples the network weights instead of linearizing the network.
# It needs `link_approx="mc"`. We use the LA from the grid search above:

# %%
probs_nn = la(X_test, pred_type="nn", link_approx="mc", n_samples=100)

print(f"LA test NLL (NN sampling): {nll(probs_nn, y_test):.3f}")

# %% [markdown]
# Other structures and subsets of weights only change the arguments of `Laplace`:
# `last_layer` and `subnetwork` work like `all`. The function-space methods
# (`hessian_structure="gp"`) don't support the Bernoulli likelihood yet.
