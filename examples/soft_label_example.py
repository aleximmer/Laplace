# %% [markdown]
# # Soft Labels
#
# Classification targets are often not a single class. Several annotators can
# disagree on an image, a teacher model can pass on its probabilities, and label
# smoothing spreads some mass over the other classes. `nn.CrossEntropyLoss` takes such
# targets as class probabilities with the shape of the logits, and so does `Laplace`.
#
# We train a MAP network on synthetic data where five annotators label each point,
# so the target of a point is the fraction of votes per class. Then we put a Laplace
# approximation (LA) on top of it, tune its prior precision, and compare the soft
# negative log likelihood (NLL) of the MAP and the LA.
#
# First, we create the data: points in the plane with 3 classes. The annotators
# sample a class from the true class probabilities of the point, so they disagree
# most near the class boundaries.

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.utils import RunningNLLMetric

torch.manual_seed(711)

N_CLASSES, N_ANNOTATORS = 3, 5


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty(n, 2).uniform_(-2, 2)

    angle = torch.atan2(X[:, 1], X[:, 0])
    centers = torch.tensor([0.0, 2.0, -2.0])
    logits = 3 * torch.cos(angle[:, None] - centers)

    # One vote per annotator, then the fraction of votes per class
    votes = torch.multinomial(logits.softmax(-1), N_ANNOTATORS, replacement=True)
    y = nn.functional.one_hot(votes, N_CLASSES).float().mean(dim=1)

    return X, y


X_train, y_train = make_data(300)
X_val, y_val = make_data(200)
X_test, y_test = make_data(500)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=100)

print("targets of the first points:")
print(y_train[:4])

# %% [markdown]
# Each row is the fraction of votes per class, so the rows sum to one:
#
# ```
# tensor([[0.4000, 0.6000, 0.0000],
#         [0.0000, 0.2000, 0.8000],
#         [1.0000, 0.0000, 0.0000],
#         [0.0000, 0.4000, 0.6000]])
# ```
#
# ## Training a MAP
#
# The targets `y` have shape `(batch, 3)` and a floating-point dtype. That is all
# `nn.CrossEntropyLoss` needs to read them as class probabilities instead of class
# indices of shape `(batch,)`. We train the network with plain PyTorch:


# %%
def get_model() -> nn.Module:
    torch.manual_seed(711)
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, N_CLASSES))


model = get_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(300):
    for X, y in train_loader:
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()

# %% [markdown]
# We measure the fit with the NLL of the predicted probabilities under the soft test
# targets, averaged over the test points. `RunningNLLMetric` accepts both class
# indices and class probabilities. With soft targets, the NLL cannot go below the
# entropy of the targets, so even a perfect model has a positive NLL.


# %%
def soft_nll(probs: torch.Tensor, y: torch.Tensor) -> float:
    metric = RunningNLLMetric()
    metric.update(probs, y)

    return metric.compute().item()


with torch.no_grad():
    nll_map = soft_nll(model(X_test).softmax(-1), y_test)

entropy = -torch.special.xlogy(y_test, y_test).sum(-1).mean().item()

print(f"MAP test NLL: {nll_map:.3f}")
print(f"entropy of the test targets: {entropy:.3f}")

# %% [markdown]
# ```
# MAP test NLL: 0.397
# entropy of the test targets: 0.271
# ```
#
# ## Fitting the LA
#
# We pass the criterion to `Laplace`, which turns it into a `CategoricalLikelihood`.
# The loader yields the same soft targets. Every point counts as one observation,
# and the class dimension of the targets does not. The curvature of the generalized
# Gauss-Newton (GGN) does not depend on the targets, so only the loss and the
# marginal likelihood see the soft labels.
#
# The marginal likelihood tunes the prior precision without validation data:

# %%
la = Laplace(
    model,
    nn.CrossEntropyLoss(reduction="sum"),
    subset_of_weights="all",
    hessian_structure="kron",
)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")

print(f"observations: {la.n_obs}")
print(f"prior precision (marglik): {la.prior_precision.item():.2f}")
print(f"LA test NLL (marglik): {soft_nll(la(X_test), y_test):.3f}")

# %% [markdown]
# ```
# observations: 300
# prior precision (marglik): 0.57
# LA test NLL (marglik): 0.517
# ```
#
# The 300 targets of shape `(3,)` are 300 observations. The LA with this prior
# precision is less confident than the MAP and has a higher test NLL. The MAP is
# already close to what the votes allow here, so there is little to gain from the
# posterior.
#
# ## Tuning on validation data
#
# Grid search takes validation targets of the same kind. Its default loss is
# `RunningNLLMetric`, which reads soft validation targets as class probabilities. The
# validation loader above already yields them:

# %%
la.optimize_prior_precision(method="gridsearch", val_loader=val_loader, grid_size=20)

print(f"prior precision (grid search): {la.prior_precision.item():.2f}")
print(f"LA test NLL (grid search): {soft_nll(la(X_test), y_test):.3f}")

# %% [markdown]
# ```
# prior precision (grid search): 10000.00
# LA test NLL (grid search): 0.397
# ```
#
# The soft validation NLL prefers the largest prior precision of the default grid,
# which gives back the NLL of the MAP.
#
# ## One-hot targets need `.float()`
#
# One-hot targets work too, as long as they are floats. Integer targets with the
# shape of the logits are rejected, as in PyTorch, since they are most likely a
# one-hot encoding that was not converted:

# %%
y_one_hot = nn.functional.one_hot(y_train.argmax(-1), N_CLASSES)

try:
    Laplace(model, criterion, subset_of_weights="all").fit(
        DataLoader(TensorDataset(X_train, y_one_hot), batch_size=100)
    )
except ValueError as error:
    print(error)

# %% [markdown]
# ```
# The targets have the shape of class probabilities, but their dtype is torch.int64.
# Call `.float()` on one-hot targets, or pass class indices without the class dim,
# i.e. with 1 dim.
# ```
#
# With `.float()`, one-hot targets give the same LA as the class indices:

# %%
y_index = y_train.argmax(-1)
las = {}

for name, targets in [("index", y_index), ("one-hot", y_one_hot.float())]:
    la_ = Laplace(model, criterion, subset_of_weights="all", hessian_structure="kron")
    la_.fit(DataLoader(TensorDataset(X_train, targets), batch_size=100))
    las[name] = la_

print(f"loss (index): {las['index'].loss.item():.3f}")
print(f"loss (one-hot): {las['one-hot'].loss.item():.3f}")

# %% [markdown]
# ```
# loss (index): 58.502
# loss (one-hot): 58.502
# ```
#
# ## What is supported
#
# Class probabilities work with the parametric structures (`full`, `kron`, `diag`),
# the last-layer and subnetwork variants, and the Curvlinops, ASDL, and BackPACK
# backends. For sequence models with logits `(batch, length, classes)`, the targets
# are probabilities `(batch, length, classes)`, and every position is an
# observation. BackPACK and the last-layer variant have no sequences. The `weight` and `label_smoothing` options of the criterion apply to
# soft targets as they do in PyTorch. `ignore_index` does not, since a position of
# probabilities cannot be ignored.
#
# > **Note:** The function-space methods `ELLA` and `VaLLA` reduce class
# > probabilities to hard labels with an `argmax`, so they don't treat soft labels
# > properly yet.
