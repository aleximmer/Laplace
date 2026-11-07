# %% [markdown]
# # Soft Labels
#
# A classification target doesn't have to be a single class. It can be a probability
# vector over the classes, or a soft label. `nn.CrossEntropyLoss` takes these targets
# when they are floats with the shape of the logits, and `Laplace` takes them in the
# same way. So if your training loop already works with soft labels, the same data
# loader works for `la.fit`.
#
# We go through three scenarios: label noise, knowledge distillation, and sequences
# with a soft label per position.
#
# Let's start with the imports and some toy data: points in the plane with 3 classes.

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace

torch.manual_seed(711)

N_CLASSES = 3


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.empty(n, 2).uniform_(-2, 2)

    angle = torch.atan2(X[:, 1], X[:, 0])
    centers = torch.tensor([0.0, 2.0, -2.0])
    logits = 3 * torch.cos(angle[:, None] - centers)

    return X, torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


def get_model(width: int = 32) -> nn.Module:
    return nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, N_CLASSES))


def train(
    model: nn.Module,
    loader: DataLoader,
    n_epochs: int = 100,
    label_smoothing: float = 0.0,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(n_epochs):
        for X, y in loader:
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

    return model.eval()


X_train, y_train = make_data(300)
X_val, y_val = make_data(200)
X_test, _ = make_data(8)

# %% [markdown]
# ## Label noise
#
# Suppose an annotator gives the wrong class with probability 0.2, with every wrong
# class equally likely. An observed label is then not certain. The true class is the
# observed one with probability 0.8, and each of the others with probability 0.1. We
# use these probabilities as the targets. They must be floats, as in
# `nn.CrossEntropyLoss`.


# %%
def noisy_label_targets(y: torch.Tensor, flip_prob: float) -> torch.Tensor:
    one_hot = nn.functional.one_hot(y, N_CLASSES).float()

    return (1 - flip_prob) * one_hot + (1 - one_hot) * flip_prob / (N_CLASSES - 1)


y_noisy = noisy_label_targets(y_train, flip_prob=0.2)
noisy_loader = DataLoader(TensorDataset(X_train, y_noisy), batch_size=100)

print(y_train[:2])
print(y_noisy[:2])

# %% [markdown]
# ```
# tensor([0, 2])
# tensor([[0.8000, 0.1000, 0.1000],
#         [0.1000, 0.1000, 0.8000]])
# ```
#
# We train the network as usual and pass the criterion to `Laplace`, which turns it
# into a `CategoricalLikelihood`. The rest is the same as with class indices. Each row
# of `y_noisy` counts as one observation, so the class dim of the targets doesn't add
# any.

# %%
model = train(get_model(), noisy_loader)

la = Laplace(
    model,
    nn.CrossEntropyLoss(),
    subset_of_weights="last_layer",
    hessian_structure="full",
)
la.fit(noisy_loader)
la.optimize_prior_precision()

print(f"Observations: {la.n_obs}")
print(f"Predictive: {la(X_test).shape}")

# %% [markdown]
# ```
# Observations: 300
# Predictive: torch.Size([8, 3])
# ```
#
# If you'd rather spread the noise evenly over all classes, keep the class indices and
# use `label_smoothing`. Train with it, and pass the same criterion to `Laplace`, which
# takes the option from there.

# %%
index_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100)
smoothed_model = train(get_model(), index_loader, label_smoothing=0.1)

la = Laplace(
    smoothed_model,
    nn.CrossEntropyLoss(label_smoothing=0.1),
    subset_of_weights="last_layer",
    hessian_structure="kron",
)
la.fit(index_loader)

print(f"Label smoothing: {la.likelihood.label_smoothing}")

# %% [markdown]
# ```
# Label smoothing: 0.1
# ```
#
# ## Knowledge distillation
#
# In knowledge distillation, a small student network learns from the predictions of a
# larger teacher network. Here, the targets are the teacher's probabilities. We train
# the teacher on the class indices, then compute the student's targets once.

# %%
teacher = train(get_model(width=128), index_loader)


@torch.no_grad()
def teacher_targets(X: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    return (teacher(X) / temperature).softmax(-1)


student_loader = DataLoader(
    TensorDataset(X_train, teacher_targets(X_train)), batch_size=100
)
student = train(get_model(width=16), student_loader)

# %% [markdown]
# Then we fit `Laplace` on the student with the same loader. This time we tune the
# prior precision with a grid search on validation data. The validation targets can be
# soft too, and the default loss of the grid search, `RunningNLLMetric`, takes them.

# %%
val_loader = DataLoader(TensorDataset(X_val, teacher_targets(X_val)), batch_size=100)

la = Laplace(
    student,
    nn.CrossEntropyLoss(),
    subset_of_weights="all",
    hessian_structure="kron",
)
la.fit(student_loader)
la.optimize_prior_precision(
    method="gridsearch",
    pred_type="glm",
    link_approx="probit",
    val_loader=val_loader,
    progress_bar=False,
)

print(f"Predictive: {la(X_test).shape}")

# %% [markdown]
# ```
# Predictive: torch.Size([8, 3])
# ```
#
# ## Sequences
#
# For a model with logits `(batch, length, classes)`, the targets are probabilities of
# the same shape, so each position has its own soft label. Every position is an
# observation. Here, we reuse the noisy data above as 60 sequences of length 5. Linear
# layers act on each position separately, so `model`, which we trained on the single
# points, works for the sequences.

# %%
X_seq = X_train.reshape(60, 5, 2)
y_seq = y_noisy.reshape(60, 5, N_CLASSES)
seq_loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=20)

la = Laplace(
    model,
    nn.CrossEntropyLoss(),
    subset_of_weights="all",
    hessian_structure="kron",
)
la.fit(seq_loader)
la.optimize_prior_precision()

print(f"Targets: {y_seq.shape}")
print(f"Observations: {la.n_obs}")

# %% [markdown]
# ```
# Targets: torch.Size([60, 5, 3])
# Observations: 300
# ```
#
# Sequences need a backend that supports them, such as the default `CurvlinopsGGN`.
# BackPACK and last-layer Laplace don't take sequence outputs.
#
# ## One-hot targets
#
# One-hot targets are just soft labels with all the mass on one class, but they must be
# floats. `torch.nn.functional.one_hot` returns integers, which `nn.CrossEntropyLoss`
# rejects, and so does `Laplace`. Call `.float()` on them first.

# %%
y_one_hot = nn.functional.one_hot(y_train, N_CLASSES)
one_hot_data = TensorDataset(X_train, y_one_hot)

try:
    Laplace(model, nn.CrossEntropyLoss(), hessian_structure="diag").fit(
        DataLoader(one_hot_data, batch_size=100)
    )
except ValueError as error:
    print(error)

one_hot_data = TensorDataset(X_train, y_one_hot.float())
Laplace(model, nn.CrossEntropyLoss(), hessian_structure="diag").fit(
    DataLoader(one_hot_data, batch_size=100)
)

# %% [markdown]
# ```
# The targets have the shape of class probabilities, but their dtype is torch.int64.
# Call `.float()` on one-hot targets, or pass class indices without the class dim,
# i.e. with 1 dim.
# ```
#
# > **Note:** The function-space methods `ELLA` and `VaLLA` take hard labels only, and
# > class probabilities raise a `ValueError`.
