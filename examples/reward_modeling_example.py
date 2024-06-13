import logging
import warnings

import numpy as np
import torch
import torch.utils.data as data_utils
from datasets import Dataset
from torch import nn, optim
from torch.nn import functional as F

from laplace import Laplace

logging.basicConfig(level="ERROR")
warnings.filterwarnings("ignore")

# make deterministic
torch.manual_seed(0)
np.random.seed(0)


# Pairwise comparison dataset. The label indicates which `x0` or `x1` is preferred.
data_dict = [
    {
        "x0": torch.randn(3),
        "x1": torch.randn(3),
        "label": torch.randint(2, size=(1,)).item(),
    }
    for _ in range(10)
]
dataset = Dataset.from_list(data_dict)


class SimpleRewardModel(nn.Module):
    """
    A simple reward model, compatible with the Bradley-Terry likelihood.
    During training, it assumes that `x` is a tensor of shape `(batch_size, 2, dim)`.
    The second dimension of size 2 is preserved through the forward pass, resulting in
    a logit tensor of shape `(batch_size, 2)` (the net itself is single-output).
    Then, the standard cross-entropy loss is applied.

    During testing, this model behaves like a standard single-output regression model.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 1))

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                If training == True then shape (batch_size, 2, dim)
                Else shape (batch_size, dim)

        Returns:
            logits: torch.Tensor
                If training then shape (batch_size, 2)
                Else shape (batch_size, 1)
        """
        if len(x.shape) == 3:
            batch_size, _, dim = x.shape

            # Flatten to (batch_size*2, dim)
            flat_x = x.reshape(-1, dim)

            # Forward
            flat_logits = self.net(flat_x)  # (batch_size*2, 1)

            # Reshape back to (batch_size, 2)
            return flat_logits.reshape(batch_size, 2)
        else:
            logits = self.net(x)  # (batch_size, 1)
            return logits


# Preprocess to coalesce x0 and x1 into a single array/tensor
def append_x0_x1(row):
    # The tensor values above are automatically casted as lists by `Dataset`
    row["x"] = np.stack([row["x0"], row["x1"]])  # (2, dim)
    return row


tensor_dataset = dataset.map(append_x0_x1, remove_columns=["x0", "x1"])
tensor_dataset.set_format(type="torch", columns=["x", "label"])
tensor_dataloader = data_utils.DataLoader(
    data_utils.TensorDataset(tensor_dataset["x"], tensor_dataset["label"]), batch_size=3
)

reward_model = SimpleRewardModel()
opt = optim.AdamW(reward_model.parameters(), weight_decay=1e-3)

# Train as usual
for epoch in range(10):
    for x, y in tensor_dataloader:
        opt.zero_grad()
        out = reward_model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()

# Laplace !!! Notice the likelihood !!!
reward_model.eval()
la = Laplace(reward_model, likelihood="reward_modeling", subset_of_weights="all")
la.fit(tensor_dataloader)
la.optimize_prior_precision()

x_test = torch.randn(5, 3)
pred_mean, pred_var = la(x_test)
print(
    f"Input shape {tuple(x_test.shape)}, predictive mean of shape "
    + f"{tuple(pred_mean.shape)}, predictive covariance of shape "
    + f"{tuple(pred_var.shape)}"
)
