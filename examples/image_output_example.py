from collections.abc import MutableMapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from laplace import Laplace
from laplace.curvature import CurvlinopsGGN

torch.manual_seed(0)

N_DATA = 64
CHANNELS = 3  # C
HEIGHT = 8  # H
WIDTH = 8  # W


class Denoiser(nn.Module):
    """
    Toy diffusion denoiser. It predicts the noise `eps` from a noisy image `x_t`
    and the diffusion time `t`, in PyTorch's usual `(N, C, H, W)` layout.
    """

    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(CHANNELS + 1, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(16, CHANNELS, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_map = t.view(-1, 1, 1, 1).expand(-1, 1, *x_t.shape[-2:])
        h = self.body(torch.cat([x_t, t_map], dim=1))
        return self.head(h)  # (N, C, H, W)


class ChannelsLast(nn.Module):
    """
    Laplace wrapper. Takes a single dict-like input and moves the channel axis
    of the output to the end, i.e., `(N, C, H, W)` -> `(N, H, W, C)`.
    """

    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(self, data: MutableMapping) -> torch.Tensor:
        device = next(self.parameters()).device
        x_t = data["x_t"].to(device)
        t = data["t"].to(device)
        return self.denoiser(x_t, t).movedim(1, -1)  # (N, H, W, C)


class NoisyImages(Dataset):
    """
    Random stand-in for a diffusion training set. The target `eps` must have the
    same layout as the wrapped model's output, i.e., `(H, W, C)` per example.
    """

    def __init__(self, n_data: int) -> None:
        self.x_t = torch.randn(n_data, CHANNELS, HEIGHT, WIDTH)
        self.t = torch.rand(n_data)
        self.eps = torch.randn(n_data, CHANNELS, HEIGHT, WIDTH)

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x_t": self.x_t[i],
            "t": self.t[i],
            "eps": self.eps[i].movedim(0, -1),  # (H, W, C)
        }


dataloader = DataLoader(NoisyImages(N_DATA), batch_size=16)
data = next(iter(dataloader))

model = ChannelsLast(Denoiser())
# Train it as usual
model.eval()

out = model.denoiser(data["x_t"], data["t"])
print(f"Output of the original model: {tuple(out.shape)}")
print(f"Output of the wrapped model: {tuple(model(data).shape)}")


# Diagonal Laplace over all weights
la = Laplace(
    model,
    "regression",
    subset_of_weights="all",
    hessian_structure="diag",
    backend=CurvlinopsGGN,
    dict_key_x="x_t",
    dict_key_y="eps",
)
la.fit(dataloader)
la.optimize_prior_precision()

f_mu, f_var = la(data, pred_type="glm", diagonal_output=True)
print(f"[Diag Laplace] Mean: {tuple(f_mu.shape)}, variance: {tuple(f_var.shape)}")

# Back to PyTorch's layout, e.g., for plotting a per-pixel uncertainty map
f_var = f_var.movedim(-1, 1)
print(f"[Diag Laplace] Variance in (N, C, H, W): {tuple(f_var.shape)}")


# Laplace over the head only: switch off the gradients of everything else
model = ChannelsLast(Denoiser())
model.eval()

for p in model.denoiser.body.parameters():
    p.requires_grad = False

la = Laplace(
    model,
    "regression",
    subset_of_weights="all",
    hessian_structure="full",
    backend=CurvlinopsGGN,
    dict_key_x="x_t",
    dict_key_y="eps",
)
la.fit(dataloader)
la.optimize_prior_precision()

f_mu, f_var = la(data, pred_type="glm", diagonal_output=True)
print(f"[Head-only Laplace] Mean: {tuple(f_mu.shape)}, variance: {tuple(f_var.shape)}")


# K-FAC Laplace over all weights, with the sampling predictive
model = ChannelsLast(Denoiser())
model.eval()

la = Laplace(
    model,
    "regression",
    subset_of_weights="all",
    hessian_structure="kron",
    backend=CurvlinopsGGN,
    dict_key_x="x_t",
    dict_key_y="eps",
)
la.fit(dataloader)
la.optimize_prior_precision()

f_mu, f_var = la(data, pred_type="nn", link_approx="mc", n_samples=20)
print(f"[K-FAC Laplace] Mean: {tuple(f_mu.shape)}, variance: {tuple(f_var.shape)}")
