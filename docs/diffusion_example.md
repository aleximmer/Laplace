In this example, we apply Laplace to a model whose output is an image, such as the
denoiser of a diffusion model. The same recipe works for other image- or video-output
models trained with a Gaussian likelihood. For per-pixel classification, see the
[semantic segmentation example](segmentation_example.md).

Laplace expects the output axis to be the last one, e.g., `(N, K)` for standard models
or `(N, L, K)` for [language models](lm_example.md). Convolutional models in PyTorch
instead return `(N, C, H, W)`, with the channels on axis 1. Laplace does not reorder
the output for you, so you need to wrap your model so that it returns `(N, H, W, C)`.
The wrapper is a single `movedim` call. Your data loader must emit the targets in the
same layout.

!!! important "Intended usage"

    The wrapped model must return `(N, H, W, C)`, and the targets must also be
    `(N, H, W, C)`. Every pixel is an observation in the likelihood, i.e., the
    log-likelihood sums over all `N * H * W` pixels, each with a `C`-dimensional
    output.

First, the imports and some constants.

```python
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
```

Next, we define a toy denoiser. It predicts the noise `eps` from a noisy image `x_t` and
the diffusion time `t`, and returns its prediction in PyTorch's usual `(N, C, H, W)`
layout. In practice, this would be the U-Net of your diffusion model.

```python
class Denoiser(nn.Module):
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
```

## Wrapping the model

The wrapper does two things. First, since the denoiser takes two inputs, it takes a
single dict-like input and unpacks it, as in the [Huggingface example](huggingface_example.md).
Second, it moves the channel axis of the output to the end. Notice that we also move
the inputs to the right device inside `forward`.

```python
class ChannelsLast(nn.Module):
    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def forward(self, data: MutableMapping) -> torch.Tensor:
        device = next(self.parameters()).device
        x_t = data["x_t"].to(device)
        t = data["t"].to(device)
        return self.denoiser(x_t, t).movedim(1, -1)  # (N, H, W, C)
```

The dataset emits dicts as well. The target `eps` gets the same `movedim` so that it
matches the wrapped model's output. Here we use random tensors in place of a real
diffusion training set.

```python
class NoisyImages(Dataset):
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
```

```
Output of the original model: (16, 3, 8, 8)
Output of the wrapped model: (16, 8, 8, 3)
```

## Reproducing BayesDiff and FLARE

Image outputs are large. A single example has `H * W * C` outputs, so the Jacobian that
the GLM predictive needs has `H * W * C` rows per example and one column per parameter
in the Laplace approximation. For a 64x64 RGB image, that is 12,288 rows. Diffusion
papers keep this affordable by putting the Laplace approximation on a small part of the
denoiser. [BayesDiff](https://arxiv.org/abs/2310.11142) uses the last layer, while
[FLARE](https://arxiv.org/abs/2602.09170) uses a random subset of the weights. Below we
set up both on the toy denoiser. They use the Curvlinops backend; see the caveats at the
end for why.

### BayesDiff: diagonal last-layer Laplace

BayesDiff fits a last-layer Laplace approximation with a diagonal Hessian and estimates
the variance of `eps` by sampling weights from the posterior, with 100 samples.

Laplace's own last-layer option (`subset_of_weights="last_layer"`) only supports
`nn.Linear` last layers, and the head here is a convolution. To get the same effect,
switch off the gradients of every other layer and set `subset_of_weights="all"`. Laplace
only computes the Hessian and the Jacobians of parameters with `requires_grad=True`. The
same trick works for any set of layers.

The sampling predictive (`pred_type="nn"`) needs `link_approx="mc"`. It runs the model
once per weight sample and never computes the Jacobian.

```python
for p in model.denoiser.body.parameters():
    p.requires_grad = False

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

f_mu, f_var = la(data, pred_type="nn", link_approx="mc", n_samples=100)
print(f"[BayesDiff] Mean: {tuple(f_mu.shape)}, variance: {tuple(f_var.shape)}")
```

```
[BayesDiff] Mean: (16, 8, 8, 3), variance: (16, 8, 8, 3)
```

The variance has one entry per pixel and channel, in the wrapped layout. To get back to
`(N, C, H, W)`, e.g., to plot a per-pixel uncertainty map, undo the `movedim`:

```python
f_var = f_var.movedim(-1, 1)
print(f"[BayesDiff] Variance in (N, C, H, W): {tuple(f_var.shape)}")
```

```
[BayesDiff] Variance in (N, C, H, W): (16, 3, 8, 8)
```

### FLARE: full Laplace over a random subnetwork

FLARE argues that a last-layer Laplace approximation is not enough for diffusion models.
It draws a uniformly random subset of the weights once, fits a full GGN over that
subset, and pushes the posterior through the linearized denoiser, i.e., $J \Sigma J^\top$,
where $J$ is the Jacobian of the denoiser with respect to the subset and $\Sigma$ is the
posterior covariance.

This maps to subnetwork Laplace (`subset_of_weights="subnetwork"`) with the GLM
predictive. Subnetwork Laplace takes the indices of the chosen weights in the flattened
parameter vector, i.e., `torch.nn.utils.parameters_to_vector(model.parameters())`.
`RandomSubnetMask` picks them uniformly at random. Other masks in `laplace.utils`, such
as `LargestMagnitudeSubnetMask`, pick them by a score instead. With a few hundred
weights, a full Hessian is cheap.

```python
from laplace.utils import RandomSubnetMask

model = ChannelsLast(Denoiser())
model.eval()

subnetwork_mask = RandomSubnetMask(model, n_params_subnet=500)
subnetwork_indices = subnetwork_mask.select()

la = Laplace(
    model,
    "regression",
    subset_of_weights="subnetwork",
    hessian_structure="full",
    subnetwork_indices=subnetwork_indices,
    backend=CurvlinopsGGN,
    dict_key_x="x_t",
    dict_key_y="eps",
)
la.fit(dataloader)
la.optimize_prior_precision()

f_mu, f_var = la(data, pred_type="glm", diagonal_output=True)
print(f"[FLARE] Hessian: {tuple(la.H.shape)}")
print(f"[FLARE] Mean: {tuple(f_mu.shape)}, variance: {tuple(f_var.shape)}")
```

```
[FLARE] Hessian: (500, 500)
[FLARE] Mean: (16, 8, 8, 3), variance: (16, 8, 8, 3)
```

With `diagonal_output=True`, you get the diagonal of $J \Sigma J^\top$, the same shape as the
mean. Without it, you get a `C x C` covariance per pixel, i.e., a tensor of shape
`(N, H, W, C, C)`.

FLARE scores each generated sample by the trace of its epistemic covariance divided by
the output dimension. Only the diagonal is needed for that:

```python
score = f_var.flatten(start_dim=1).mean(dim=1)
print(f"[FLARE] Score: {tuple(score.shape)}")
```

```
[FLARE] Score: (16,)
```

FLARE also carries this covariance through the reverse diffusion steps and takes the
score at the final step. That recursion sits outside Laplace; the snippet above gives
you the per-step term.

### Other options

A diagonal Hessian over all weights also works with the GLM predictive, and so does
K-FAC (`hessian_structure="kron"`), which is cheap to fit for large models. When the
Jacobians of the GLM predictive do not fit in memory, use the sampling predictive
(`pred_type="nn"` with `link_approx="mc"`), as in BayesDiff. It works with any Hessian
structure.

## Video outputs

The recipe is the same for video. A model with output `(N, C, T, H, W)` is wrapped with
`.movedim(1, -1)` so that it returns `(N, T, H, W, C)`, and the targets get the same
layout. Diagonal and full Hessians work as above. K-FAC currently does not support
`nn.Conv3d` layers.

## Caveats

- Use the Curvlinops backends (`CurvlinopsGGN`, `CurvlinopsEF`). The ASDL and BackPACK
  backends raise an error for image outputs.
- `subset_of_weights="last_layer"` requires an `nn.Linear` last layer. Use the
  `requires_grad` approach from the BayesDiff section for convolutional heads.

The full script is in
[`examples/diffusion_example.py`](https://github.com/aleximmer/Laplace/blob/main/examples/diffusion_example.py).
