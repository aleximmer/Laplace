In this example, we apply Laplace to semantic segmentation, i.e., classification with
one prediction per pixel. The predictive gives a distribution over the classes at every
pixel, from which we get a per-pixel uncertainty map.

Laplace expects the class axis to be the last one, e.g., `(N, K)` for standard
classifiers or `(N, L, K)` for [language models](lm_example.md). Convolutional models in
PyTorch instead return logits `(N, K, H, W)`, with the classes on axis 1. Laplace does
not reorder the output for you, so you need to wrap your model so that it returns
`(N, H, W, K)`. The wrapper is a single `movedim` call. The targets are the usual class
indices of shape `(N, H, W)` and do not need to change.

!!! important "Intended usage"

    The wrapped model must return logits `(N, H, W, K)`, and the targets must be class
    indices `(N, H, W)`. Every pixel is an observation in the likelihood, i.e., the
    log-likelihood sums over all `N * H * W` pixels.

First, the imports and some constants.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace
from laplace.curvature import CurvlinopsGGN

torch.manual_seed(0)

N_DATA = 64
CHANNELS = 3  # C
HEIGHT = 8  # H
WIDTH = 8  # W
N_CLASSES = 5  # K
```

Next, we define a toy segmentation network. It returns per-pixel logits in PyTorch's
usual `(N, K, H, W)` layout. In practice, this would be your U-Net or similar.

```python
class SegmentationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(16, N_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))  # (N, K, H, W)
```

## Wrapping the model

The wrapper moves the class axis of the logits to the end.

```python
class ClassesLast(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).movedim(1, -1)  # (N, H, W, K)
```

We use random images and masks in place of a real segmentation data set.

```python
images = torch.randn(N_DATA, CHANNELS, HEIGHT, WIDTH)
masks = torch.randint(N_CLASSES, (N_DATA, HEIGHT, WIDTH))
dataloader = DataLoader(TensorDataset(images, masks), batch_size=16)
X, _ = next(iter(dataloader))

model = ClassesLast(SegmentationNet())
# Train it as usual
model.eval()

print(f"Output of the original model: {tuple(model.net(X).shape)}")
print(f"Output of the wrapped model: {tuple(model(X).shape)}")
```

```
Output of the original model: (16, 5, 8, 8)
Output of the wrapped model: (16, 8, 8, 5)
```

## Fitting Laplace

Segmentation networks are usually large, so we use a K-FAC Hessian over all weights.
K-FAC with the Curvlinops backends supports `nn.Linear` and `nn.Conv2d` layers only.
Segmentation networks often also have normalization layers, e.g., `nn.BatchNorm2d` or
`nn.GroupNorm`, and `nn.ConvTranspose2d` for upsampling. If those have trainable
parameters, `fit` raises a `NotImplementedError`. Either freeze them, so that the
Laplace approximation only covers the other layers, or use `hessian_structure="diag"`,
which supports them. The toy network above has no such layers, so the following does
nothing here.

```python
for module in model.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.ConvTranspose2d)):
        for p in module.parameters():
            p.requires_grad = False
```

Note that we set `subset_of_weights="all"` explicitly. The default, `"last_layer"`,
requires an `nn.Linear` last layer, so it fails for a convolutional head.

```python
la = Laplace(
    model,
    "classification",
    subset_of_weights="all",
    hessian_structure="kron",
    backend=CurvlinopsGGN,
)
la.fit(dataloader)
la.optimize_prior_precision()
```

## Predictive and uncertainty map

The GLM predictive returns the class probabilities of every pixel, in the wrapped
layout `(N, H, W, K)`. Their entropy is a per-pixel uncertainty map.

```python
probs = la(X, pred_type="glm", link_approx="probit")
print(f"[GLM] Probabilities: {tuple(probs.shape)}")

entropy = -(probs * probs.log()).sum(dim=-1)
print(f"[GLM] Entropy: {tuple(entropy.shape)}")
```

```
[GLM] Probabilities: (16, 8, 8, 5)
[GLM] Entropy: (16, 8, 8)
```

To get back to PyTorch's layout, e.g., for plotting, undo the `movedim`:

```python
probs = probs.movedim(-1, 1)
print(f"[GLM] Probabilities in (N, K, H, W): {tuple(probs.shape)}")
```

```
[GLM] Probabilities in (N, K, H, W): (16, 5, 8, 8)
```

The GLM predictive needs the Jacobian of the `H * W * K` logits of every image with
respect to the weights in the Laplace approximation. For large images, that does not fit
in memory. The sampling predictive runs the model once per weight sample from the
posterior instead and never computes the Jacobian.

```python
probs = la(X, pred_type="nn", link_approx="mc", n_samples=20)
print(f"[NN] Probabilities: {tuple(probs.shape)}")
```

```
[NN] Probabilities: (16, 8, 8, 5)
```

## Caveats

- Use the Curvlinops backends (`CurvlinopsGGN`, `CurvlinopsEF`). The ASDL and BackPACK
  backends raise an error for image outputs.
- `subset_of_weights="last_layer"` requires an `nn.Linear` last layer. For a
  convolutional head, switch off the gradients of all other layers and use
  `subset_of_weights="all"`, as in the
  [diffusion example](diffusion_example.md#bayesdiff-diagonal-last-layer-laplace).
- The targets must be class indices. Class probabilities as targets are not supported
  for image outputs.
- Pixels labeled with `-100` (PyTorch's default `ignore_index`), e.g., void pixels, are
  skipped by the loss, but still count toward the number of observations. With
  `CurvlinopsGGN`, they also still count toward the K-FAC Hessian used on this page and
  toward the diagonal Hessian. The full Hessian and the `CurvlinopsEF` backend leave
  them out. Masking them everywhere is tracked in
  [#313](https://github.com/aleximmer/Laplace/issues/313).

The full script is in
[`examples/segmentation_example.py`](https://github.com/aleximmer/Laplace/blob/main/examples/segmentation_example.py).
