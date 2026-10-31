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


class SegmentationNet(nn.Module):
    """
    Toy segmentation network. It returns per-pixel logits in PyTorch's usual
    `(N, K, H, W)` layout.
    """

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


class ClassesLast(nn.Module):
    """
    Laplace wrapper. Moves the class axis of the logits to the end, i.e.,
    `(N, K, H, W)` -> `(N, H, W, K)`.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).movedim(1, -1)  # (N, H, W, K)


# Random stand-in for a segmentation data set. The targets are class indices
# `(N, H, W)`, as usual, and do not need to be reordered.
images = torch.randn(N_DATA, CHANNELS, HEIGHT, WIDTH)
masks = torch.randint(N_CLASSES, (N_DATA, HEIGHT, WIDTH))
dataloader = DataLoader(TensorDataset(images, masks), batch_size=16)
X, _ = next(iter(dataloader))

model = ClassesLast(SegmentationNet())
# Train it as usual
model.eval()

print(f"Output of the original model: {tuple(model.net(X).shape)}")
print(f"Output of the wrapped model: {tuple(model(X).shape)}")


# K-FAC Laplace over all weights
la = Laplace(
    model,
    "classification",
    subset_of_weights="all",
    hessian_structure="kron",
    backend=CurvlinopsGGN,
)
la.fit(dataloader)
la.optimize_prior_precision()

# GLM predictive: a distribution over the classes per pixel
probs = la(X, pred_type="glm", link_approx="probit")
print(f"[GLM] Probabilities: {tuple(probs.shape)}")

# Per-pixel uncertainty map
entropy = -(probs * probs.log()).sum(dim=-1)
print(f"[GLM] Entropy: {tuple(entropy.shape)}")

# Back to PyTorch's layout, e.g., for plotting
probs = probs.movedim(-1, 1)
print(f"[GLM] Probabilities in (N, K, H, W): {tuple(probs.shape)}")

# Sampling predictive, for images whose Jacobians do not fit in memory
probs = la(X, pred_type="nn", link_approx="mc", n_samples=20)
print(f"[NN] Probabilities: {tuple(probs.shape)}")
