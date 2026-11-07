import torch
from torch.nn import functional as F
from torchmetrics import Metric


class RunningNLLMetric(Metric):
    """Negative log-likelihood of categorical predictions, averaged over positions.

    The targets are class indices, or class probabilities as in
    `nn.CrossEntropyLoss`, e.g. one-hot or soft labels. A position is an entry of the
    leading dims, so the metric is the average over examples for probabilities
    `(batch, classes)`, and over examples and tokens for `(batch, length, classes)`.

    Parameters
    ----------
    ignore_index: int, default = -100
        which class label to ignore when computing the NLL loss. It only applies to
        class indices, since every position of class probabilities counts.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_valid_labels", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.ignore_index: int = ignore_index

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Parameters
        ----------
        probs: torch.Tensor
            probability tensor of shape (..., n_classes)

        targets: torch.Tensor
            integer tensor of class indices of shape (...), or floating-point tensor
            of class probabilities of shape (..., n_classes)
        """
        if targets.is_floating_point() and targets.shape == probs.shape:
            # Class probabilities. A class that has probability 0 in the targets adds
            # nothing, even where the prediction is 0 too.
            self.nll_sum += -torch.special.xlogy(targets, probs).sum()
            self.n_valid_labels += probs.shape[:-1].numel()
            return

        probs = probs.reshape(-1, probs.shape[-1])
        targets = targets.reshape(-1)

        self.nll_sum += F.nll_loss(
            probs.log(), targets, ignore_index=self.ignore_index, reduction="sum"
        )
        self.n_valid_labels += (targets != self.ignore_index).sum()

    def compute(self) -> torch.Tensor:
        return self.nll_sum / self.n_valid_labels


class RunningBernoulliNLLMetric(Metric):
    """Negative log-likelihood of Bernoulli predictions, averaged over positions.

    The NLL of a position is the sum over its outputs, e.g. over the labels of a
    multi-label example or the channels of a pixel. A position is an entry of the
    leading dims of the targets, so the metric is the average over examples for
    targets `(batch, outputs)`, and over examples and pixels for `(batch, H, W,
    outputs)`. Soft targets in \\([0, 1]\\) are fine.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_state("nll_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_positions", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Parameters
        ----------
        probs: torch.Tensor
            probabilities of the outputs, shape (..., n_outputs)

        targets: torch.Tensor
            floating-point tensor in [0, 1] with the shape of `probs`
        """
        self.nll_sum += F.binary_cross_entropy(
            probs, targets.to(probs.dtype), reduction="sum"
        )
        self.n_positions += probs.shape[:-1].numel()

    def compute(self) -> torch.Tensor:
        return self.nll_sum / self.n_positions
