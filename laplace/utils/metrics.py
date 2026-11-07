import torch
from torch.nn import functional as F
from torchmetrics import Metric


class RunningNLLMetric(Metric):
    """
    NLL metrics that

    Parameters
    ----------
    ignore_index: int, default = -100
        which class label to ignore when computing the NLL loss
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
            integer tensor of shape (...)
        """
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
        if probs.shape != targets.shape:
            raise ValueError(
                f"The targets have shape {tuple(targets.shape)}, but the "
                f"probabilities have shape {tuple(probs.shape)}."
            )

        self.nll_sum += F.binary_cross_entropy(
            probs, targets.to(probs.dtype), reduction="sum"
        )
        self.n_positions += probs.shape[:-1].numel()

    def compute(self) -> torch.Tensor:
        return self.nll_sum / self.n_positions
