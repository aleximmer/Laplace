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
        probs = probs.view(-1, probs.shape[-1])
        targets = targets.view(-1)

        self.nll_sum += F.nll_loss(
            probs.log(), targets, ignore_index=self.ignore_index, reduction="sum"
        )
        self.n_valid_labels += (targets != self.ignore_index).sum()

    def compute(self) -> torch.Tensor:
        return self.nll_sum / self.n_valid_labels
