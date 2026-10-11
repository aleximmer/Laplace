# ELLA

ELLA approximates the function-space posterior of a pretrained network using a
Nyström basis, following [Deng, Zhou, and Zhu](https://arxiv.org/abs/2210.12642).

```python
from laplace import Laplace

ella = Laplace(
    model,
    "classification",
    subset_of_weights="all",
    hessian_structure="gp",
    functional_approximation="nystrom",
    subsample_size=100,
    n_eigenvalues=20,
)
ella.fit(train_loader)
probabilities = ella(x)  # [batch, classes]
logits, covariance = ella.predictive_moments(x)
```

`fit` selects training inputs, builds a Nyström basis from parameter Jacobians,
and accumulates the generalized Gauss–Newton matrix in that basis. It returns
`None`. The `fit_history_` attribute records processed example counts and
validation negative log likelihood when validation data are supplied. To tune
scalar prior precision on validation data, call
`optimize_prior_precision(method="gridsearch", val_loader=...)`.

The prior precision must be a positive scalar and the prior mean must be zero.
Choose `0 < n_eigenvalues <= subsample_size`, with `subsample_size` no larger
than the number of eligible training rows. If the sampled kernel has too few
positive directions, `fit` raises a `ValueError` reporting its usable rank.
Reduce `n_eigenvalues`, increase `subsample_size`, or try a different seed.

Basis selection and curvature accumulation draw from the loader independently.
A stochastic sampler may select different rows for each pass; `n_data` records
the number of examples processed during curvature accumulation.

ELLA supports classification, regression, and reward modeling. Reward modeling
fits pairwise preferences and predicts a scalar reward on individual inputs.
`ella(pair_inputs, fitting=True)` returns pairwise class probabilities. For
classification, `ella(x)` returns probabilities. For regression, it returns the
latent mean and covariance without observation noise.

`predictive_moments(x, joint=True)` returns cross-input covariance.
`functional_samples` and `predictive_samples` return tensors shaped
`[samples, batch, outputs]`. Pass `joint=True` to preserve cross-input
dependence. ELLA does not provide weight-space `sample`, the subset-of-data GP
marginal likelihood, or a tracked training log likelihood.

`state_dict` and `load_state_dict` restore the fitted basis. Restoring a
checkpoint requires an equivalent pretrained model with matching input keys,
curvature backend, backend options, and backpropagation setting. The checkpoint
also restores the seed used to build the basis.
Model weights and buffers are checked against the checkpoint and must already
match; `load_state_dict` does not load them into the network.

Run `pytest tests/test_ella.py` for CPU coverage and
`pytest -m cuda tests/test_ella_cuda.py` with CUDA PyTorch.

See the [ELLA API reference](api_reference/ella.md).
Run the standalone API example with `python -m examples.ella` from the repository
root. It uses an untrained linear network to demonstrate fitting and tensor
shapes; applications should supply a pretrained model.
