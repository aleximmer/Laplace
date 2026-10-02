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

Run `pytest tests/test_ella.py` for CPU coverage and
`pytest -m cuda tests/test_ella_cuda.py` with CUDA PyTorch.

See the [ELLA API reference](api_reference/ella.md).
The standalone runnable example is `examples/ella.py` in the repository.
