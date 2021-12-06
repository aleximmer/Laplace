<div align="center">
	<img src="https://raw.githubusercontent.com/AlexImmer/Laplace/main/logo/laplace_logo.png" alt="Laplace" width="300"/>
</div>

[![Main](https://travis-ci.com/AlexImmer/Laplace.svg?token=rpuRxEjQS6cCZi7ptL9y&branch=main)](https://travis-ci.com/AlexImmer/Laplace)

The laplace package facilitates the application of Laplace approximations for entire neural networks or just their last layer.
The package enables posterior approximations, marginal-likelihood estimation, and various posterior predictive computations.
The library documentation is available at [https://aleximmer.github.io/Laplace](https://aleximmer.github.io/Laplace).

There is also a corresponding paper, [*Laplace Redux — Effortless Bayesian Deep Learning*](https://arxiv.org/abs/2106.14806), which introduces the library, provides an introduction to the Laplace approximation, reviews its use in deep learning, and empirically demonstrates its versatility and competitiveness. Please consider referring to the paper when using our library:
```bibtex
@article{daxberger2021laplace,
  title={Laplace Redux--Effortless Bayesian Deep Learning},
  author={Daxberger, Erik and Kristiadi, Agustinus and Immer, Alexander
          and Eschenhagen, Runa and Bauer, Matthias and Hennig, Philipp},
  journal={arXiv preprint arXiv:2106.14806},
  year={2021}
}
```

## Setup

We assume `python3.8` since the package was developed with that version.
To install laplace with `pip`, run the following:
```bash
pip install laplace-torch
```

For development purposes, clone the repository and then install:
```bash
# or after cloning the repository for development
pip install -e .
# run tests
pip install -e .[tests]
pytest tests/
```

## Structure
The laplace package consists of two main components:

1. The subclasses of [`laplace.BaseLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/baselaplace.py) that implement different sparsity structures: different subsets of weights (`'all'` and `'last_layer'`) and different structures of the Hessian approximation (`'full'`, `'kron'`, and `'diag'`). This results in six currently available options: `laplace.FullLaplace`, `laplace.KronLaplace`, `laplace.DiagLaplace`, and the corresponding last-layer variations `laplace.FullLLLaplace`, `laplace.KronLLLaplace`,  and `laplace.DiagLLLaplace`, which are all subclasses of [`laplace.LLLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/lllaplace.py). All of these can be conveniently accessed via the [`laplace.Laplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/laplace.py) function.
2. The backends in [`laplace.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/) which provide access to Hessian approximations of
the corresponding sparsity structures, for example, the diagonal GGN.

Additionally, the package provides utilities for
decomposing a neural network into feature extractor and last layer for `LLLaplace` subclasses ([`laplace.feature_extractor`](https://github.com/AlexImmer/Laplace/blob/main/laplace/feature_extractor.py))
and
effectively dealing with Kronecker factors ([`laplace.matrix`](https://github.com/AlexImmer/Laplace/blob/main/laplace/matrix.py)).

## Extendability
To extend the laplace package, new `BaseLaplace` subclasses can be designed, for example,
a block-diagonal structure or subset-of-weights Laplace.
Alternatively, extending or integrating backends (subclasses of [`curvature.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvature.py)) allows to provide different Hessian
approximations to the Laplace approximations.
For example, currently the [`curvature.BackPackInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/backpack.py) based on [BackPACK](https://github.com/f-dangel/backpack/) and [`curvature.AsdlInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/asdl.py) based on [ASDL](https://github.com/kazukiosawa/asdfghjkl) are available.
The `curvature.AsdlInterface` provides a Kronecker factored empirical Fisher while the `curvature.BackPackInterface`
does not, and only the `curvature.BackPackInterface` provides access to Hessian approximations
for a regression (MSELoss) loss function.

## Example usage

### *Post-hoc* prior precision tuning of last-layer LA

In the following example, a pre-trained model is loaded,
then the Laplace approximation is fit to the training data,
and the prior precision is optimized with cross-validation `'CV'`.
After that, the resulting LA is used for prediction with
the `'probit'` predictive for classification.

```python
from laplace import Laplace

# pre-trained model
model = load_map_model()  

# User-specified LA flavor
la = Laplace(model, 'classification',
             subset_of_weights='all',
             hessian_structure='diag')
la.fit(train_loader)
la.optimize_prior_precision(method='CV', val_loader=val_loader)

# User-specified predictive approx.
pred = la(x, link_approx='probit')
```

### Differentiating the log marginal likelihood w.r.t. hyperparameters

The marginal likelihood can be used for model selection and is differentiable
for continuous hyperparameters like the prior precision or observation noise.
Here, we fit the library default, KFAC last-layer LA and differentiate
the log marginal likelihood.

```python
from laplace import Laplace

# Un- or pre-trained model
model = load_model()  

# Default to recommended last-layer KFAC LA:
la = Laplace(model, likelihood='regression')
la.fit(train_loader)

# ML w.r.t. prior precision and observation noise
ml = la.log_marginal_likelihood(prior_prec, obs_noise)
ml.backward()
```

## Documentation

The documentation is available [here](https://aleximmer.github.io/Laplace) or can be generated and/or viewed locally:

```bash
# assuming the repository was cloned
pip install -e .[docs]
# create docs and write to html
bash update_docs.sh
# .. or serve the docs directly
pdoc --http 0.0.0.0:8080 laplace --template-dir template
```

## References

This package relies on various improvements to the Laplace approximation for neural networks, which was originally due to MacKay [1].

- [1] MacKay, DJC. [*A Practical Bayesian Framework for Backpropagation Networks*](https://authors.library.caltech.edu/13793/). Neural Computation 1992.
- [2] Gibbs, M. N. [*Bayesian Gaussian Processes for Regression and Classification*](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.147.1130&rep=rep1&type=pdf). PhD Thesis 1997.
- [3] Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N., Sundaram, N., Patwary, M., Prabhat, M., Adams, R. [*Scalable Bayesian Optimization Using Deep Neural Networks*](https://arxiv.org/abs/1502.05700). ICML 2015.
- [4] Ritter, H., Botev, A., Barber, D. [*A Scalable Laplace Approximation for Neural Networks*](https://openreview.net/forum?id=Skdvd2xAZ). ICLR 2018.
- [5] Foong, A. Y., Li, Y., Hernández-Lobato, J. M., Turner, R. E. [*'In-Between' Uncertainty in Bayesian Neural Networks*](https://arxiv.org/abs/1906.11537). ICML UDL Workshop 2019.
- [6] Khan, M. E., Immer, A., Abedi, E., Korzepa, M. [*Approximate Inference Turns Deep Networks into Gaussian Processes*](https://arxiv.org/abs/1906.01930). NeurIPS 2019.
- [7] Kristiadi, A., Hein, M., Hennig, P. [*Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks*](https://arxiv.org/abs/2002.10118). ICML 2020.
- [8] Immer, A., Korzepa, M., Bauer, M. [*Improving predictions of Bayesian neural nets via local linearization*](https://arxiv.org/abs/2008.08400). AISTATS 2021.
- [9] Immer, A., Bauer, M., Fortuin, V., Rätsch, G., Khan, EM. [*Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning*](https://arxiv.org/abs/2104.04975). ICML 2021.
