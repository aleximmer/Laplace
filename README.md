<div align="center">
 <img src="https://raw.githubusercontent.com/AlexImmer/Laplace/main/logo/laplace_logo.png" alt="Laplace" width="300"/>

![pytest](https://github.com/aleximmer/laplace/actions/workflows/pytest-default.yml/badge.svg)
![lint](https://github.com/aleximmer/laplace/actions/workflows/lint-ruff.yml/badge.svg)
![format](https://github.com/aleximmer/laplace/actions/workflows/format-ruff.yml/badge.svg)

<br />

[📖 Documentation \& API reference](https://aleximmer.github.io/Laplace)

<br />
</div>

The laplace package facilitates the application of Laplace approximations for entire neural networks, subnetworks of neural networks, or just their last layer.
The package enables posterior approximations, marginal-likelihood estimation, and various posterior predictive computations.

There is also a corresponding paper, [_Laplace Redux — Effortless Bayesian Deep Learning_](https://arxiv.org/abs/2106.14806), which introduces the library, provides an introduction to the Laplace approximation, reviews its use in deep learning, and empirically demonstrates its versatility and competitiveness. Please consider referring to the paper when using our library:

```bibtex
@inproceedings{laplace2021,
  title={Laplace Redux--Effortless {B}ayesian Deep Learning},
  author={Erik Daxberger and Agustinus Kristiadi and Alexander Immer
          and Runa Eschenhagen and Matthias Bauer and Philipp Hennig},
  booktitle={{N}eur{IPS}},
  year={2021}
}
```

The [code](https://github.com/runame/laplace-redux) to reproduce the experiments in the paper is also publicly available; it provides examples of how to use our library for predictive uncertainty quantification, model selection, and continual learning.

> [!IMPORTANT]
> As a user, one should not expect Laplace to work automatically.
> That is, one should experiment with different Laplace's options
> (hessian_factorization, prior precision tuning method, predictive method, backend,
> etc!). Try looking at various papers that use Laplace for references on how to
> set all those options depending on the applications/problems at hand.

## Installation

> [!IMPORTANT]
> We assume Python >= 3.9 since lower versions are [(soon to be) deprecated](https://devguide.python.org/versions/).
> PyTorch version 2.0 and up is also required for full compatibility.

To install laplace with `pip`, run the following:

```bash
pip install laplace-torch
```

Additionally, if you want to use the `asdfghjkl` backend, please install it via:

```bash
pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl
```

## Simple usage

> [!TIP]
> Check out <https://aleximmer.github.io/Laplace> for more usage examples
> and API reference.

In the following example, a pre-trained model is loaded,
then the Laplace approximation is fit to the training data
(using a diagonal Hessian approximation over all parameters),
and the prior precision is optimized with cross-validation `"gridsearch"`.
After that, the resulting LA is used for prediction with
the `"probit"` predictive for classification.

```python
from laplace import Laplace

# Pre-trained model
model = load_map_model()

# User-specified LA flavor
la = Laplace(model, "classification",
             subset_of_weights="all",
             hessian_structure="diag")
la.fit(train_loader)
la.optimize_prior_precision(
    method="gridsearch",
    pred_type="glm",
    link_approx="probit",
    val_loader=val_loader
)

# User-specified predictive approx.
pred = la(x, pred_type="glm", link_approx="probit")
```

## Contributing

Pull requests are very welcome.
Please follow the guidelines in <https://aleximmer.github.io/Laplace/devs_guide>
