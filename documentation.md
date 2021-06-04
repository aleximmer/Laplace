## Setup

```bash
pip install -r requirements.txt
# for development
pip install -e .
# for "production"
pip install .

# for asdfghjkl backend
pip install dependencies/asdfghjkl-0.0.1-py3-none-any.whl

# run tests
pip install pytest
pytest tests/
```

## Structure 
The laplace package consists of two main-components: 

1. The subclasses of `laplace.BaseLaplace` that implement different sparsity structures
(last-layer `LLLaplace` vs. `FullLaplace` and `DiagLaplace` vs. `KronLaplace`) all of which
can be conveniently accessed with the method `Laplace`.
2. The backends in `laplace.curvature` which provide access to Hessian approximations of
the corresponding sparsity structures, for example, the diagonal GGN.

Additionally, the package provies utilities for
decomposing a neural network into feature extractor and last layer for `LLLaplace` (`laplace.feature_extractor`)
and
effectively dealing with Kronecker factors (`Kron` and `KronDecomposed`).

## Extendability
To extend the laplace package, new `BaseLaplace` subclasses can be designed, for example,
a block-diagonal structure or subset-of-weights Laplace.
Alternatively, extending or integrating backends allows to provide different Hessian
approximations to the laplace approximations.
For example, currently the `BackPackInterface` and `AsdfInterface` are available.
The `AsdfInterface` provides a Kronecker factored empirical Fisher while the `BackPackInterface`
does not, and only the `BackPackInterface` provides access to Hessian approximations
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
la.optimize_prior_precision(method='CV')

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
