### Setup dev environment

For development purposes, e.g. if you would like to make contributions, follow
the following steps:

**With `uv`**

1. Install [`uv`](https://github.com/astral-sh/uv), e.g. `pip install --upgrade uv`
2. Then clone this repository and install the development dependencies:

```bash
git clone git@github.com:aleximmer/Laplace.git
uv sync --all-extras
```

3. `laplace-torch` is now available in editable mode, e.g. you can run:

```bash
uv run python examples/regression_example.py

# Or, equivalently:
uv venv
source .venv/bin/activate
python examples/regression_example.py
```

**With `pip`**

```bash
git clone git@github.com:aleximmer/Laplace.git

# Recommended to create a virtualenv before the following step
pip install -e ".[dev]"

# Run as usual, e.g.
python examples/regression_examples.py
```

## Contributing

Pull requests are very welcome. Please follow these guidelines:

1. Follow the [development setup](#setup-dev-environment).
2. Use [ruff](https://github.com/astral-sh/ruff) as autoformatter. Please refer to the following [makefile](https://github.com/aleximmer/Laplace/blob/main/makefile) and run it via `make ruff`. Please note that the order of `ruff check --fix` and `ruff format` is important!
3. Also use [ruff](https://github.com/astral-sh/ruff) as linter. Please manually fix all linting errors/warnings before opening a pull request.
4. Fully document your changes in the form of Python docstrings, typehinting, and (if applicable) code/markdown examples in the `./examples` subdirectory.
   1. See `docs/api_reference/*.md` on how to include a newly added class in the docs.
5. Provide as many test cases as possible. Make sure all test cases pass.

Issues, bug reports, and ideas are also very welcome!

## Documentation

The documentation is available [here](https://aleximmer.github.io/Laplace) or can be generated and/or viewed locally:

**With `uv`**

```bash
# assuming the repository was cloned
uv sync --all-extras
# serve the docs locally
uv run mkdocs serve
```

**With `pip`**

```bash
# assuming the repository was cloned
pip install -e ".[dev]"
# serve the docs locally
mkdocs serve
```

## Publishing the `laplace-torch` package to PyPI

1. Update the package version in `pyproject.toml`
2. Have your PyPI token ready
3. Build the wheel: `uv build`
4. Run `uv publish`
5. Create a new release on Github

!!! tip

      It's a good idea to test on TestPyPI first. To do so, simply run the above but with
      ```
      uv publish --publish-url https://test.pypi.org/legacy/
      ```

## Structure

The laplace package consists of two main components:

1. The subclasses of [`laplace.BaseLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/baselaplace.py) that implement different sparsity structures: different subsets of weights (`'all'`, `'subnetwork'` and `'last_layer'`) and different structures of the Hessian approximation (`'full'`, `'kron'`, `'lowrank'`, `'diag'` and `'gp'`). This results in _ten_ currently available options: `laplace.FullLaplace`, `laplace.KronLaplace`, `laplace.DiagLaplace`, `laplace.FunctionalLaplace` the corresponding last-layer variations `laplace.FullLLLaplace`, `laplace.KronLLLaplace`, `laplace.DiagLLLaplace` and `laplace.FunctionalLLLaplace` (which are all subclasses of [`laplace.LLLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/lllaplace.py)), [`laplace.SubnetLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/subnetlaplace.py) (which only supports `'full'` and `'diag'` Hessian approximations) and `laplace.LowRankLaplace` (which only supports inference over `'all'` weights). All of these can be conveniently accessed via the [`laplace.Laplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/laplace.py) function.
2. The backends in [`laplace.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/) which provide access to Hessian approximations of
   the corresponding sparsity structures, for example, the diagonal GGN.

Additionally, the package provides utilities for
decomposing a neural network into feature extractor and last layer for `LLLaplace` subclasses ([`laplace.utils.feature_extractor`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/feature_extractor.py))
and
effectively dealing with Kronecker factors ([`laplace.utils.matrix`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/matrix.py)).

Finally, the package implements several options to select/specify a subnetwork for `SubnetLaplace` (as subclasses of [`laplace.utils.subnetmask.SubnetMask`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/subnetmask.py)).
Automatic subnetwork selection strategies include: uniformly at random (`laplace.utils.subnetmask.RandomSubnetMask`), by largest parameter magnitudes (`LargestMagnitudeSubnetMask`), and by largest marginal parameter variances (`LargestVarianceDiagLaplaceSubnetMask` and `LargestVarianceSWAGSubnetMask`).
In addition to that, subnetworks can also be specified manually, by listing the names of either the model parameters (`ParamNameSubnetMask`) or modules (`ModuleNameSubnetMask`) to perform Laplace inference over.

## Extendability

To extend the laplace package, new `BaseLaplace` subclasses can be designed, for example,
Laplace with a block-diagonal Hessian structure.
One can also implement custom subnetwork selection strategies as new subclasses of `SubnetMask`.

Alternatively, extending or integrating backends (subclasses of [`curvature.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvature.py)) allows to provide different Hessian
approximations to the Laplace approximations.
For example, currently the [`curvature.CurvlinopsInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvlinops.py) based on [Curvlinops](https://github.com/f-dangel/curvlinops) and the native `torch.func` (previously known as `functorch`), [`curvature.BackPackInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/backpack.py) based on [BackPACK](https://github.com/f-dangel/backpack/) and [`curvature.AsdlInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/asdl.py) based on [ASDL](https://github.com/kazukiosawa/asdfghjkl) are available.
