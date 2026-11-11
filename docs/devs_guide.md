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
pip install -e ".[dev,all]"

# Run as usual, e.g.
python examples/regression_examples.py
```

**CPU-only PyTorch**

On Linux, the default PyTorch wheel installs the CUDA libraries even on machines
without an NVIDIA GPU. To get the much smaller CPU build instead (this is what CI uses):

```bash
# With uv
uv venv
uv pip install --torch-backend cpu -e ".[dev,all]"
uv run --no-sync python examples/regression_example.py

# With pip, install the CPU build of torch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,all]"
```

`uv.lock` pins the CUDA build of PyTorch. In the CPU-only uv setup, don't run `uv sync`,
and pass `--no-sync` to every `uv run` (or set `UV_NO_SYNC=1`). Otherwise uv
reinstalls the CUDA packages.

Simpler still, activate the virtualenv once and skip `uv run` altogether:

```bash
source .venv/bin/activate
python examples/regression_example.py
pytest tests/test_lllaplace.py
```

## Contributing

Pull requests are very welcome. Please follow these guidelines:

1. Follow the [development setup](#setup-dev-environment).
2. Use [ruff](https://github.com/astral-sh/ruff) as autoformatter. Please refer to the following [makefile](https://github.com/aleximmer/Laplace/blob/main/makefile) and run it via `make ruff`. Please note that the order of `ruff check --fix` and `ruff format` is important!
3. Also use [ruff](https://github.com/astral-sh/ruff) as linter. Please manually fix all linting errors/warnings before opening a pull request.
4. Fully document your changes in the form of Python docstrings, typehinting, and (if applicable) an example in the `./examples` subdirectory. See [writing an example](#writing-an-example).
   1. See `docs/api_reference/*.md` on how to include a newly added class in the docs.
5. Provide as many test cases as possible. Make sure all test cases pass.
6. CI also runs every script in `./examples`. New scripts are picked up automatically, so
   make sure they run on a CPU in a few minutes. To run them locally:

    ```bash
    uv pip install -e ".[dev,asdl]" -r examples/requirements.txt
    pytest examples/test_examples.py -m "not slow"
    ```

    CI only checks that the code runs, so it should be fast. If a script is slow, give it
    command-line options for fewer iterations, a data subset or a tiny model, and set them
    in `ARGS` in `examples/test_examples.py`. If the defaults use a large model or a full
    dataset, also add the script to `FULL`. CI then runs it without arguments once a month
    instead of on every pull request. To run those yourself, use `-m slow`.

!!! warning

    AI-assisted contributions are welcome, but you must understand the problem you are solving.
    You must also understand the solutions your AI agent proposes and review them yourself.
    When opening a PR, you **must** write the description yourself.
    First-time contributors must disclose their AI usage and confirm that they understand the changes.

Maintainers can ask Claude for a review of any pull request, including one from a fork,
by commenting `@claude review` on it. The review runs as a GitHub Actions workflow and
posts its findings as inline comments, followed by a summary that names the reviewed
commit. It checks for bugs and for compliance with `AGENTS.md`, and never approves or
blocks a pull request. Commenting again after new commits runs a fresh review. The
review instructions live in `.claude/skills/pr-review/SKILL.md`.

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

### Writing an example

Each example page on the doc site is a script in `./examples`, so the code you read is
the code CI runs. The scripts use the [jupytext](https://jupytext.readthedocs.io/)
percent format: `# %% [markdown]` starts a prose cell, whose lines are Markdown behind
`# `, and `# %%` starts a code cell. They run as plain Python, and JupyterLab or VS Code
with jupytext opens them as notebooks (`jupytext --to ipynb` converts one if you prefer).

```python
# %% [markdown]
# # My Example
#
# Some prose with math, $\sigma = 0.3$.

# %%
import torch
```

A script whose first line is `# %% [markdown]` becomes a page at `/<script name>/`. Add
it to the `nav` in `mkdocs.yml`, or the strict docs build fails. The pages are rendered
without running the code, so paste important printed output into a prose cell. Prose
cells don't go through the site's Markdown extensions, so:

- write math as `$...$` and `$$...$$`, not `\(...\)`;
- link to images as `../assets/<file>` and to other pages as `../<page>/`, not `<page>.md`;
- use a `> **Note:** ...` blockquote instead of an admonition;
- parse arguments with `parser.parse_known_args()`, since Jupyter passes its own.

`examples/test_examples.py` checks these rules.

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

1. The subclasses of [`laplace.BaseLaplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/baselaplace.py) that implement different sparsity structures: different subsets of weights (`'all'`, `'subnetwork'` and `'last_layer'`) and different structures of the Hessian approximation (`'full'`, `'kron'`, `'lowrank'`, `'diag'` and `'gp'`). The weight-space and subset-of-data GP estimators include `laplace.FullLaplace`, `laplace.KronLaplace`, `laplace.DiagLaplace`, `laplace.FunctionalLaplace`, their last-layer variations, the subnetwork variants, and `laplace.LowRankLaplace`. The function-space ELLA and VaLLA variants are selected through the same [`laplace.Laplace`](https://github.com/AlexImmer/Laplace/blob/main/laplace/laplace.py) function.
2. The backends in [`laplace.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/) which provide access to Hessian approximations of
   the corresponding sparsity structures, for example, the diagonal GGN.

For `hessian_structure="gp"`, the factory also accepts
`functional_approximation="sod" | "nystrom" | "variational"`. The latter two
select [`ELLA`](api_reference/ella.md) and [`VaLLA`](api_reference/valla.md), respectively. These and
`FunctionalLaplace` share
[`BaseFunctionalLaplace`](api_reference/functionallaplace.md#laplace.BaseFunctionalLaplace)
prediction and sampling methods, while each estimator
implements its own fitting and posterior covariance. New function-space methods
should implement `_glm_predictive_distribution`, `fit`, and method-specific
checkpoint state; they must not inherit another method's marginal-likelihood
formula by default.

The predictive-moment method returns a mean shaped `[batch, outputs]` and
covariance shaped `[batch, outputs, outputs]`. With `joint=True`, flatten the
mean to `[batch * outputs]` and return covariance shaped
`[batch * outputs, batch * outputs]`, ordered by input and then output.
These moments exclude observation noise. Implement `_refresh_posterior` if
changing hyperparameters invalidates cached posterior factors.

Reusable function-space helpers live in `laplace/utils/functional_utils.py`
and are internal. Import them from `laplace.utils.functional_utils` in library
code. They are not part of the public API, so they can change without warning.

Additionally, the package provides utilities for
decomposing a neural network into feature extractor and last layer for `LLLaplace` subclasses ([`laplace.utils.feature_extractor`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/feature_extractor.py))
and
effectively dealing with Kronecker factors ([`laplace.utils.matrix`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/matrix.py)).

Finally, the package implements several options to select/specify a subnetwork for `SubnetLaplace` (as subclasses of [`laplace.utils.subnetmask.SubnetMask`](https://github.com/AlexImmer/Laplace/blob/main/laplace/utils/subnetmask.py)).
Automatic subnetwork selection strategies include: uniformly at random (`laplace.utils.subnetmask.RandomSubnetMask`), by largest parameter magnitudes (`LargestMagnitudeSubnetMask`), and by largest marginal parameter variances (`LargestVarianceDiagLaplaceSubnetMask` and `LargestVarianceSWAGSubnetMask`).
In addition to that, subnetworks can also be specified manually, by listing the names of either the model parameters (`ParamNameSubnetMask`) or modules (`ModuleNameSubnetMask`) to perform Laplace inference over.

## Public API

The public API is the `__all__` of `laplace`, `laplace.curvature` and `laplace.utils`,
plus the members listed in `PUBLIC_COMMON` and `PUBLIC_EXTRA` in
`tests/test_public_api.py`. Everything else is internal, underscore or not. Each
public name has one import path. Laplace classes, likelihoods, enums and
`marglik_training` come from `laplace`; curvature backends from `laplace.curvature`;
subnet masks, metrics, `FeatureExtractor`, `Kron` and `KronDecomposed` from
`laplace.utils`. Submodules such as `laplace.baselaplace` are not public paths.

New public names can be added at any time. To remove or rename one, keep the old
name working with a `DeprecationWarning` for at least one release. Internal names
can change without that. The `INTERNAL` set in `tests/test_public_api.py` lists the
internal members that have no underscore yet; rename one to `_name` and drop it from
the set. The API reference hides these names through the `filters` regex in
`mkdocs.yml`, so drop it there too. A test checks that the two lists agree.

`tests/test_public_api.py` fails when a fitted instance has a member that is in none
of the lists, or when a snapshotted signature loses or renames a parameter. These
tests carry the `public_api` marker and run in their own CI workflow,
`test-public-api`. To add a public name:

1. Add it to the `__all__` of its package and to `EXPECTED_EXPORTS`, or, for a method
   or attribute, to `PUBLIC_COMMON` or `PUBLIC_EXTRA`.
2. For a callable, run `uv run python -m tests.public_api_snapshot` to add its
   signature to the snapshot. Existing entries are left as they are.
3. Document it under `docs/api_reference/` with its public import path, for example
   `::: laplace.FullLaplace`.

## Extendability

To extend the laplace package, new `BaseLaplace` subclasses can be designed, for example,
Laplace with a block-diagonal Hessian structure.
One can also implement custom subnetwork selection strategies as new subclasses of `SubnetMask`.

Alternatively, extending or integrating backends (subclasses of [`curvature.curvature`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvature.py)) allows to provide different Hessian
approximations to the Laplace approximations.
For example, currently the [`curvature.CurvlinopsInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/curvlinops.py) based on [Curvlinops](https://github.com/f-dangel/curvlinops) and the native `torch.func` (previously known as `functorch`), [`curvature.BackPackInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/backpack.py) based on [BackPACK](https://github.com/f-dangel/backpack/) and [`curvature.AsdlInterface`](https://github.com/AlexImmer/Laplace/blob/main/laplace/curvature/asdl.py) based on [ASDL](https://github.com/kazukiosawa/asdfghjkl) are available.
