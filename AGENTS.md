# AGENTS.md

`laplace-torch`: Laplace approximations for PyTorch networks (full net, subnetwork, or last layer).
Weight-space and function-space (GP, ELLA, VaLLA) posteriors, marginal likelihood, and predictives.
Python >= 3.11. Full docs: `docs/` (served via mkdocs), dev guide: `docs/devs_guide.md`.

## Setup

Always use `uv run` or `uv venv`, don't use global Python.

```bash
uv sync --all-extras          # or: pip install -e ".[dev,all]"
uv run python examples/regression_example.py
```

On Linux, the default PyTorch wheel pulls in CUDA even without an NVIDIA GPU.
For a CPU-only install (as in CI):

```bash
uv venv
uv pip install --torch-backend cpu -e ".[dev,all]"
uv run --no-sync pytest ...      # or: export UV_NO_SYNC=1
```

`uv.lock` pins the CUDA build, so in this setup never run `uv sync`, and pass `--no-sync`
to every `uv run`. Otherwise uv reinstalls the CUDA packages.

ASDL and BackPACK backends are extras (`asdl`, `backpack`, `all`); Curvlinops is the default.
Optional `asdfghjkl` backend: `pip install git+https://git@github.com/wiseodd/asdl@asdfghjkl`.

## Layout

- `laplace/laplace.py`: `Laplace(...)` factory, the user-facing entry point.
- `laplace/baselaplace.py`: `BaseLaplace` and subclasses per Hessian structure
  (`full`, `kron`, `lowrank`, `diag`, `gp`); also `BaseFunctionalLaplace`, ELLA, VaLLA.
- `laplace/lllaplace.py`, `laplace/subnetlaplace.py`: last-layer and subnetwork variants.
- `laplace/curvature/`: Hessian/GGN backends (Curvlinops, BackPACK, ASDL, asdfghjkl).
- `laplace/utils/`: feature extractor, Kronecker matrices, subnet masks, enums, functional helpers.
- `tests/`, `examples/`, `docs/` (incl. `docs/api_reference/*.md`).

## Commands

```bash
make ruff                        # ruff check --fix, then ruff format (order matters)
uv run pytest tests/test_x.py -k "..."   # targeted tests
uv run mkdocs build --strict     # docs, as checked in CI
```

- Full suite takes ~5 min; prefer targeted tests (`-k`, single files) for the touched area.
- If `--extra dev` fails to resolve locally (`tensordict` wheel), use
  `uv run --with pytest --with scipy --with pytest-mock pytest ...`.
- Tests marked `cuda` (e.g. `tests/test_*_cuda.py`) need an NVIDIA GPU; skip with `-m "not cuda"`.

## Conventions

- Use blank lines liberally to separate logical blocks; whitespace aids readability.
- Fix all ruff lint warnings manually; do not suppress them.
- Add docstrings and type hints for all changes. New public classes go in `docs/api_reference/*.md`
  and the `mkdocs.yml` nav.
- Add an example under `examples/` when introducing new features, see the current examples. Use it as both tutorial and API explanation. Make the writing flow.
- Run docstrings, comments, and docs through the
  [humanizer](https://github.com/blader/humanizer) skill to strip AI-sounding prose.
- Add tests for every change; all must pass.
- Data loaders yield `(X, y)` or dict-like batches (keys `dict_key_x`, `dict_key_y`);
  `len(loader.dataset)` must work.

## Function-space methods

- `hessian_structure="gp"` with `functional_approximation="sod" | "nystrom" | "variational"`
  selects `FunctionalLaplace`, `ELLA`, `VaLLA`.
- New methods implement `fit`, `_glm_predictive_distribution`, and their own checkpoint state;
  do not inherit another method's marginal-likelihood formula.
- Predictive moments: mean `[batch, outputs]`, cov `[batch, outputs, outputs]`. With `joint=True`:
  mean `[batch * outputs]`, cov `[batch * outputs, batch * outputs]`, ordered input-then-output.
  Moments exclude observation noise.
- Implement `_refresh_posterior` if hyperparameter changes invalidate cached factors.
- Shared helpers live in `laplace/utils/functional_utils.py`; underscore-prefixed ones are private.
