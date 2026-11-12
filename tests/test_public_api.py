"""Tests for the public API.

Public names are the `__all__` of `laplace`, `laplace.curvature` and `laplace.utils`,
plus the members in `PUBLIC_COMMON` and `PUBLIC_EXTRA`. Everything else is internal,
underscore or not. Removing or renaming a public name needs a release with a
`DeprecationWarning` first, and the signature snapshot catches lost or renamed
parameters.

`INTERNAL` lists the internal members without an underscore. A member of a fitted
instance that is in none of the lists fails the tests. The 0.6 refactor shrinks
`INTERNAL` as it renames these members to `_name`.
"""

import importlib
import inspect
import json
import re
from importlib.util import find_spec
from pathlib import Path

import pytest

from tests.public_api_snapshot import SNAPSHOT_PATH, resolve, snapshot
from tests.test_contracts import (
    METHODS,
    _build,
    _data,
    _fit,
    _model,
    _param,
    _skip_lowrank_without_asdfghjkl,
)

pytestmark = pytest.mark.public_api


def _expected_curvature() -> set[str]:
    names = {
        "CurvatureInterface",
        "GGNInterface",
        "EFInterface",
        "CurvlinopsInterface",
        "CurvlinopsGGN",
        "CurvlinopsEF",
        "CurvlinopsHessian",
    }

    if find_spec("asdl") is not None:
        names |= {"AsdlInterface", "AsdlGGN", "AsdlEF", "AsdlHessian"}

    if find_spec("backpack") is not None:
        names |= {"BackPackInterface", "BackPackGGN", "BackPackEF"}

    if find_spec("asdfghjkl") is not None:
        try:
            importlib.import_module("laplace.curvature.asdfghjkl")
        except ModuleNotFoundError:
            pass
        else:
            names |= {
                "AsdfghjklInterface",
                "AsdfghjklGGN",
                "AsdfghjklEF",
                "AsdfghjklHessian",
            }

    return names


EXPECTED_EXPORTS = {
    "laplace": {
        "ELLA",
        "VaLLA",
        "Laplace",
        "BaseLaplace",
        "BaseFunctionalLaplace",
        "ParametricLaplace",
        "FullLaplace",
        "KronLaplace",
        "DiagLaplace",
        "FunctionalLaplace",
        "LowRankLaplace",
        "LLLaplace",
        "FullLLLaplace",
        "KronLLLaplace",
        "DiagLLLaplace",
        "FunctionalLLLaplace",
        "SubnetLaplace",
        "FullSubnetLaplace",
        "DiagSubnetLaplace",
        "marglik_training",
        "Likelihood",
        "GaussianLikelihood",
        "CategoricalLikelihood",
        "BernoulliLikelihood",
        "BradleyTerryLikelihood",
        "HeteroscedasticGaussianLikelihood",
        "get_likelihood",
        "SubsetOfWeights",
        "FunctionalApproximation",
        "HessianStructure",
        "PredType",
        "LinkApprox",
        "TuningMethod",
        "PriorStructure",
    },
    "laplace.utils": {
        "FeatureExtractor",
        "Kron",
        "KronDecomposed",
        "SubnetMask",
        "RandomSubnetMask",
        "LargestMagnitudeSubnetMask",
        "LargestVarianceDiagLaplaceSubnetMask",
        "LargestVarianceSWAGSubnetMask",
        "ParamNameSubnetMask",
        "ModuleNameSubnetMask",
        "LastLayerSubnetMask",
        "RunningNLLMetric",
        "RunningBernoulliNLLMetric",
        "RunningGaussianNLLMetric",
    },
    "laplace.curvature": _expected_curvature(),
}


@pytest.mark.parametrize("module", EXPECTED_EXPORTS)
def test_all_matches_expected(module):
    names = importlib.import_module(module).__all__

    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_EXPORTS[module]


@pytest.mark.parametrize("module", EXPECTED_EXPORTS)
def test_exports_resolve_and_are_public(module):
    mod = importlib.import_module(module)

    for name in mod.__all__:
        assert not name.startswith("_")
        getattr(mod, name)


@pytest.mark.parametrize("name", ["HessianStructure", "select_rows", "validate"])
def test_dropped_utils_names_warn(name):
    import laplace.utils

    with pytest.deprecated_call(match=f"laplace.utils.{name}"):
        assert getattr(laplace.utils, name) is not None


def test_dropped_utils_enum_points_to_laplace():
    import laplace
    import laplace.utils

    with pytest.deprecated_call(match="from `laplace`"):
        assert laplace.utils.PredType is laplace.PredType


def test_asdl_hessian_lazy_import():
    pytest.importorskip("asdl")

    import laplace.curvature

    assert laplace.curvature.AsdlHessian.__name__ == "AsdlHessian"


PUBLIC_COMMON = {
    "fit",
    "predictive",
    "log_marginal_likelihood",
    "optimize_prior_precision",
    "sample",
    "functional_samples",
    "predictive_samples",
    "functional_variance",
    "functional_covariance",
    "log_likelihood",
    "state_dict",
    "load_state_dict",
    "model",
    "likelihood",
    "backend",
    "prior_precision",
    "prior_mean",
    "sigma_noise",
    "temperature",
    "n_params",
    "n_data",
    "n_outputs",
}

_PARAMETRIC = {"mean", "H", "posterior_precision", "log_prob"}
_FULL = _PARAMETRIC | {"posterior_covariance", "posterior_scale"}
_DIAG = _PARAMETRIC | {"posterior_scale", "posterior_variance"}
_FUNCTIONAL = {"predictive_moments"}

PUBLIC_EXTRA = {
    "full": _FULL,
    "ll_full": _FULL,
    "subnet_full": _FULL,
    "diag": _DIAG,
    "ll_diag": _DIAG,
    "subnet_diag": _DIAG,
    "kron": _PARAMETRIC,
    "lowrank": _PARAMETRIC,
    "ll_kron": _PARAMETRIC,
    "functional": _FUNCTIONAL | {"mean"},
    "ll_functional": _FUNCTIONAL | {"mean"},
    "ella": _FUNCTIONAL | {"optimize_hyperparameters"},
    "valla": _FUNCTIONAL,
}

INTERNAL: set[str] = {
    "H_facs",
    "K_MM",
    "Kinv",
    "L",
    "Sigma_inv",
    "V",
    "alpha",
    "assemble_full_samples",
    "batch_size",
    "chol_precision",
    "damping",
    "data",
    "dict_key_x",
    "dict_key_y",
    "dual_directions",
    "enable_backprop",
    "feature_gram",
    "fit_history_",
    "functional_variance_fast",
    "generator",
    "gp_kernel_prior_variance",
    "independent_outputs",
    "inducing_classes",
    "inducing_locations",
    "is_subset_params",
    "log_det_posterior_precision",
    "log_det_prior_precision",
    "log_det_ratio",
    "log_noise_variance",
    "log_prior_precision",
    "loss",
    "mc_softmax_samples",
    "mean_subnet",
    "mu",
    "n_eigenvalues",
    "n_layers",
    "n_obs",
    "n_obs_examples",
    "n_params_subnet",
    "n_subset",
    "num_inducing",
    "params",
    "prior_precision_diag",
    "scatter",
    "seed",
    "square_norm",
    "subsample_size",
    "train_loader",
}


def _public_names(obj) -> set[str]:
    """Non-underscore names from `dir`, which never runs property getters."""
    return {n for n in dir(obj) if not n.startswith("_")}


@pytest.fixture(scope="module", params=[_param(m) for m in METHODS])
def fitted(request):
    """A fitted instance, since some attributes only exist after `fit`."""
    method = request.param
    _skip_lowrank_without_asdfghjkl(method)

    _, loader = _data()
    la = _build(method, _model(), loader)
    _fit(la, method, loader)

    return method, la


def test_public_members_exist(fitted):
    method, la = fitted

    missing = (PUBLIC_COMMON | PUBLIC_EXTRA.get(method, set())) - _public_names(la)

    assert missing == set()


def test_no_unclassified_members(fitted):
    method, la = fitted
    allowed = PUBLIC_COMMON | PUBLIC_EXTRA[method] | INTERNAL

    assert _public_names(la) - allowed == set()


# Public members of likelihoods and subnet masks, so the docs can't hide them
def test_docs_filter_matches_internal():
    config = (Path(__file__).parents[1] / "mkdocs.yml").read_text()
    names = re.search(r'"!\^\((.*)\)\$"', config).group(1).split("|")

    assert set(names) == INTERNAL


def test_public_and_internal_disjoint():
    public = PUBLIC_COMMON | set().union(*PUBLIC_EXTRA.values())

    assert not public & INTERNAL


SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text())

_POSITIONAL = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
_BY_KEYWORD = ("POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY")


def _breaking_changes(old: list[dict], new: inspect.Signature) -> list[str]:
    """List the ways `new` breaks calls that worked with the `old` parameters.

    Adding parameters with a default, or anything variadic, is allowed.
    """
    new_params = list(new.parameters.values())
    by_name = {p.name: p for p in new_params}
    new_positional = [p for p in new_params if p.kind.name in _POSITIONAL]
    problems = []

    old_positional = [p for p in old if p["kind"] in _POSITIONAL]

    for i, p in enumerate(old_positional):
        at_i = new_positional[i] if i < len(new_positional) else None

        if at_i is None or at_i.name != p["name"]:
            problems.append(f"positional parameter {i} is no longer `{p['name']}`")

    # Behind `*args`, a new positional parameter takes an argument `*args` used to get
    has_var_positional = any(p["kind"] == "VAR_POSITIONAL" for p in old)

    if has_var_positional and len(new_positional) > len(old_positional):
        problems.append("new positional parameter before `*args`")

    for p in old:
        if p["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            if not any(q.kind.name == p["kind"] for q in new_params):
                problems.append(f"`{p['name']}` ({p['kind']}) was removed")

            continue

        q = by_name.get(p["name"])

        if q is None:
            problems.append(f"`{p['name']}` was removed or renamed")
            continue

        if p["kind"] == "POSITIONAL_OR_KEYWORD" and q.kind.name != p["kind"]:
            problems.append(f"`{p['name']}` is now {q.kind.name}")

        if p["kind"] == "KEYWORD_ONLY" and q.kind.name not in _BY_KEYWORD:
            problems.append(f"`{p['name']}` is now {q.kind.name}")

        if p["has_default"] and q.default is inspect.Parameter.empty:
            problems.append(f"`{p['name']}` lost its default")

    old_names = {p["name"] for p in old}
    has_var_keyword = any(p["kind"] == "VAR_KEYWORD" for p in old)

    for q in new_params:
        if q.name in old_names or q.kind.name.startswith("VAR_"):
            continue

        if q.default is q.empty:
            problems.append(f"new parameter `{q.name}` has no default")

        # It takes the keyword argument `**kwargs` used to get
        if has_var_keyword and q.kind.name != "POSITIONAL_ONLY":
            problems.append(f"new parameter `{q.name}` before `**kwargs`")

    return problems


def _sig(source: str) -> inspect.Signature:
    namespace: dict = {}
    exec(source, namespace)

    return inspect.signature(namespace["f"])


OLD = [
    {"name": "a", "kind": "POSITIONAL_OR_KEYWORD", "has_default": False},
    {"name": "b", "kind": "POSITIONAL_OR_KEYWORD", "has_default": True},
]


@pytest.mark.parametrize(
    "new",
    [
        "def f(a, c=1): ...",
        "def f(a): ...",
        "def f(a, b): ...",
        "def f(x, a, b=1): ...",
        "def f(a, b=1, *, c): ...",
    ],
)
def test_breaking_changes_flagged(new):
    assert _breaking_changes(OLD, _sig(new)) != []


@pytest.mark.parametrize(
    "new",
    [
        "def f(a, b=1): ...",
        "def f(a, b=1, *, c=0): ...",
        "def f(a, b=1, c=0): ...",
        "def f(a, b=1, **kwargs): ...",
    ],
)
def test_compatible_changes_pass(new):
    assert _breaking_changes(OLD, _sig(new)) == []


def test_positional_before_args_flagged():
    old = [{"name": "args", "kind": "VAR_POSITIONAL", "has_default": False}]

    assert _breaking_changes(old, _sig("def f(x=0, *args): ...")) != []
    assert _breaking_changes(old, _sig("def f(*args, x=0): ...")) == []


def test_new_parameter_with_kwargs_flagged():
    old = [{"name": "kwargs", "kind": "VAR_KEYWORD", "has_default": False}]

    assert _breaking_changes(old, _sig("def f(x=0, **kwargs): ...")) != []
    assert _breaking_changes(old, _sig("def f(x=0, /, **kwargs): ...")) == []


def test_keyword_only_to_args_flagged():
    old = [{"name": "x", "kind": "KEYWORD_ONLY", "has_default": False}]

    assert _breaking_changes(old, _sig("def f(*x): ...")) != []
    assert _breaking_changes(old, _sig("def f(x): ...")) == []


def test_removed_kwargs_flagged():
    old = [{"name": "kw", "kind": "VAR_KEYWORD", "has_default": False}]

    assert _breaking_changes(old, _sig("def f(a=1): ...")) != []


@pytest.mark.parametrize("qualname", SNAPSHOT)
def test_signatures_compatible(qualname):
    module, _, rest = qualname.partition(".")

    # Optional backends are exported only when their package is installed
    if module == "curvature" and rest.split(".")[0] not in _expected_curvature():
        pytest.skip(f"`{rest}` is not installed")

    sig = inspect.signature(resolve(qualname))
    params = list(sig.parameters.values())

    if params and params[0].name == "self":
        sig = sig.replace(parameters=params[1:])

    assert _breaking_changes(SNAPSHOT[qualname], sig) == []


def test_snapshot_covers_public_callables():
    assert set(snapshot()) <= set(SNAPSHOT)
