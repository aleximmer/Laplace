"""The task-name API (strings, enum-style members, string comparison) warns."""

import pickle
import warnings

import pytest
import torch
from torch import nn

from laplace import (
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood,
    Laplace,
    Likelihood,
    get_likelihood,
)

LEGACY = [
    ("regression", GaussianLikelihood, "gaussian"),
    ("classification", CategoricalLikelihood, "categorical"),
    ("reward_modeling", BradleyTerryLikelihood, "bradley_terry"),
]

MEMBERS = [
    ("REGRESSION", "regression", GaussianLikelihood),
    ("CLASSIFICATION", "classification", CategoricalLikelihood),
    ("REWARD_MODELING", "reward_modeling", BradleyTerryLikelihood),
]


@pytest.mark.parametrize("name,cls,new_name", LEGACY)
def test_get_likelihood_warns_for_task_names(name, cls, new_name):
    with pytest.warns(
        DeprecationWarning, match=f'"{name}" is deprecated.*{cls.__name__}'
    ):
        lik = get_likelihood(name)

    assert type(lik) is cls
    assert lik.name == new_name


@pytest.mark.parametrize("name,cls,_", LEGACY)
def test_laplace_factory_warns_for_task_names(name, cls, _):
    model = nn.Linear(2, 2)

    with pytest.warns(DeprecationWarning, match="is deprecated"):
        la = Laplace(model, name, subset_of_weights="all", hessian_structure="diag")

    assert type(la.likelihood) is cls


@pytest.mark.parametrize(
    "name",
    [
        "bernoulli",
        "heteroscedastic_gaussian",
        "gaussian",
        "categorical",
        "bradley_terry",
        "foo",
    ],
)
def test_other_strings_raise(name):
    with pytest.raises(ValueError, match="Invalid likelihood"):
        get_likelihood(name)


@pytest.mark.parametrize("criterion", [nn.MSELoss(), nn.CrossEntropyLoss()])
def test_torch_criteria_stay_silent(criterion):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_likelihood(criterion)


@pytest.mark.parametrize("key,_,cls", MEMBERS)
def test_enum_members_warn_and_are_cached(key, _, cls):
    with pytest.warns(DeprecationWarning, match=key):
        member = getattr(Likelihood, key)

    assert type(member) is cls

    with pytest.warns(DeprecationWarning):
        assert getattr(Likelihood, key) is member


def test_bernoulli_member_warns():
    with pytest.warns(DeprecationWarning, match="BERNOULLI"):
        member = Likelihood.BERNOULLI

    assert member.name == "bernoulli"


def test_unknown_attribute_and_subclasses_stay_plain():
    with pytest.raises(AttributeError):
        Likelihood.FOO

    with pytest.raises(AttributeError):
        GaussianLikelihood.REGRESSION


@pytest.mark.parametrize("key,value,cls", MEMBERS)
def test_enum_lookups_warn(key, value, cls):
    with pytest.warns(DeprecationWarning):
        assert type(Likelihood(value)) is cls

    with pytest.warns(DeprecationWarning):
        assert type(Likelihood[key]) is cls


def test_enum_iteration_len_and_contains_warn():
    with pytest.warns(DeprecationWarning):
        assert len(list(Likelihood)) == 4

    with pytest.warns(DeprecationWarning):
        assert len(Likelihood) == 4

    with pytest.warns(DeprecationWarning):
        assert "classification" in Likelihood

    with pytest.warns(DeprecationWarning):
        assert "foo" not in Likelihood


@pytest.mark.parametrize("name,cls,new_name", LEGACY)
def test_string_comparison_and_value_warn(name, cls, new_name):
    lik = cls()

    with pytest.warns(DeprecationWarning, match="string"):
        assert lik == name

    with pytest.warns(DeprecationWarning, match="string"):
        assert lik == new_name

    with pytest.warns(DeprecationWarning, match="value"):
        assert lik.value == new_name


def test_old_checkpoint_name_loads(model, class_loader):
    la = Laplace(
        model,
        CategoricalLikelihood(),
        subset_of_weights="all",
        hessian_structure="diag",
    )
    la.fit(class_loader)

    state = la.state_dict()
    assert state["likelihood"] == "categorical"

    state["likelihood"] = "classification"
    restored = Laplace(
        model,
        CategoricalLikelihood(),
        subset_of_weights="all",
        hessian_structure="diag",
    )
    restored.load_state_dict(state)

    assert torch.allclose(restored.posterior_precision, la.posterior_precision)

    # A different likelihood still does not load
    other = Laplace(
        model, GaussianLikelihood(), subset_of_weights="all", hessian_structure="diag"
    )

    with pytest.raises(ValueError):
        other.load_state_dict(state)


class _OldEnumMember:
    """Pickles like a member of the old `Likelihood` enum."""

    def __reduce__(self):
        return (Likelihood, ("classification",))


def test_old_enum_pickle_still_loads():
    payload = pickle.dumps(_OldEnumMember())

    with pytest.warns(DeprecationWarning):
        lik = pickle.loads(payload)

    assert isinstance(lik, CategoricalLikelihood)


def test_warning_points_at_user_code():
    with pytest.warns(DeprecationWarning) as record:
        Laplace(nn.Linear(2, 2), "regression", hessian_structure="diag")

    assert record[0].filename == __file__
