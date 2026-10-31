from __future__ import annotations

import math
import warnings
from collections.abc import MutableMapping
from copy import deepcopy
from importlib.util import find_spec
from itertools import product
from math import e, prod, sqrt

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2

from laplace import (
    DiagLaplace,
    DiagSubnetLaplace,
    FullLaplace,
    FullSubnetLaplace,
    FunctionalLaplace,
    KronLaplace,
    Laplace,
    LowRankLaplace,
)
from laplace.baselaplace import BaseLaplace
from laplace.curvature import AsdlEF, AsdlGGN, BackPackGGN
from laplace.curvature.backpack import BackPackEF
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN
from laplace.utils import KronDecomposed
from tests.utils import ListDataset, dict_data_collator, jacobians_naive

torch.manual_seed(240)
torch.set_default_dtype(torch.double)

flavors = [FullLaplace, KronLaplace, DiagLaplace]
if find_spec("asdfghjkl") is not None:
    flavors.append(LowRankLaplace)

online_flavors = [FullLaplace, KronLaplace, DiagLaplace]


def _bridge_from_moments(mean, covariance, link):
    class FixedMoments(BaseLaplace):
        def __init__(self):
            pass  # Only the link approximations are used

        def _glm_predictive_distribution(self, x, joint=False):
            return mean, covariance

    return BaseLaplace._glm_forward_call(
        FixedMoments(), mean, "classification", link_approx=link
    )


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize(
    "case, dtype, device, n_classes, magnitude",
    [
        pytest.param(
            "diagonal",
            torch.float32,
            "cpu",
            n,
            scale,
            id=f"cpu-float32-diagonal-{n}-{scale}",
        )
        for n, scale in product([2, 5], [1e20, 1e38])
    ]
    + [
        ("dense", torch.float32, "cpu", 5, 9e37),
        ("opposed", torch.float32, "cpu", 3, 3e38),
        ("opposed", torch.float64, "cpu", 2, 1e308),
        ("tied", torch.float64, "cpu", 2, 1e308),
        ("opposed", torch.float64, "cpu", 3, 1e308),
        ("tied", torch.float64, "cpu", 3, 1e308),
    ]
    + [
        pytest.param(
            case,
            dtype,
            f"cuda:{index}",
            3,
            magnitude,
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA unavailable"
                ),
            ],
            id=f"cuda-{index}-{dtype}-{case}",
        )
        for index in range(max(torch.cuda.device_count(), 1))
        for case, dtype, magnitude in [
            ("opposed", torch.float32, 3e38),
            ("opposed", torch.float64, 1e308),
            ("unequal_diagonal", torch.float32, 1e38),
            ("unequal_diagonal", torch.float64, 1e308),
        ]
    ],
)
def test_bridge_extreme_finite_inputs_have_finite_gradients(
    link, case, dtype, device, n_classes, magnitude
):
    if case in ("diagonal", "dense"):
        mean = torch.linspace(0.2, 0.3, n_classes, dtype=dtype, device=device)
        covariance = magnitude * torch.eye(n_classes, dtype=dtype, device=device)
        if case == "dense":
            covariance = magnitude * torch.ones_like(covariance) + covariance / 10
    elif case == "unequal_diagonal":
        mean = torch.tensor([3.0, 0.0, -1.0], dtype=dtype, device=device)
        covariance = torch.diag(
            torch.tensor(
                [magnitude, magnitude, magnitude / 2], dtype=dtype, device=device
            )
        )
    else:
        logits = (
            [magnitude, -magnitude] + [0.0] * (n_classes - 2)
            if case == "opposed"
            else [magnitude] * n_classes
        )
        mean = torch.tensor(logits, dtype=dtype, device=device)
        covariance = torch.eye(n_classes, dtype=dtype, device=device)
    mean = mean.unsqueeze(0).requires_grad_()
    covariance = covariance.unsqueeze(0).requires_grad_()
    probability = _bridge_from_moments(mean, covariance, link)
    assert torch.isfinite(probability).all()
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones_like(probability.sum(dim=-1))
    )
    gradients = torch.autograd.grad(probability[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bridge_common_mode_covariance_does_not_change_probabilities(link, dtype):
    mean = torch.tensor([[3.0, 0.0]], dtype=dtype, requires_grad=True)
    amplitude = torch.tensor(100.0, dtype=dtype, requires_grad=True)
    covariance = amplitude * torch.ones(1, 2, 2, dtype=dtype)
    probability = _bridge_from_moments(mean, covariance, link)
    torch.testing.assert_close(probability, torch.softmax(mean, dim=-1))
    gradient = torch.autograd.grad(probability[0, 0], amplitude)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize("reference", ["float64", "smaller_scale"])
def test_bridge_extreme_covariance_matches_reference(link, reference):
    dtype = torch.float32 if reference == "float64" else torch.float64
    mean = torch.tensor([[3.0, 0.0, -1.0]], dtype=dtype, requires_grad=True)
    diagonal = [2e38, 1e38, 1e38] if reference == "float64" else [1e308, 1e308, 5e307]
    covariance = torch.diag(torch.tensor(diagonal, dtype=dtype)).unsqueeze(0)
    covariance.requires_grad_()
    actual = _bridge_from_moments(mean, covariance, link)
    if reference == "float64":
        expected = _bridge_from_moments(mean.double(), covariance.double(), link)
        tolerance = 1e-6
    else:
        moderate = torch.diag(
            torch.tensor([1e100, 1e100, 5e99], dtype=dtype)
        ).unsqueeze(0)
        expected = _bridge_from_moments(mean, moderate, link)
        tolerance = 1e-12
    torch.testing.assert_close(
        actual.double(), expected, rtol=tolerance, atol=tolerance
    )
    gradients = torch.autograd.grad(actual[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def bridge_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


@pytest.fixture
def bridge_default_dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize("covariance_scale", [1e-310, 1e-308, 1e-300])
@pytest.mark.parametrize(
    "logits, diagonal",
    [
        pytest.param([0.5, -0.5], [1.0, 1.0], id="zero-sum-identity"),
        pytest.param([0.5, 0.2], [1.0, 1.0], id="nonzero-sum-identity"),
        pytest.param([0.5, 0.2], [1.0, 2.0], id="nonzero-sum-diagonal"),
    ],
)
@pytest.mark.parametrize(
    "bridge_default_dtype", [torch.float32, torch.float64], indirect=True
)
def test_bridge_tiny_covariance_warns_about_gradient_limit(
    bridge_device, bridge_default_dtype, link, covariance_scale, logits, diagonal
):
    mean = torch.tensor(
        [logits],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    shape = mean.new_tensor(diagonal)
    scale = mean.new_tensor(covariance_scale, requires_grad=True)
    covariance = (scale * shape.diag()).unsqueeze(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = _bridge_from_moments(mean, covariance, link)
    subnormal = covariance_scale < torch.finfo(covariance.dtype).tiny
    assert len(caught) == int(subnormal)
    if subnormal:
        assert caught[0].category is RuntimeWarning
        assert "subnormal covariance" in str(caught[0].message)
    # Binary conditional variances are equal, so their common scale cancels
    # from Bridge. The covariance-dependent mean correction also preserves
    # scaling; normalized Bridge saturates at these tiny scales.
    ratio = (shape[0] - shape[1]) / shape.sum()
    gap = mean.detach()[0, 0] - mean.detach()[0, 1] - ratio * mean.detach().sum()
    first_probability = gap.sigmoid() if link == "bridge" else mean.new_tensor(1.0)
    expected = torch.stack((first_probability, 1 - first_probability)).unsqueeze(0)
    torch.testing.assert_close(probability, expected, rtol=1e-12, atol=1e-12)
    mean_gradient, scale_gradient = torch.autograd.grad(
        probability[0, 0], (mean, scale)
    )
    slope = expected[:, 0] * expected[:, 1]
    expected_mean_gradient = torch.stack(
        (slope * (1 - ratio), slope * (-1 - ratio)), dim=-1
    )
    torch.testing.assert_close(
        mean_gradient, expected_mean_gradient, rtol=1e-12, atol=1e-12
    )
    unsupported_scale_gradient = (
        link == "bridge" and covariance_scale == 1e-310 and sum(logits) != 0
    )
    # Subnormal nonzero-sum inputs can overflow individual covariance derivatives
    # before their cancellation. The warning exposes this limit without promising
    # a representable scale gradient, or requiring any particular failure value.
    if not unsupported_scale_gradient:
        torch.testing.assert_close(
            scale_gradient, torch.zeros_like(scale_gradient), rtol=0, atol=0
        )


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize(
    "case, dtype, magnitude, requires_grad, grad_enabled, should_warn",
    [
        ("no-grad", torch.float64, 1e-310, True, False, False),
        ("detached-covariance", torch.float64, 1e-310, False, True, False),
        ("zero", torch.float64, 0.0, True, True, False),
        ("normal", torch.float64, 1e-300, True, True, False),
        (
            "normal-boundary",
            torch.float64,
            torch.finfo(torch.float64).tiny,
            True,
            True,
            False,
        ),
        ("single-class", torch.float64, 1e-310, True, True, False),
        ("mixed-batch", torch.float64, 1e-310, True, True, True),
        ("float32-subnormal", torch.float32, 1e-40, True, True, True),
    ],
)
def test_bridge_subnormal_covariance_warning_policy(
    bridge_device,
    link,
    case,
    dtype,
    magnitude,
    requires_grad,
    grad_enabled,
    should_warn,
):
    scales = [0.0, 1e-300, magnitude] if case == "mixed-batch" else [magnitude]
    logits = [0.5] if case == "single-class" else [0.5, 0.2]
    mean = torch.tensor(
        [logits] * len(scales),
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    covariance = torch.stack(
        [
            scale * torch.eye(len(logits), dtype=dtype, device=bridge_device)
            for scale in scales
        ]
    ).requires_grad_(requires_grad)
    mean_before, covariance_before = mean.detach().clone(), covariance.detach().clone()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.set_grad_enabled(grad_enabled):
            probability = _bridge_from_moments(mean, covariance, link)
    assert len(caught) == int(should_warn)
    if should_warn:
        assert caught[0].category is RuntimeWarning
        assert "subnormal covariance" in str(caught[0].message)
    assert torch.isfinite(probability).all()
    torch.testing.assert_close(probability.sum(dim=-1), torch.ones_like(mean[:, 0]))
    torch.testing.assert_close(mean.detach(), mean_before)
    torch.testing.assert_close(covariance.detach(), covariance_before)


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
@pytest.mark.parametrize(
    "logits, diagonal, bridge_default_dtype",
    [
        ([1.7e308, -1.7e308, -1.7e308], [1.0, 1.0, 1.0], torch.float64),
        ([-1e308, -1e308], [10.0, 1.0], torch.float64),
    ]
    + [
        pytest.param(
            [magnitude, -magnitude],
            [1.0, 1.0],
            default,
            id=f"float64-means-{magnitude}-default-{default}",
        )
        for magnitude, default in product([1e40, 1e308], [torch.float32, torch.float64])
    ],
    indirect=["bridge_default_dtype"],
)
def test_bridge_mean_correction_overflow(
    bridge_device, bridge_default_dtype, link, logits, diagonal
):
    mean = torch.tensor(
        [logits], dtype=torch.float64, device=bridge_device, requires_grad=True
    )
    covariance = torch.diag(
        torch.tensor(diagonal, dtype=torch.float64, device=bridge_device)
    ).unsqueeze(0)
    covariance.requires_grad_()
    probability = _bridge_from_moments(mean, covariance, link)
    expected = torch.zeros_like(mean)
    expected[:, 0] = 1
    torch.testing.assert_close(probability, expected)
    gradients = torch.autograd.grad(probability[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "link, dtype, logits, diagonal, expected, compare_gradients",
    [
        (link, dtype, logits, diagonal, expected, False)
        for link, dtype in product(
            ["bridge", "bridge_norm"], [torch.float32, torch.float64]
        )
        for logits, diagonal, expected in [
            ([1e20, 1e20, -1e20, -1e20], [1.0, 2.0, 3.0, 4.0], [0.64, 0.36, 0.0, 0.0]),
            ([1e20, 1e20, -2e20], [1.0, 2.0, 3.0], [8 / 13, 5 / 13, 0.0]),
        ]
    ]
    + [
        pytest.param(
            "bridge_norm",
            torch.float64,
            [1e308, 1e308, -1e308, -1e308],
            [0.01, 0.02, 0.03, 0.04],
            [0.64, 0.36, 0.0, 0.0],
            True,
            id="normalized-extreme-ties-analytical-gradients",
        )
    ],
)
def test_bridge_large_tied_logits_preserve_variance_weights(
    bridge_device, link, dtype, logits, diagonal, expected, compare_gradients
):
    mean = torch.tensor([logits], dtype=dtype, device=bridge_device, requires_grad=True)
    covariance = torch.diag(
        torch.tensor(diagonal, dtype=dtype, device=bridge_device)
    ).unsqueeze(0)
    covariance.requires_grad_()
    probability = _bridge_from_moments(mean, covariance, link)
    expected = torch.tensor([expected], dtype=dtype, device=bridge_device)
    torch.testing.assert_close(probability, expected)
    gradients = torch.autograd.grad(probability[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    if compare_gradients:
        # The two dominant classes have corrected variances 0.009 and 0.016.
        # Differentiate sigmoid((mu0 - mu1) / sqrt(c) + log(v1 / v0)),
        # including the covariance-dependent mean correction; c = 0.0175 / sqrt(2).
        mean_gradient = (0.64 * 0.36 / sqrt(0.0175 / sqrt(2))) * mean.new_tensor(
            [[1.1, -0.9, 0.1, 0.1]]
        )
        covariance_gradient = covariance.new_tensor(
            [
                [
                    [-20.16, 5.44, 5.44, 5.44],
                    [-5.44, 8.96, -5.44, -5.44],
                    [0.32, 0.32, 0.32, 0.32],
                    [0.32, 0.32, 0.32, 0.32],
                ]
            ]
        )
        for actual, expected_gradient in zip(
            gradients, (mean_gradient, covariance_gradient)
        ):
            torch.testing.assert_close(
                actual, expected_gradient, rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
def test_bridge_extreme_normalization_and_fallback_gradients(bridge_device, link):
    mean = torch.tensor(
        [[1e308, -1e308], [0.5, -0.5]],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    covariance = torch.stack(
        (
            0.01 * torch.eye(2, dtype=torch.float64, device=bridge_device),
            torch.ones(2, 2, dtype=torch.float64, device=bridge_device),
        )
    )
    covariance.requires_grad_()
    probability = _bridge_from_moments(mean, covariance, link)
    expected = torch.softmax(mean.detach(), dim=-1)
    torch.testing.assert_close(probability, expected)
    mean_gradient, covariance_gradient = torch.autograd.grad(
        probability[:, 0].sum(), (mean, covariance)
    )
    assert torch.isfinite(mean_gradient).all()
    assert torch.isfinite(covariance_gradient).all()

    # The second row has only common-mode uncertainty, so its fallback is
    # softmax(mean). Its nonzero gradients must survive the extreme first row.
    slope = expected[:, 0] * expected[:, 1]
    expected_gradient = torch.stack((slope, -slope), dim=-1)
    torch.testing.assert_close(mean_gradient, expected_gradient)
    torch.testing.assert_close(
        covariance_gradient[1].sum(),
        torch.zeros((), dtype=torch.float64, device=bridge_device),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
def test_bridge_large_common_offset_preserves_logit_differences(bridge_device, link):
    mean = torch.tensor(
        [[1e15 + 0.125, 1e15 - 0.125]],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    covariance = torch.eye(2, dtype=torch.float64, device=bridge_device).unsqueeze(0)
    covariance.requires_grad_()
    probability = _bridge_from_moments(mean, covariance, link)
    difference = 0.25 / sqrt(0.5) if link == "bridge_norm" else 0.25
    p0 = torch.tensor(difference, dtype=torch.float64, device=bridge_device).sigmoid()
    expected = torch.stack((p0, 1 - p0)).unsqueeze(0)
    torch.testing.assert_close(probability, expected)
    gradients = torch.autograd.grad(probability[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "magnitude, covariance_scale",
    [(1e154, 1e308), (1e308, 1e308), (0.5, 1e-308), (0.5, 1e-310)],
)
def test_bridge_norm_variance_scaling_preserves_logit_differences(
    bridge_device, magnitude, covariance_scale
):
    mean = torch.tensor(
        [[magnitude, -magnitude]],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    covariance = (
        covariance_scale * torch.eye(2, dtype=torch.float64, device=bridge_device)
    ).unsqueeze(0)
    covariance.requires_grad_()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability = _bridge_from_moments(mean, covariance, "bridge_norm")
    assert len(caught) == int(covariance_scale < torch.finfo(covariance.dtype).tiny)
    if caught:
        assert caught[0].category is RuntimeWarning
        assert "subnormal covariance" in str(caught[0].message)
    # For two equally uncertain centered logits, normalized Bridge is sigmoid
    # of the mean difference divided by the centered standard deviation.
    difference = torch.tensor(
        2 * (magnitude / sqrt(covariance_scale / 2)),
        dtype=torch.float64,
        device=bridge_device,
    )
    p0 = difference.sigmoid()
    expected = torch.stack((p0, 1 - p0)).unsqueeze(0)
    torch.testing.assert_close(probability, expected)
    gradients = torch.autograd.grad(probability[0, 0], (mean, covariance))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if covariance_scale < 1:
        for gradient in gradients:
            torch.testing.assert_close(
                gradient, torch.zeros_like(gradient), rtol=0, atol=0
            )


@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
def test_bridge_gradcheck(bridge_device, link):
    mean = torch.tensor(
        [[0.2, -0.4, 0.8]],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    covariance = torch.tensor(
        [[[2.0, 0.25, 0.1], [0.25, 1.0, 0.05], [0.1, 0.05, 3.0]]],
        dtype=torch.float64,
        device=bridge_device,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda mu, cov: _bridge_from_moments(mu, cov, link), (mean, covariance)
    )


@pytest.mark.parametrize("method", ["sod", "diag"])
@pytest.mark.parametrize("link", ["bridge", "bridge_norm"])
def test_bridge_probabilities_with_zero_sum_logit_covariance(method, link):
    class ContrastModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.5]))

        def forward(self, inputs):
            logit = inputs * self.weight
            return torch.cat((logit, -logit), dim=-1)

    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    loader = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 0])), batch_size=3)
    if method == "diag":
        estimator = Laplace(
            ContrastModel(), "classification", "all", "diag", backend=CurvlinopsGGN
        )
    else:
        estimator = Laplace(
            ContrastModel(),
            "classification",
            "all",
            "gp",
            backend=CurvlinopsGGN,
            n_subset=2,
        )
    estimator.fit(loader)
    probabilities = estimator(
        inputs[:1], pred_type="glm" if method == "diag" else "gp", link_approx=link
    )
    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, "output_size", 2)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model_1d():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 1))
    setattr(model, "output_size", 1)
    model_params = list(model.parameters())
    setattr(model, "n_layers", len(model_params))  # number of parameter groups
    setattr(model, "n_params", len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def large_model():
    model = wide_resnet50_2()
    return model


@pytest.fixture
def custom_model():
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 100), nn.ReLU(), nn.Linear(100, 2))

        def forward(self, data: MutableMapping | torch.Tensor):
            if isinstance(data, MutableMapping):
                x = data["test_input_key"].to(next(self.parameters()).device)
            else:
                x = data

            logits = self.net(x)
            return logits

    return CustomModel()


@pytest.fixture
def reward_model():
    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 1))

        def forward(self, x):
            """
            x: torch.Tensor
                If training == True then shape (batch_size, 2, dim)
                Else shape (batch_size, dim)
            """
            if len(x.shape) == 3:
                batch_size, _, dim = x.shape

                # Flatten to (batch_size*2, dim)
                flat_x = x.reshape(-1, dim)

                # Forward
                flat_logits = self.net(flat_x)  # (batch_size*2, 1)

                # Reshape back to (batch_size, 2)
                return flat_logits.reshape(batch_size, 2)
            else:
                logits = self.net(x)  # (batch_size, 1)
                return logits

    return RewardModel()


@pytest.fixture
def multidim_model():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(8, num_heads=1, batch_first=True)
            self.final_layer = nn.Linear(8, 2)

        def forward(self, x):
            """x is (B, L, D)"""
            out = self.attn(x, x, x, need_weights=False)[0]  # (B, L, D)
            return self.final_layer(out)  # (B, L, O)

    model = Model()

    for mod_name, mod in model.named_modules():
        if mod_name == "final_layer":
            for p in mod.parameters():
                p.requires_grad = True
        else:
            for p in mod.parameters():
                p.requires_grad = False

    return model


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader_1d():
    torch.manual_seed(9999)
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader_1d_flat():
    torch.manual_seed(9999)
    X = torch.randn(10, 3)
    y = torch.randn((10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader_multidim():
    X = torch.randn(10, 6, 8)
    y = torch.randn(10, 6, 2)
    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)


@pytest.fixture
def custom_loader_clf():
    data = []
    for _ in range(10):
        datum = {
            "test_input_key": torch.randn(5),
            "test_label_key": torch.randint(2, (1,)),
        }
        data.append(datum)
    return DataLoader(ListDataset(data), batch_size=3, collate_fn=dict_data_collator)


@pytest.fixture
def custom_loader_reg():
    data = []
    for _ in range(10):
        datum = {
            "test_input_key": torch.randn(5),
            "test_label_key": torch.randn(2),
        }
        data.append(datum)
    return DataLoader(ListDataset(data), batch_size=3, collate_fn=dict_data_collator)


@pytest.fixture
def reward_loader():
    X = torch.randn(10, 2, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reward_test_X():
    X = torch.randn(10, 3)
    return X


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init(laplace, model):
    lap = laplace(model, "classification")
    assert torch.allclose(lap.mean, lap.prior_mean)
    if laplace in [FullLaplace, DiagLaplace]:
        H = lap.H.clone()
        lap._init_H()
        assert torch.allclose(H, lap.H)
    elif laplace == LowRankLaplace:
        assert lap.H is None
    else:
        H = [[k.clone() for k in kfac] for kfac in lap.H.kfacs]
        lap._init_H()
        for kfac1, kfac2 in zip(H, lap.H.kfacs):
            for k1, k2 in zip(kfac1, kfac2):
                assert torch.allclose(k1, k2)


@pytest.mark.skip(reason="Does not work well with Github actions")
def test_laplace_large_init(large_model):
    FullLaplace(large_model, "classification")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_invalid_likelihood(laplace, model):
    with pytest.raises(ValueError):
        laplace(model, "otherlh")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_noise(laplace, model):
    # float
    sigma_noise = 1.2
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # torch.tensor 0-dim
    sigma_noise = torch.tensor(1.2)
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # torch.tensor 1-dim
    sigma_noise = torch.tensor(1.2).reshape(-1)
    laplace(model, likelihood="regression", sigma_noise=sigma_noise)

    # for classification should fail
    sigma_noise = 1.2
    with pytest.raises(ValueError):
        laplace(model, likelihood="classification", sigma_noise=sigma_noise)

    # other than that should fail
    # higher dim
    sigma_noise = torch.tensor(1.2).reshape(1, 1)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", sigma_noise=sigma_noise)
    # other datatype, only reals supported
    sigma_noise = "1.2"
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", sigma_noise=sigma_noise)


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_precision(laplace, model):
    # float
    precision = 10.6
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 0-dim
    precision = torch.tensor(10.6)
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim
    precision = torch.tensor(10.7).reshape(-1)
    laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim param-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
    if laplace == KronLaplace:
        # Kron should not accept per parameter prior precision
        with pytest.raises(ValueError):
            laplace(model, likelihood="regression", prior_precision=precision)
    else:
        laplace(model, likelihood="regression", prior_precision=precision)
    # torch.tensor 1-dim layer-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_layers)
    laplace(model, likelihood="regression", prior_precision=precision)

    # other than that should fail
    # higher dim
    precision = torch.tensor(10.6).reshape(1, 1)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)
    # unmatched dim
    precision = torch.tensor(10.6).reshape(-1).repeat(17)
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)
    # other datatype, only reals supported
    precision = "1.5"
    with pytest.raises(ValueError):
        laplace(model, likelihood="regression", prior_precision=precision)


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_prior_mean_and_scatter(laplace, model, class_loader):
    mean = parameters_to_vector(model.parameters())
    P = len(mean)
    lap_scalar_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=1.0
    )
    assert torch.allclose(lap_scalar_mean.prior_mean, torch.tensor([1.0]))
    lap_tensor_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(1)
    )
    assert torch.allclose(lap_tensor_mean.prior_mean, torch.tensor([1.0]))
    lap_tensor_scalar_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(1)[0]
    )
    assert torch.allclose(lap_tensor_scalar_mean.prior_mean, torch.tensor(1.0))
    lap_tensor_full_mean = laplace(
        model, "classification", prior_precision=1e-2, prior_mean=torch.ones(P)
    )
    assert torch.allclose(lap_tensor_full_mean.prior_mean, torch.ones(P))

    lap_scalar_mean.fit(class_loader)
    lap_tensor_mean.fit(class_loader)
    lap_tensor_scalar_mean.fit(class_loader)
    lap_tensor_full_mean.fit(class_loader)
    expected = torch.tensor(0).reshape(-1)
    # assert expected.ndim == 0
    expected = ((mean - 1) * 1e-2) @ (mean - 1)
    assert torch.allclose(lap_scalar_mean.scatter, expected)
    assert lap_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_mean.scatter, expected)
    assert lap_tensor_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_scalar_mean.scatter, expected)
    assert lap_tensor_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_full_mean.scatter, expected)
    assert lap_tensor_full_mean.scatter.shape == expected.shape

    # too many dims
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P).unsqueeze(-1)
        laplace(model, "classification", prior_precision=1e-2, prior_mean=prior_mean)

    # unmatched dim
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P - 3)
        laplace(model, "classification", prior_precision=1e-2, prior_mean=prior_mean)

    # invalid argument type
    with pytest.raises(ValueError):
        laplace(model, "classification", prior_precision=1e-2, prior_mean="72")


@pytest.mark.parametrize("laplace", flavors)
def test_laplace_init_temperature(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood="classification", temperature=T)
    assert lap.temperature == T


@pytest.mark.parametrize(
    "laplace,lh", product(flavors, ["classification", "regression"])
)
def test_laplace_functionality(laplace, lh, model, reg_loader, class_loader):
    if lh == "classification":
        loader = class_loader
        sigma_noise = 1.0
    else:
        loader = reg_loader
        sigma_noise = 0.3
    lap = laplace(model, lh, sigma_noise=sigma_noise, prior_precision=0.7)
    lap.fit(loader)
    assert lap.n_data == len(loader.dataset)
    assert lap.n_outputs == model.output_size
    f = model(loader.dataset.tensors[0])
    y = loader.dataset.tensors[1]
    assert f.shape == torch.Size([10, 2])

    # Test log likelihood (Train)
    log_lik = lap.log_likelihood
    # compute true log lik
    if lh == "classification":
        log_lik_true = Categorical(logits=f).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
    else:
        assert y.size() == f.size()
        log_lik_true = Normal(loc=f, scale=sigma_noise).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
        # change likelihood and test again
        lap.sigma_noise = 0.72
        log_lik = lap.log_likelihood
        log_lik_true = Normal(loc=f, scale=0.72).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)

    # Test marginal likelihood
    # lml = log p(y|f) - 1/2 theta @ prior_prec @ theta
    #       + 1/2 logdet prior_prec - 1/2 log det post_prec
    lml = log_lik_true
    theta = parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(theta, lap.mean)
    prior_prec = torch.diag(lap.prior_precision_diag)
    assert prior_prec.shape == torch.Size([len(theta), len(theta)])
    lml = lml - 1 / 2 * theta @ prior_prec @ theta
    if laplace == DiagLaplace:
        log_det_post_prec = lap.posterior_precision.log().sum()
    elif laplace == LowRankLaplace:
        (U, eigval), p0 = lap.posterior_precision
        log_det_post_prec = (U @ torch.diag(eigval) @ U.T + p0.diag()).logdet()
    else:
        log_det_post_prec = lap.posterior_precision.logdet()
    lml = lml + 1 / 2 * (prior_prec.logdet() - log_det_post_prec)
    assert torch.allclose(lml, lap.log_marginal_likelihood())

    # test functional variance
    if laplace == FullLaplace:
        Sigma = lap.posterior_covariance
    elif laplace == KronLaplace:
        Sigma = lap.posterior_precision.to_matrix(exponent=-1)
    elif laplace == LowRankLaplace:
        (U, eigval), p0 = lap.posterior_precision
        Sigma = (U @ torch.diag(eigval) @ U.T + p0.diag()).inverse()
    elif laplace == DiagLaplace:
        Sigma = torch.diag(lap.posterior_variance)
    Js, f = jacobians_naive(model, loader.dataset.tensors[0])
    true_f_var = torch.einsum("mkp,pq,mcq->mkc", Js, Sigma, Js)
    comp_f_var = lap.functional_variance(Js)
    assert torch.allclose(true_f_var, comp_f_var, rtol=1e-4)

    # test sampling
    torch.manual_seed(61)
    samples = lap.sample(n_samples=1)
    assert samples.shape == torch.Size([1, len(theta)])
    samples = lap.sample(n_samples=1_000_000)
    assert samples.shape == torch.Size([1_000_000, len(theta)])
    assert torch.allclose(samples.mean(dim=0), lap.mean, rtol=0, atol=1e-2)
    assert torch.allclose(samples.T.cov(), Sigma, rtol=0, atol=1e-2)


@pytest.mark.parametrize("laplace", online_flavors)
def test_overriding_fit(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    if type(lap.posterior_precision) is KronDecomposed:
        P = lap.posterior_precision.to_matrix()
    else:
        P = lap.posterior_precision.clone()
    m = lap.mean.clone()
    marglik = lap.log_marginal_likelihood().detach().clone()
    assert lap.n_data == len(reg_loader.dataset)
    lap.fit(reg_loader, override=True)
    assert torch.allclose(lap.mean, m)
    if type(lap.posterior_precision) is KronDecomposed:
        assert torch.allclose(lap.posterior_precision.to_matrix(), P)
    else:
        assert torch.allclose(lap.posterior_precision, P)
    assert torch.allclose(marglik, lap.log_marginal_likelihood())
    assert lap.n_data == len(reg_loader.dataset)


@pytest.mark.parametrize("laplace", online_flavors)
def test_online_fit(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    if type(lap.H) is KronDecomposed:
        P = lap.H.to_matrix().clone()
    else:
        P = lap.H.clone()
    loss, n_data = deepcopy(lap.loss.item()), deepcopy(lap.n_data)
    # fit a second and third time but don't override
    lap.fit(reg_loader, override=False)
    lap.fit(reg_loader, override=False)
    # Hessian should be now roughly 3x the one before
    assert torch.allclose(3 * torch.tensor(loss), lap.loss)
    assert (3 * n_data) == lap.n_data
    if type(lap.H) is KronDecomposed:
        assert torch.allclose(lap.H.to_matrix(), 3 * P)
    else:
        assert torch.allclose(lap.H, 3 * P)


def test_log_prob_full(model, class_loader):
    lap = FullLaplace(model, "classification", prior_precision=0.7)
    theta = torch.randn_like(parameters_to_vector(model.parameters()))
    # posterior without fitting is just prior
    posterior = Normal(loc=torch.zeros_like(theta), scale=sqrt(1 / 0.7))
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta).sum())
    lap.fit(class_loader)
    posterior = MultivariateNormal(
        loc=lap.mean, precision_matrix=lap.posterior_precision
    )
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta))


def test_log_prob_kron(model, class_loader):
    lap = KronLaplace(model, "classification", prior_precision=0.24)
    theta = torch.randn_like(parameters_to_vector(model.parameters()))
    posterior = Normal(loc=lap.mean, scale=sqrt(1 / 0.24))
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta).sum())
    lap.fit(class_loader)
    posterior = MultivariateNormal(
        loc=lap.mean, precision_matrix=lap.posterior_precision.to_matrix()
    )
    assert torch.allclose(lap.log_prob(theta), posterior.log_prob(theta))


@pytest.mark.parametrize("laplace", flavors)
def test_regression_predictive(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    f_mu_glm, f_var_glm = lap(X, pred_type="glm")
    assert torch.allclose(f_mu_glm, f)
    assert f_var_glm.shape == torch.Size(
        [f_mu_glm.shape[0], f_mu_glm.shape[1], f_mu_glm.shape[1]]
    )
    assert len(f_mu_glm) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu_nn, f_var_nn = lap(X, pred_type="nn", link_approx="mc")
    assert f_mu_nn.shape == f_var_nn.shape
    assert f_var_nn.shape == torch.Size([f_mu_nn.shape[0], f_mu_nn.shape[1]])
    assert len(f_mu_nn) == len(X)

    # Test joint prediction
    f_mu_joint, f_cov_joint = lap(X, pred_type="glm", joint=True)
    assert len(f_mu_joint.shape) == 1
    assert f_mu_joint.shape[0] == prod(f_mu_glm.shape)
    assert len(f_cov_joint.shape) == 2
    assert f_cov_joint.shape == (f_mu_joint.shape[0], f_mu_joint.shape[0])

    # The "diagonal" of the joint cov should equal the non-joint var
    b, k = y.shape
    f_var_joint = torch.einsum("bkbl->bkl", f_cov_joint.reshape(b, k, b, k))
    assert torch.allclose(f_var_joint, f_var_glm)


@pytest.mark.parametrize("laplace", flavors)
def test_classification_predictive(laplace, model, class_loader):
    lap = laplace(model, "classification", prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    f_pred = lap(X, pred_type="glm", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="probit")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1
    f_pred = lap(X, pred_type="glm", link_approx="bridge_norm")
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1

    # NN predictive
    f_pred = lap(X, pred_type="nn", link_approx="mc", n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(
        f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
    )  # sum up to 1


@pytest.mark.parametrize("laplace", flavors)
def test_regression_predictive_samples(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    fsamples = lap.predictive_samples(X, pred_type="glm", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # NN predictive (only diagonal variance estimation)
    fsamples = lap.predictive_samples(X, pred_type="nn", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])


@pytest.mark.parametrize("laplace", flavors)
def test_classification_predictive_samples(laplace, model, class_loader):
    lap = laplace(model, "classification", prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type="linear")

    # GLM predictive
    fsamples = lap.predictive_samples(X, pred_type="glm", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1

    # NN predictive
    fsamples = lap.predictive_samples(X, pred_type="nn", n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1


@pytest.mark.parametrize("laplace", flavors)
def test_functional_samples(laplace, model, reg_loader):
    lap = laplace(model, "regression", sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    generator = torch.Generator()

    fsamples_reg_glm = lap.functional_samples(
        X, pred_type="glm", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_reg_glm.shape == torch.Size([100, f.shape[0], f.shape[1]])

    fsamples_reg_nn = lap.functional_samples(
        X, pred_type="nn", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_reg_nn.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # The samples should not be affected by the likelihood
    lap.likelihood = "classification"

    fsamples_clf_glm = lap.functional_samples(
        X, pred_type="glm", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_clf_glm.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert torch.allclose(fsamples_clf_glm, fsamples_reg_glm)

    fsamples_clf_nn = lap.functional_samples(
        X, pred_type="nn", n_samples=100, generator=generator.manual_seed(123)
    )
    assert fsamples_clf_nn.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert torch.allclose(fsamples_clf_nn, fsamples_reg_nn)


@pytest.mark.parametrize("laplace", [KronLaplace, DiagLaplace])
def test_reward_modeling(laplace, reward_model, reward_loader, reward_test_X):
    lap = laplace(reward_model, "reward_modeling")
    lap.fit(reward_loader)
    f = reward_model(reward_test_X)

    # error
    with pytest.raises(ValueError):
        lap(reward_test_X, pred_type="linear")

    # GLM predictive, functional variance tested already above.
    f_mu, f_var = lap(reward_test_X, pred_type="glm")
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(reward_test_X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(reward_test_X, pred_type="nn", link_approx="mc")
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(reward_test_X)


@pytest.mark.parametrize("backend", [AsdlGGN, AsdlEF])
def test_asdl_different_numbers_of_rows_raises(backend):
    # Token-wise layer (B * L rows), then a head after pooling over tokens (B rows)
    B, L, D, K = 4, 3, 5, 2

    class PooledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(D, 6)
            self.head = nn.Linear(6, K)

        def forward(self, x):
            return self.head(torch.tanh(self.first(x)).mean(1))

    model = PooledModel()
    setattr(model, "output_size", K)
    x, y = torch.randn(B, L, D), torch.randn(B, K)

    with pytest.raises(RuntimeError, match="Curvlinops"):
        backend(model, "regression").full(x, y, N=B)


@pytest.mark.parametrize(
    "asdl_backend, curvlinops_backend",
    [(AsdlGGN, CurvlinopsGGN), (AsdlEF, CurvlinopsEF)],
)
def test_asdl_pooled_outputs_match_curvlinops(asdl_backend, curvlinops_backend):
    # Tokens `(B, L, D)` -> shared layer -> pooled over tokens `(B, K)`: the
    # observations are the sequences, not the tokens
    B, L, D, K = 4, 3, 5, 2

    class PooledModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, K)

        def forward(self, x):
            return self.lin(x["input"]).mean(1)

    model = PooledModel()
    setattr(model, "output_size", K)
    data = {"input": torch.randn(B, L, D), "output": torch.randn(B, K)}
    kwargs = dict(dict_key_x="input", dict_key_y="output")

    asdl = asdl_backend(model, "regression", **kwargs)
    curvlinops = curvlinops_backend(model, "regression", **kwargs)

    loss_asdl, H_asdl = asdl.full(data, data["output"], N=B)
    loss_curvlinops, H_curvlinops = curvlinops.full(data, data["output"], N=B)
    assert torch.allclose(loss_asdl, loss_curvlinops)
    assert torch.allclose(H_asdl, H_curvlinops)

    Js, f = asdl.jacobians(data)
    Js_ref, f_ref = asdl.functorch_jacobians(data)
    assert Js.shape == (B, K, model.lin.weight.numel() + K)
    assert torch.allclose(f, f_ref)
    assert torch.allclose(Js, Js_ref)


def test_reward_modeling_full_asdl_ggn_matches_curvlinops(reward_model, reward_loader):
    # Pairs are fed as `2B` rows to the layers, but each pair is one observation
    la_asdl = FullLaplace(reward_model, "reward_modeling", backend=AsdlGGN)
    la_asdl.fit(reward_loader)
    la_curvlinops = FullLaplace(
        deepcopy(reward_model), "reward_modeling", backend=CurvlinopsGGN
    )
    la_curvlinops.fit(reward_loader)

    assert torch.allclose(la_asdl.H, la_curvlinops.H)


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace])
@pytest.mark.parametrize("diagonal_output", [True, False])
@pytest.mark.parametrize("method", ["functional_samples", "predictive_samples"])
def test_multidim_glm_samples(laplace, diagonal_output, method):
    N, L, D, K, n_samples = 12, 3, 4, 2, 7
    X, Y = torch.randn(N, D), torch.randn(N, K)
    X_multidim, Y_multidim = X.reshape(N // L, L, D), Y.reshape(N // L, L, K)

    model_flat = nn.Linear(D, K)
    model_multidim = deepcopy(model_flat)
    la_flat = laplace(model_flat, "regression", backend=CurvlinopsGGN)
    la_flat.fit(DataLoader(TensorDataset(X, Y), batch_size=N))
    la_multidim = laplace(model_multidim, "regression", backend=CurvlinopsGGN)
    la_multidim.fit(DataLoader(TensorDataset(X_multidim, Y_multidim), batch_size=N))

    kwargs = dict(pred_type="glm", n_samples=n_samples, diagonal_output=diagonal_output)
    samples_flat = getattr(la_flat, method)(
        X, generator=torch.Generator().manual_seed(0), **kwargs
    )
    samples_multidim = getattr(la_multidim, method)(
        X_multidim, generator=torch.Generator().manual_seed(0), **kwargs
    )

    assert samples_multidim.shape == (n_samples, N // L, L, K)
    assert torch.allclose(samples_flat, samples_multidim.reshape(n_samples, N, K))


@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN])
def test_multidim_kron_asdl_raises(multidim_model, reg_loader_multidim, backend):
    lap = KronLaplace(multidim_model, "regression", backend=backend)
    with pytest.raises(NotImplementedError, match="Curvlinops"):
        lap.fit(reg_loader_multidim)


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace])
@pytest.mark.parametrize("backend", [BackPackGGN, BackPackEF])
@pytest.mark.parametrize("likelihood", ["regression", "classification"])
def test_multidim_backpack_raises(laplace, backend, likelihood):
    X = torch.randn(10, 6, 8)
    y = (
        torch.randn(10, 6, 2)
        if likelihood == "regression"
        else torch.randint(2, (10, 6))
    )
    lap = laplace(nn.Linear(8, 2), likelihood, backend=backend)
    with pytest.raises(NotImplementedError, match="BackPack"):
        lap.fit(DataLoader(TensorDataset(X, y), batch_size=4))


def test_multidim_last_layer_glm_raises(reg_loader_multidim):
    model = nn.Sequential(nn.Linear(8, 5), nn.Tanh(), nn.Linear(5, 2))
    lap = Laplace(
        model,
        "regression",
        subset_of_weights="last_layer",
        hessian_structure="full",
        backend=CurvlinopsGGN,
    )
    lap.fit(reg_loader_multidim)
    x, _ = next(iter(reg_loader_multidim))

    with pytest.raises(NotImplementedError, match="Last-layer"):
        lap(x, pred_type="glm")


def _dense(H: torch.Tensor | KronDecomposed) -> torch.Tensor:
    return H.to_matrix() if isinstance(H, KronDecomposed) else H


@pytest.mark.parametrize(
    "laplace",
    [FullLaplace, DiagLaplace, KronLaplace, FullSubnetLaplace, DiagSubnetLaplace],
)
@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN])
@pytest.mark.parametrize("model_type", ["linear", "mlp"])
def test_multidim_classification_equivalent_to_flat(laplace, backend, model_type):
    """Classification with logits `(B, L, C)` and targets `(B, L)`, e.g. next-token
    prediction, must give the same posterior and predictive as the flat data set with
    logits `(B * L, C)`, since every token is an observation."""
    N, L, D, C = 12, 3, 4, 5
    X, y = torch.randn(N, D), torch.randint(C, (N,))
    X_multidim, y_multidim = X.reshape(N // L, L, D), y.reshape(N // L, L)

    loader_flat = DataLoader(TensorDataset(X, y), batch_size=N)
    loader_multidim = DataLoader(
        TensorDataset(X_multidim, y_multidim), batch_size=N // L
    )

    if model_type == "linear":
        model_flat = nn.Linear(D, C)
    else:
        model_flat = nn.Sequential(nn.Linear(D, 5), nn.Tanh(), nn.Linear(5, C))

    model_multidim = deepcopy(model_flat)

    kwargs = {}
    if laplace in (FullSubnetLaplace, DiagSubnetLaplace):
        # Include the output bias, so that the predictive covariance has full rank
        n_params = sum(p.numel() for p in model_flat.parameters())
        kwargs["subnetwork_indices"] = torch.LongTensor(
            [0, 3, 8, *range(n_params - C, n_params)]
        )

    la_flat = laplace(model_flat, "classification", backend=backend, **kwargs)
    la_flat.fit(loader_flat)

    la_multidim = laplace(model_multidim, "classification", backend=backend, **kwargs)
    la_multidim.fit(loader_multidim)

    # Posterior
    assert la_multidim.n_data == N // L
    assert la_flat.n_obs == la_multidim.n_obs == N
    assert torch.allclose(la_flat.loss, la_multidim.loss)
    assert torch.allclose(la_flat.log_likelihood, la_multidim.log_likelihood)
    assert torch.allclose(_dense(la_flat.H), _dense(la_multidim.H))
    assert torch.allclose(
        la_flat.log_marginal_likelihood(), la_multidim.log_marginal_likelihood()
    )

    # GLM predictive
    for link_approx in ["probit", "bridge", "bridge_norm", "mc"]:
        torch.manual_seed(0)
        probs_flat = la_flat(X, pred_type="glm", link_approx=link_approx)
        torch.manual_seed(0)
        probs_multidim = la_multidim(
            X_multidim, pred_type="glm", link_approx=link_approx
        )
        assert probs_multidim.shape == (N // L, L, C)
        assert torch.allclose(probs_flat, probs_multidim.reshape(N, C))

    # NN predictive: same seed, same posterior samples
    torch.manual_seed(0)
    probs_flat = la_flat(X, pred_type="nn", link_approx="mc")
    torch.manual_seed(0)
    probs_multidim = la_multidim(X_multidim, pred_type="nn", link_approx="mc")
    assert torch.allclose(probs_flat, probs_multidim.reshape(N, C))


@pytest.mark.parametrize("method", ["marglik", "gridsearch"])
def test_multidim_classification_prior_precision_equivalent_to_flat(method):
    N, L, D, C = 12, 3, 4, 5
    X, y = torch.randn(N, D), torch.randint(C, (N,))
    X_multidim, y_multidim = X.reshape(N // L, L, D), y.reshape(N // L, L)

    loader_flat = DataLoader(TensorDataset(X, y), batch_size=N)
    loader_multidim = DataLoader(
        TensorDataset(X_multidim, y_multidim), batch_size=N // L
    )

    model_flat = nn.Linear(D, C)
    la_flat = DiagLaplace(model_flat, "classification", backend=CurvlinopsGGN)
    la_flat.fit(loader_flat)
    la_multidim = DiagLaplace(
        deepcopy(model_flat), "classification", backend=CurvlinopsGGN
    )
    la_multidim.fit(loader_multidim)

    kwargs = {"method": method}
    if method == "gridsearch":
        kwargs["n_steps"] = 10
    la_flat.optimize_prior_precision(val_loader=loader_flat, **kwargs)
    la_multidim.optimize_prior_precision(val_loader=loader_multidim, **kwargs)

    assert torch.allclose(la_flat.prior_precision, la_multidim.prior_precision)


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace, KronLaplace])
@pytest.mark.parametrize("y_shape", [(8,), (8, 3, 1), (8, 3, 4)])
def test_multidim_classification_wrong_target_shape_raises(laplace, y_shape):
    X = torch.randn(8, 3, 4)
    y = torch.randint(5, y_shape)
    lap = laplace(nn.Linear(4, 5), "classification", backend=CurvlinopsGGN)

    with pytest.raises(ValueError, match="targets"):
        lap.fit(DataLoader(TensorDataset(X, y), batch_size=4))


@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_multidim_functional_laplace_raises(likelihood):
    X = torch.randn(8, 3, 4)
    y = (
        torch.randint(2, (8, 3))
        if likelihood == "classification"
        else torch.randn(8, 3, 2)
    )
    lap = FunctionalLaplace(nn.Linear(4, 2), likelihood, n_subset=4)

    with pytest.raises(NotImplementedError, match="FunctionalLaplace"):
        lap.fit(DataLoader(TensorDataset(X, y), batch_size=4))


def test_asdl_sum_rows_per_observation():
    A = torch.arange(12.0).reshape(6, 2)
    assert torch.equal(AsdlGGN._sum_rows_per_observation(A, 6), A)
    assert torch.equal(
        AsdlGGN._sum_rows_per_observation(A, 3), A.reshape(3, 2, 2).sum(1)
    )

    with pytest.raises(ValueError):
        AsdlGGN._sum_rows_per_observation(A, 4)  # not a multiple
    with pytest.raises(ValueError):
        AsdlGGN._sum_rows_per_observation(A, 12)  # fewer rows than observations


@pytest.mark.parametrize("laplace", [FullSubnetLaplace, DiagSubnetLaplace])
@pytest.mark.parametrize("backend", [CurvlinopsGGN, AsdlGGN])
def test_multidim_subnetwork_equivalent_to_flat(laplace, backend):
    N, L, D, K = 12, 3, 4, 2
    X, Y = torch.randn(N, D), torch.randn(N, K)
    X_multidim, Y_multidim = X.reshape(N // L, L, D), Y.reshape(N // L, L, K)
    indices = torch.LongTensor([0, 3, 5, 8])

    model_flat = nn.Linear(D, K)
    model_multidim = deepcopy(model_flat)
    la_flat = laplace(
        model_flat, "regression", subnetwork_indices=indices, backend=backend
    )
    la_flat.fit(DataLoader(TensorDataset(X, Y), batch_size=N))
    la_multidim = laplace(
        model_multidim, "regression", subnetwork_indices=indices, backend=backend
    )
    la_multidim.fit(DataLoader(TensorDataset(X_multidim, Y_multidim), batch_size=N))

    assert torch.allclose(la_flat.H, la_multidim.H)

    mean_flat, var_flat = la_flat(X, pred_type="glm")
    mean_multidim, var_multidim = la_multidim(X_multidim, pred_type="glm")
    assert torch.allclose(mean_flat, mean_multidim.reshape(N, K))
    assert torch.allclose(var_flat, var_multidim.reshape(N, K, K))


@pytest.mark.parametrize("backend", [AsdlGGN, CurvlinopsGGN])
def test_multidim_variable_length_batches(backend):
    # Sequence lengths differ between batches, e.g. when padding to the longest sequence
    D, K = 4, 2
    lengths = [(2, 3), (3, 1), (1, 5)]  # (n_sequences, sequence_length) per batch
    batches = [(torch.randn(b, L, D), torch.randn(b, L, K)) for b, L in lengths]
    X = torch.cat([x.reshape(-1, D) for x, _ in batches])
    Y = torch.cat([y.reshape(-1, K) for _, y in batches])

    model_flat = nn.Linear(D, K)
    model_multidim = deepcopy(model_flat)

    la_flat = FullLaplace(model_flat, "regression", backend=backend)
    la_flat.fit(DataLoader(TensorDataset(X, Y), batch_size=len(X)))
    la_multidim = FullLaplace(model_multidim, "regression", backend=backend)
    la_multidim.fit(DataLoader(batches, batch_size=None))

    assert la_multidim.n_obs == la_flat.n_obs == len(X)
    assert torch.allclose(la_flat.loss, la_multidim.loss)
    assert torch.allclose(la_flat.H, la_multidim.H)
    assert torch.allclose(la_flat.log_likelihood, la_multidim.log_likelihood)
    assert torch.allclose(
        la_flat.log_marginal_likelihood(), la_multidim.log_marginal_likelihood()
    )


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace])
@pytest.mark.parametrize("backend", [AsdlGGN, AsdlEF, CurvlinopsGGN, CurvlinopsEF])
def test_multidim_classification_variable_length_batches(laplace, backend):
    D, C = 4, 3
    lengths = [(2, 3), (3, 1), (1, 5)]  # (n_sequences, sequence_length) per batch
    batches = [(torch.randn(b, L, D), torch.randint(C, (b, L))) for b, L in lengths]
    X = torch.cat([x.reshape(-1, D) for x, _ in batches])
    y = torch.cat([y.reshape(-1) for _, y in batches])

    model_flat = nn.Linear(D, C)
    model_multidim = deepcopy(model_flat)

    la_flat = laplace(model_flat, "classification", backend=backend)
    la_flat.fit(DataLoader(TensorDataset(X, y), batch_size=len(X)))
    la_multidim = laplace(model_multidim, "classification", backend=backend)
    la_multidim.fit(DataLoader(batches, batch_size=None))

    assert la_multidim.n_obs == la_flat.n_obs == len(X)
    assert torch.allclose(la_flat.loss, la_multidim.loss)
    assert torch.allclose(la_flat.H, la_multidim.H)
    assert torch.allclose(
        la_flat.log_marginal_likelihood(), la_multidim.log_marginal_likelihood()
    )


@pytest.mark.parametrize("backend", [AsdlGGN, AsdlEF])
@pytest.mark.parametrize("seq_len", [1, 5])
def test_asdl_jacobians_token_ids(backend, seq_len):
    # Token ids `(B, L)` -> outputs `(B, L, K)`: the token axis must be kept, even if L=1
    B, V, D, K = 4, 11, 6, 3

    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.emb.weight.requires_grad = False
            self.lin = nn.Linear(D, K)

        def forward(self, ids):
            return self.lin(self.emb(ids))

    model = TokenModel()
    setattr(model, "output_size", K)
    ids = torch.randint(V, (B, seq_len))
    interface = backend(model, "regression")

    Js, f = interface.jacobians(ids)
    Js_ref, f_ref = interface.functorch_jacobians(ids)

    assert Js.shape == (B, seq_len, K, K * D + K)
    assert torch.allclose(f, f_ref)
    assert torch.allclose(Js, Js_ref)


@pytest.mark.parametrize("backend", [AsdlGGN, AsdlEF])
def test_asdl_jacobians_stale_output_size(backend, reward_model, reward_test_X):
    # E.g. after pairwise fitting in reward modeling, `output_size` is 2 but
    # the model outputs 1 value per item at prediction time
    setattr(reward_model, "output_size", 2)
    interface = backend(reward_model, "regression")

    Js, f = interface.jacobians(reward_test_X)
    Js_ref, f_ref = interface.functorch_jacobians(reward_test_X)

    assert Js.shape == Js_ref.shape
    assert torch.allclose(f, f_ref)
    assert torch.allclose(Js, Js_ref)


@pytest.mark.parametrize("backend", [AsdlGGN, AsdlEF])
@pytest.mark.parametrize("enable_backprop", [True, False])
def test_reward_modeling_asdl_glm(
    backend, enable_backprop, reward_model, reward_loader, reward_test_X
):
    # Pairwise fitting and single-item prediction have different output sizes
    lap = DiagLaplace(
        reward_model,
        "reward_modeling",
        backend=backend,
        enable_backprop=enable_backprop,
    )
    lap.fit(reward_loader)

    f_mu, f_var = lap(reward_test_X, pred_type="glm")
    assert torch.allclose(f_mu, reward_model(reward_test_X))
    assert f_var.shape == (len(reward_test_X), 1, 1)


@pytest.mark.parametrize("laplace", [KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN, BackPackGGN, BackPackEF]
)
@pytest.mark.parametrize(
    "lik,custom_loader",
    [
        ("classification", "custom_loader_clf"),
        ("regression", "custom_loader_reg"),
        ("reward_modeling", "custom_loader_clf"),
    ],
)
def test_dict_data(laplace, backend, lik, custom_loader, custom_model, request):
    custom_loader = request.getfixturevalue(custom_loader)

    if "backpack" not in backend.__name__.lower():
        with pytest.raises(KeyError):
            # Raises an error since custom_loader's input is under the key 'test_input_key'
            # but the default is 'input_ids'
            lap = laplace(custom_model, lik, backend=backend)
            lap.fit(custom_loader)

    lap = laplace(
        custom_model,
        lik,
        backend=backend,
        dict_key_x="test_input_key",
        dict_key_y="test_label_key",
    )

    if "backpack" in backend.__name__.lower():
        # Unsupported, thus raises an exception
        with pytest.raises(ValueError):
            lap.fit(custom_loader)

        return

    lap.fit(custom_loader)

    test_data = next(iter(custom_loader))
    f = custom_model(test_data)

    if lik == "classification":
        f_pred = lap(test_data, pred_type="glm")
        assert f_pred.shape == f.shape
        assert torch.allclose(
            f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
        )  # sum up to 1

        f_pred = lap(test_data, pred_type="nn", link_approx="mc")
        assert f_pred.shape == f.shape
        assert torch.allclose(
            f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double)
        )
    else:
        f_pred, f_var = lap(test_data, pred_type="glm")
        assert f_pred.shape == f.shape
        assert torch.allclose(f_pred, f)
        assert f_var.shape == (f_pred.shape[0], f_pred.shape[1], f_pred.shape[1])

        f_pred, f_var = lap(test_data, pred_type="nn", link_approx="mc")
        assert f_pred.shape == f.shape
        assert f_var.shape == (f_pred.shape[0], f_pred.shape[1])


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="glm")

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm_joint(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_cov = lap(X, pred_type="glm", joint=True)

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_cov.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_glm_mc(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="glm", link_approx="mc")

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
@pytest.mark.parametrize(
    "backend", [BackPackGGN, CurvlinopsGGN, CurvlinopsEF, AsdlGGN, AsdlEF]
)
def test_backprop_nn(laplace, model, reg_loader, backend):
    X, y = reg_loader.dataset.tensors
    X.requires_grad = True

    lap = laplace(model, "regression", enable_backprop=True, backend=backend)
    lap.fit(reg_loader)
    f_mu, f_var = lap(X, pred_type="nn", link_approx="mc", n_samples=10)

    try:
        grad_X_mu = torch.autograd.grad(f_mu.sum(), X, retain_graph=True)[0]
        grad_X_var = torch.autograd.grad(f_var.sum(), X)[0]

        assert grad_X_mu.shape == X.shape
        assert grad_X_var.shape == X.shape
    except ValueError:
        assert False


@pytest.mark.parametrize("laplace", [FullLaplace, KronLaplace, DiagLaplace])
def test_reg_glm_predictive_correct_behavior(laplace, model, reg_loader):
    X, y = reg_loader.dataset.tensors
    n_batch = X.shape[0]
    n_outputs = y.shape[-1]

    lap = laplace(model, "regression")
    lap.fit(reg_loader)

    # Joint predictive ignores diagonal_output
    f_mean, f_var = lap(X, pred_type="glm", joint=True, diagonal_output=True)
    assert f_var.shape == (n_batch * n_outputs, n_batch * n_outputs)

    f_mean, f_var = lap(X, pred_type="glm", joint=True, diagonal_output=False)
    assert f_var.shape == (n_batch * n_outputs, n_batch * n_outputs)

    # diagonal_output affects non-joint
    f_mean, f_var = lap(X, pred_type="glm", joint=False, diagonal_output=True)
    assert f_var.shape == (n_batch, n_outputs)

    f_mean, f_var = lap(X, pred_type="glm", joint=False, diagonal_output=False)
    assert f_var.shape == (n_batch, n_outputs, n_outputs)


@pytest.mark.parametrize(
    "likelihood,custom_loader",
    [
        ("classification", "custom_loader_clf"),
        ("regression", "custom_loader_reg"),
        ("reward_modeling", "custom_loader_clf"),
    ],
)
def test_dict_data_diagEF_curvlinops(custom_model, custom_loader, likelihood, request):
    custom_loader = request.getfixturevalue(custom_loader)
    lap = DiagLaplace(
        custom_model,
        likelihood=likelihood,
        backend=CurvlinopsEF,
        dict_key_x="test_input_key",
        dict_key_y="test_label_key",
    )
    lap.fit(custom_loader)

    # Must match the same data passed as tensors
    batches = list(custom_loader)
    X = torch.cat([b["test_input_key"] for b in batches])
    y = torch.cat(
        [b["test_label_key"].reshape(len(b["test_input_key"]), -1) for b in batches]
    )
    y = y.squeeze(-1) if likelihood != "regression" else y
    lap_tensor = DiagLaplace(custom_model, likelihood=likelihood, backend=CurvlinopsEF)
    lap_tensor.fit(DataLoader(TensorDataset(X, y), batch_size=3))

    assert torch.allclose(lap.H, lap_tensor.H)
    assert torch.allclose(lap.loss, lap_tensor.loss)


@pytest.mark.parametrize(
    "likelihood", ["classification", "regression", "reward_modeling"]
)
@pytest.mark.parametrize("prior_prec_type", ["scalar", "layerwise", "diag"])
def test_gridsearch(model, likelihood, prior_prec_type, reg_loader, class_loader):
    if likelihood == "regression":
        dataloader = reg_loader
    else:
        dataloader = class_loader

    if prior_prec_type == "scalar":
        prior_prec = 1.0
    elif prior_prec_type == "layerwise":
        prior_prec = torch.ones(model.n_layers)
    else:
        prior_prec = torch.ones(model.n_params)

    lap = DiagLaplace(model, likelihood, prior_precision=prior_prec)
    lap.fit(dataloader)

    # Should not raise an error
    lap.optimize_prior_precision(method="gridsearch", val_loader=dataloader, n_steps=10)


@pytest.mark.parametrize(
    "base,expected",
    [(None, [1.0, 10.0, 100.0]), (e, [1.0, e, e**2])],
)
@pytest.mark.parametrize("laplace_class", [DiagLaplace, FunctionalLaplace])
def test_gridsearch_log_prior_prec_base(
    model, reg_loader, monkeypatch, base, expected, laplace_class
):
    lap = (
        FunctionalLaplace(model, "regression", n_subset=3)
        if laplace_class is FunctionalLaplace
        else DiagLaplace(model, "regression")
    )
    intervals = []

    def capture_interval(loss, interval, val_loader, **kwargs):
        intervals.append(interval)
        return torch.tensor(1.0)

    monkeypatch.setattr(lap, "_gridsearch", capture_interval)
    if laplace_class is FunctionalLaplace:
        monkeypatch.setattr(lap, "_build_Sigma_inv", lambda: None)
    base_kwargs = {} if base is None else {"log_prior_prec_base": base}
    lap.optimize_prior_precision(
        method="gridsearch",
        val_loader=reg_loader,
        loss=lambda _: 0.0,
        log_prior_prec_min=0,
        log_prior_prec_max=2,
        grid_size=3,
        **base_kwargs,
    )

    assert len(intervals) == 1
    torch.testing.assert_close(intervals[0], torch.tensor(expected))


def test_gridsearch_custom_base_end_to_end(model, reg_loader):
    lap = DiagLaplace(model, "regression")
    lap.fit(reg_loader)
    lap.optimize_prior_precision(
        method="gridsearch",
        val_loader=reg_loader,
        log_prior_prec_min=0,
        log_prior_prec_max=1,
        grid_size=2,
        log_prior_prec_base=e,
    )

    candidates = torch.tensor([1.0, e], dtype=lap.prior_precision.dtype)
    assert torch.any(torch.isclose(lap.prior_precision, candidates))


@pytest.mark.parametrize("laplace", flavors)
def test_parametric_fit_y_shape(model_1d, reg_loader_1d, reg_loader_1d_flat, laplace):
    lap = laplace(model_1d, likelihood="regression")
    lap.fit(reg_loader_1d)  # OK

    lap2 = laplace(model_1d, likelihood="regression")

    with pytest.raises(ValueError):
        lap2.fit(reg_loader_1d_flat)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN])
@pytest.mark.parametrize("enable_backprop", [True, False])
def test_functional_variance_multidim(
    multidim_model, reg_loader_multidim, laplace, backend, enable_backprop
):
    if laplace in [KronLaplace, LowRankLaplace]:
        pytest.skip("Kron/LowRankLaplace doesn't support multidim batch yet with GLM.")

    lap = laplace(
        multidim_model, "regression", backend=backend, enable_backprop=enable_backprop
    )
    lap.fit(reg_loader_multidim)

    x, _ = next(iter(reg_loader_multidim))

    pred_mean, pred_var = lap(x, pred_type="glm", joint=False)
    assert pred_mean.shape == (*x.shape[:-1], lap.n_outputs)
    assert pred_var.shape == (*x.shape[:-1], lap.n_outputs, lap.n_outputs)

    pred_mean, pred_var = lap(x, pred_type="glm", joint=False, diagonal_output=True)
    assert pred_mean.shape == (*x.shape[:-1], lap.n_outputs)
    assert pred_var.shape == (*x.shape[:-1], lap.n_outputs)

    pred_mean, pred_var = lap(x, pred_type="nn", link_approx="mc", n_samples=5)
    assert pred_mean.shape == (*x.shape[:-1], lap.n_outputs)
    assert pred_var.shape == (*x.shape[:-1], lap.n_outputs)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN])
@pytest.mark.parametrize("enable_backprop", [True, False])
def test_functional_variance_multidim_nn(
    multidim_model, reg_loader_multidim, laplace, backend, enable_backprop
):
    if laplace in [LowRankLaplace]:
        pytest.skip("LowRankLaplace doesn't support multidim batch yet.")

    if laplace is KronLaplace and backend in [AsdlEF, AsdlGGN]:
        pytest.skip("Kron with ASDL doesn't support multidim batch (tested below).")

    lap = laplace(
        multidim_model, "regression", backend=backend, enable_backprop=enable_backprop
    )
    lap.fit(reg_loader_multidim)

    x, _ = next(iter(reg_loader_multidim))

    pred_mean, pred_var = lap(x, pred_type="nn", link_approx="mc", n_samples=5)
    assert pred_mean.shape == (*x.shape[:-1], lap.n_outputs)
    assert pred_var.shape == (*x.shape[:-1], lap.n_outputs)


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN])
@pytest.mark.parametrize("enable_backprop", [True, False])
def test_functional_covariance_multidim(
    multidim_model, reg_loader_multidim, laplace, backend, enable_backprop
):
    if laplace in [KronLaplace, LowRankLaplace]:
        pytest.skip("Kron/LowRankLaplace doesn't support multidim batch yet.")

    lap = laplace(
        multidim_model, "regression", backend=backend, enable_backprop=enable_backprop
    )
    lap.fit(reg_loader_multidim)

    x, _ = next(iter(reg_loader_multidim))
    pred_mean, pred_cov = lap(x, pred_type="glm", joint=True)

    assert pred_mean.shape == (math.prod(x.shape[:-1]) * lap.n_outputs,)
    assert pred_cov.shape == (
        math.prod(x.shape[:-1]) * lap.n_outputs,
        math.prod(x.shape[:-1]) * lap.n_outputs,
    )


@pytest.mark.parametrize("laplace", [FullLaplace, DiagLaplace, KronLaplace])
@pytest.mark.parametrize("backend", [AsdlEF, AsdlGGN, CurvlinopsEF, CurvlinopsGGN])
@pytest.mark.parametrize("model_type", ["linear", "mlp"])
def test_multidim_equivalent_to_flat(laplace, backend, model_type):
    """A pointwise model on a reshaped data set must give the same posterior and
    predictive as on the original flat data set."""
    if laplace is KronLaplace and backend in (AsdlEF, AsdlGGN):
        pytest.skip("ASDL Kron raises for multidim regression")

    N, L, D, K = 12, 3, 4, 2
    X, Y = torch.randn(N, D), torch.randn(N, K)
    X_multidim, Y_multidim = X.reshape(N // L, L, D), Y.reshape(N // L, L, K)

    loader_flat = DataLoader(TensorDataset(X, Y), batch_size=N)
    loader_multidim = DataLoader(
        TensorDataset(X_multidim, Y_multidim), batch_size=N // L
    )

    if model_type == "linear":
        model_flat = nn.Linear(D, K)
    else:
        model_flat = nn.Sequential(nn.Linear(D, 5), nn.Tanh(), nn.Linear(5, K))

    model_multidim = deepcopy(model_flat)

    la_flat = laplace(model_flat, "regression", backend=backend)
    la_flat.fit(loader_flat)

    la_multidim = laplace(model_multidim, "regression", backend=backend)
    la_multidim.fit(loader_multidim)

    # Posterior: `n_data` counts sequences, but the likelihood counts all "tokens"
    # Count `n_obs` instead, i.e., in the case of `(N, L, D)` data, `n_obs = N*L`.
    assert la_multidim.n_data == N // L
    assert la_flat.n_obs == la_multidim.n_obs
    assert torch.allclose(la_flat.loss, la_multidim.loss)
    assert torch.allclose(la_flat.log_likelihood, la_multidim.log_likelihood)
    assert torch.allclose(_dense(la_flat.H), _dense(la_multidim.H))
    assert torch.allclose(
        la_flat.log_marginal_likelihood(), la_multidim.log_marginal_likelihood()
    )

    # GLM predictive (marginal)
    mean_flat, var_flat = la_flat(X, pred_type="glm")
    mean_multidim, var_multidim = la_multidim(X_multidim, pred_type="glm")
    assert mean_multidim.shape == (N // L, L, K)
    assert var_multidim.shape == (N // L, L, K, K)
    assert torch.allclose(mean_flat, mean_multidim.reshape(N, K))
    assert torch.allclose(var_flat, var_multidim.reshape(N, K, K))

    # GLM predictive (joint)
    mean_flat, cov_flat = la_flat(X, pred_type="glm", joint=True)
    mean_multidim, cov_multidim = la_multidim(X_multidim, pred_type="glm", joint=True)
    assert torch.allclose(mean_flat, mean_multidim)
    assert torch.allclose(cov_flat, cov_multidim)

    # NN predictive: same seed, same posterior samples
    torch.manual_seed(0)
    mean_flat, var_flat = la_flat(X, pred_type="nn", link_approx="mc")
    torch.manual_seed(0)
    mean_multidim, var_multidim = la_multidim(
        X_multidim, pred_type="nn", link_approx="mc"
    )
    assert torch.allclose(mean_flat, mean_multidim.reshape(N, K))
    assert torch.allclose(var_flat, var_multidim.reshape(N, K))


@pytest.mark.parametrize("laplace", flavors)
@pytest.mark.parametrize(
    "backend", [AsdlEF, AsdlGGN, BackPackEF, BackPackGGN, CurvlinopsEF, CurvlinopsGGN]
)
@pytest.mark.parametrize("dtype", [torch.half, torch.float, torch.double])
@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_dtype(laplace, backend, dtype, likelihood):
    X = torch.randn((10, 3), dtype=dtype)
    Y = torch.randn((10, 3), dtype=dtype)

    data = TensorDataset(X, Y)
    dataloader = DataLoader(data, batch_size=10)

    model = nn.Linear(3, 3, dtype=dtype)

    try:
        la = laplace(model, likelihood, backend=backend)
        la.fit(dataloader)

        assert la.H is not None

        if isinstance(la.H, torch.Tensor):
            assert la.H.dtype == dtype
        elif isinstance(la.H, KronDecomposed):
            assert la.H.eigenvalues[0][0].dtype == dtype
            assert la.H.eigenvectors[0][0].dtype == dtype

        assert la.log_marginal_likelihood().dtype == dtype

        y_pred, y_var = la(X, pred_type="glm")
        assert y_pred.dtype == dtype
        assert y_var.dtype == dtype

        y_pred = la(X, pred_type="nn", num_samples=3)
        assert y_pred.dtype == dtype
    except (ValueError, AttributeError, RuntimeError, SystemExit) as e:
        if "must have the same dtype" in str(e):
            assert False  # Fail the test
        else:
            pass  # Ignore
