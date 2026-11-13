import math

import pytest
import torch

from uav_abnormality.models.condition_alignment import (
    ConditionAlignment, weighted_geometric_median, weighted_median)


def scalar_reports(values, history=1):
    x = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
    x = x.expand(1, history, -1, -1).clone()
    mask = torch.ones_like(x, dtype=torch.bool)
    c = torch.arange(x.shape[2], dtype=x.dtype).reshape(1, 1, -1, 1)
    c = c.expand_as(x).clone()
    return x, mask, c, mask[..., 0].clone()


def test_weighted_median_is_midpoint_and_ignores_empty_observations():
    values = torch.tensor([[1., 3., 999.], [1., 3., 8.], [0., 1., 2.]],
                          requires_grad=True)
    weights = torch.tensor([[1., 1., 0.], [0., 2., 1.], [0., 0., 0.]])
    result = weighted_median(values, weights)
    torch.testing.assert_close(result[:2], torch.tensor([2., 3.]))
    assert torch.isnan(result[2])
    result[:2].sum().backward()
    torch.testing.assert_close(values.grad[:2],
                               torch.tensor([[.5, .5, 0.], [0., 1., 0.]]))
    assert torch.isfinite(values.grad).all()


def test_weighted_median_interval_with_nonuniform_weights():
    points = torch.tensor([0., 1., 6., 100.])
    torch.testing.assert_close(weighted_median(points, torch.tensor([1., 2., 3., 0.])),
                               torch.tensor(3.5))
    assert weighted_median(points, torch.tensor([1., 4., 1., 0.])) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_median(points, torch.tensor([1., -1., 1., 1.]))


def test_geometric_median_is_rotation_equivariant_and_scalar_path_exact():
    triangle = torch.tensor([[0., 0.], [2., 0.], [1., math.sqrt(3.)]],
                            dtype=torch.float64, requires_grad=True)
    weights = torch.ones(3, dtype=torch.float64)
    center = weighted_geometric_median(triangle, weights)
    torch.testing.assert_close(center, torch.tensor([1., math.sqrt(3.) / 3],
                                                    dtype=torch.float64),
                               atol=2e-6, rtol=2e-6)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    torch.testing.assert_close(weighted_geometric_median(triangle @ rotation, weights),
                               center @ rotation)
    center.sum().backward()
    assert torch.isfinite(triangle.grad).all()
    scalar = torch.tensor([[1.], [5.]])
    assert weighted_geometric_median(scalar, torch.ones(2)).item() == 3


def test_exact_additive_identity_and_composition():
    model = ConditionAlignment(1, 2, architecture="linear")
    with torch.no_grad():
        model.condition_model.weight.copy_(torch.tensor([[2., -1.]]))
    x, a, b, c = (torch.randn(7, 1), torch.randn(7, 2),
                  torch.randn(7, 2), torch.randn(7, 2))
    assert torch.equal(model.align(x, a, a), x)
    torch.testing.assert_close(model.align(x, a, c),
                               model.align(model.align(x, a, b), b, c))
    torch.testing.assert_close(model.align(x, a, b),
                               x + ((b - a) * torch.tensor([2., -1.])).sum(-1, keepdim=True))


def test_formula_scalar_center_error_quality_mad_and_score():
    data = scalar_reports([5., 0., 2., 4., 100.])
    model = ConditionAlignment(1, 1, tau0=.25)
    diag = model.diagnostics(*data)
    # Target 0 has peers [0,2,4,100]: center=3, MAD=median([3,1,1,97])=2.
    assert diag["reference"][0, 0, 0, 0] == 3
    assert diag["error"][0, 0] == 2
    assert diag["quality"][0, 0].item() == pytest.approx(1 / 3)
    assert diag["dispersion"][0, 0] == 2
    assert diag["score"][0, 0] == 1
    assert diag["reference_count"][0, 0] == 4
    assert diag["evaluated"].all()


def test_reference_selection_center_and_mad_exclude_target_value():
    model = ConditionAlignment(1, 1, max_neighbors=3)
    data = scalar_reports([5., 0., 2., 4., 100.], history=3)
    before = model.diagnostics(*data)
    data[0][:, :, 0] = 9000
    after = model.diagnostics(*data)
    torch.testing.assert_close(before["reference"][:, :, 0], after["reference"][:, :, 0])
    torch.testing.assert_close(before["dispersion"][:, 0], after["dispersion"][:, 0])
    torch.testing.assert_close(before["reference_count"][:, 0], after["reference_count"][:, 0])
    assert after["error"][0, 0] > before["error"][0, 0]


def test_quality_uses_fixed_error_not_adaptive_dispersion():
    model = ConditionAlignment(1, 1, tau0=.1)
    compact = model.diagnostics(*scalar_reports([3., -.1, .1]))
    diffuse = model.diagnostics(*scalar_reports([3., -2., 2.]))
    assert compact["error"][0, 0] == diffuse["error"][0, 0] == 3
    assert compact["quality"][0, 0] == diffuse["quality"][0, 0] == .25
    assert compact["score"][0, 0].item() == pytest.approx(30.)
    assert diffuse["score"][0, 0].item() == pytest.approx(1.5)


def test_condition_neighbors_ignore_values_and_use_stable_distance_ties():
    model = ConditionAlignment(1, 1, max_neighbors=2, neighbor_radius=1.5)
    x, mask, c, cm = scalar_reports([10., 0., 4., 999.])
    c[0, 0, :, 0] = torch.tensor([0., -1., 1., 2.])
    diag = model.diagnostics(x, mask, c, cm)
    assert diag["reference"][0, 0, 0, 0] == 2
    assert diag["reference_count"][0, 0] == 2
    x[0, 0, 3, 0] = -10000
    changed = model.diagnostics(x, mask, c, cm)
    torch.testing.assert_close(diag["reference"][:, :, 0], changed["reference"][:, :, 0])
    assert not diag["evaluated"][0, 1]


def test_missing_endpoint_is_unevaluated_and_nan_padding_has_finite_gradients():
    x, mask, c, cm = scalar_reports([1., 2., 3., 4.], history=3)
    mask[0, -1, 3] = False
    cm[0, 1, 1] = False
    mask[0, 0, 2] = False
    clean = (x.clone(), mask, c.clone(), cm)
    x[~mask] = float("nan")
    c[~cm] = float("nan")
    model = ConditionAlignment(1, 1, architecture="linear")
    diag = model.diagnostics(x, mask, c, cm)
    expected = model.diagnostics(*clean)
    torch.testing.assert_close(diag["score"], expected["score"], equal_nan=True)
    assert not diag["evaluated"][0, 3]
    assert torch.isnan(diag["score"][0, 3])
    model.loss(x, mask, c, cm).backward()
    assert torch.isfinite(model.condition_model.weight.grad).all()


def test_common_history_distance_and_peer_dispersion_use_same_positions():
    x, mask, c, cm = scalar_reports([5., 0., 2.], history=3)
    x[0, :, :, 0] = torch.tensor([[900., 0., 2.], [5., 0., 4.], [9., 0., 2.]])
    mask[0, 0, 0] = False
    model = ConditionAlignment(1, 1)
    diag = model.diagnostics(x, mask, c, cm)
    # Common target positions t1,t2: references 2,1; errors 3,8;
    # peer distances 2,1 for each peer, then MAD=1.5.
    assert diag["error"][0, 0].item() == pytest.approx(5.5)
    assert diag["dispersion"][0, 0].item() == pytest.approx(1.5)


def test_empty_reference_loss_is_differentiable_zero():
    data = scalar_reports([1., 3.])
    model = ConditionAlignment(1, 1)
    loss = model.loss(*data)
    assert loss == 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    diag = model.diagnostics(*data)
    assert not diag["evaluated"].any()
    assert torch.isnan(diag["quality"]).all()


def test_huber_objective_caps_large_error_gradient():
    model = ConditionAlignment(1, 1, architecture="linear", huber_delta=1)
    x, mask, c, cm = scalar_reports([100., 0., 0.])
    # All three scored: e=[100,50,50], Huber(e)=e-.5.
    assert model.loss(x, mask, c, cm).item() == pytest.approx(200 / 3 - .5)
    x.requires_grad_()
    model.loss(x, mask, c, cm).backward()
    assert x.grad.abs().max() < 1
    assert torch.isfinite(x.grad).all()


def test_linear_model_learns_known_relation_without_latent_environment_labels():
    torch.manual_seed(12)
    model = ConditionAlignment(1, 1, architecture="linear")
    c = torch.linspace(-1, 1, 7).reshape(1, 1, 7, 1).expand(8, 4, -1, -1)
    environment = torch.randn(8, 4, 1, 1) * 3
    x = environment + 1.75 * c
    mask = torch.ones_like(x, dtype=torch.bool)
    cm = mask[..., 0]
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    before = model.loss(x, mask, c, cm).item()
    for _ in range(160):
        optimizer.zero_grad()
        loss = model.loss(x, mask, c, cm)
        loss.backward()
        optimizer.step()
    assert model.loss(x, mask, c, cm).item() < before * 1e-4
    assert model.condition_model.weight.item() == pytest.approx(1.75, abs=.005)


def test_model_batching_preserves_report_results_and_has_no_history_state():
    torch.manual_seed(7)
    model = ConditionAlignment(1, 2)
    x = torch.randn(3, 4, 5, 1)
    mask = torch.ones_like(x, dtype=torch.bool)
    c = torch.randn(3, 4, 5, 2)
    cm = mask[..., 0]
    together = model.score(x, mask, c, cm)
    separate = torch.cat([model.score(x[i:i + 1], mask[i:i + 1],
                                      c[i:i + 1], cm[i:i + 1]) for i in range(3)])
    torch.testing.assert_close(together, separate)
    torch.testing.assert_close(together, model.score(x, mask, c, cm))


def test_vector_distance_and_missing_coordinate_convention():
    model = ConditionAlignment(2, 1)
    x = torch.tensor([[[[3., 4.], [0., 0.], [0., 0.], [0., 0.]]]])
    mask = torch.ones_like(x, dtype=torch.bool)
    c = torch.zeros(1, 1, 4, 1)
    cm = torch.ones(1, 1, 4, dtype=torch.bool)
    diag = model.diagnostics(x, mask, c, cm)
    assert diag["error"][0, 0].item() == pytest.approx(5 / math.sqrt(2))
    mask[0, 0, 3, 0] = False
    assert not model.diagnostics(x, mask, c, cm)["evaluated"][0, 3]
    model.loss(x, mask, c, cm).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("kwargs", [
    {"tau0": 0}, {"tau0": float("nan")}, {"huber_delta": float("inf")},
    {"neighbor_radius": -1}, {"min_references": 3, "max_neighbors": 2},
    {"architecture": "unknown"}])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        ConditionAlignment(1, 1, **kwargs)
