import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_abnormality.models.prbtd import PRBTD


def load_upstream(relative_path, name):
    path = Path(__file__).resolve().parents[1] / "third_party" / "PRBTD" / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(**kwargs):
    torch.manual_seed(5)
    return PRBTD(3, 2, 4, d_model=4, d_inner=8, dk_s=2, dk_t=2,
                 growth_rate=2, **kwargs)


def test_predictor_is_causal_and_trainable():
    model = make_model()
    x = torch.randn(2, 4, 3, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    loss = model.loss(x, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.predictor.mconv3.weight.grad.abs().sum() > 0
    model.eval()
    changed = x.clone()
    changed[:, -1] += 100
    torch.testing.assert_close(model(x, mask), model(changed, mask))


def test_missing_values_do_not_affect_prediction_or_score():
    model = make_model().eval()
    x = torch.randn(2, 4, 3, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, 2] = False
    changed = x.clone()
    changed[~mask] = float("nan")
    torch.testing.assert_close(model.score(x, mask), model.score(changed, mask))
    assert torch.equal(model.score(x, mask)[:, 2], torch.zeros(2))


def test_stream_batching_and_reset():
    model = make_model().eval()
    x = torch.randn(3, 4, 3, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    batched = model.stream_score(x, mask)
    final_reputation = model.reputation
    model.reset_state()
    individual = torch.cat([model.stream_score(x[i:i+1], mask[i:i+1]) for i in range(3)])
    torch.testing.assert_close(batched, individual)
    torch.testing.assert_close(final_reputation, model.reputation)
    model.score(x, mask)
    torch.testing.assert_close(final_reputation, model.reputation)
    model.reset_state()
    assert model.reputation is None


def test_pure_score_has_no_batch_dependent_state():
    model = make_model().eval()
    x = torch.randn(3, 4, 3, 2)
    mask = torch.ones_like(x, dtype=torch.bool)
    together = model.score(x, mask)
    separately = torch.cat([model.score(x[i:i+1], mask[i:i+1]) for i in range(3)])
    torch.testing.assert_close(together, separately)


def test_signed_implication_penalizes_contradicting_source():
    model = make_model().eval()
    state = model._new_state(torch.zeros(1), 1)
    prediction = torch.zeros(1, 3, 2)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    value = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]]])
    for _ in range(8):
        scores = model._step(value, prediction, mask, state)
    assert scores[0, 2] > scores[0, :2].max()
    assert state["reputation"][0, 2] < state["reputation"][0, :2].min()


def test_scalar_reputation_and_implication_match_upstream_equations():
    model = make_model()
    reputation = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    quality = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    expected = []
    for rep, q in zip(reputation.tolist(), quality.tolist()):
        rate = 0.018 if q >= 0.5 else 0.06
        expected.append(0.5 + (rate * (q - 0.5) * 2 + (1 - rate) * (2 * rep - 1)) * 0.5)
    torch.testing.assert_close(model._update_reputation(reputation, quality),
                               torch.tensor(expected, dtype=torch.float64))
    error = torch.tensor([[[0.1], [0.2], [-0.8]]], dtype=torch.float64)
    mask = torch.ones_like(error, dtype=torch.bool)
    env = torch.tensor([[0.15]], dtype=torch.float64)
    actual = model._implication_matrix(error, mask, env, [(error, mask)])[0].numpy()
    vectors = np.array([[-0.05, 0.15], [0.05, 0.15], [-0.95, 0.15]])
    expected = vectors @ vectors.T / np.outer(np.linalg.norm(vectors, axis=1),
                                             np.linalg.norm(vectors, axis=1))
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_checkpoint_roundtrip_and_all_missing_loss():
    model = make_model()
    x = torch.zeros(2, 4, 3, 2)
    mask = torch.zeros_like(x, dtype=torch.bool)
    loss = model.loss(x, mask)
    loss.backward()
    assert loss.item() == 0
    other = make_model()
    other.load_state_dict(model.state_dict())
    model.eval()
    other.eval()
    observed = torch.ones_like(mask)
    torch.testing.assert_close(model.score(x, observed), other.score(x, observed))


def test_predictor_matches_pinned_upstream_before_output_activation():
    module = load_upstream("Code/Predict/utils/transformer.py", "original_cfsttn")
    model = make_model().eval()
    upstream = module.Transformer((1, 3, 1, 1, 3), [], [], growth_rate=2,
                                  num_init_features=2, d_model=4, d_inner=8,
                                  layers=1, nheads_spatial=2, nheads_temporal=2,
                                  dk_s=2, dk_t=2, drop_rate=0.0).eval()
    upstream.load_state_dict(model.predictor.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 1, 1, 3)
    with torch.no_grad():
        original, _ = upstream(inputs)
        adapted, _ = model.predictor(inputs)
    torch.testing.assert_close(original, adapted.sigmoid())


def test_scalar_truth_discovery_matches_pinned_upstream():
    upstream = load_upstream("Code/Method/PRBTD.py", "original_prbtd")
    model = make_model(residual_mode="relative")
    values = [1.1, 1.2, 0.2]
    reports = [[0, 0, value, 1.0] for value in values]
    rep = {i: 0.5 for i in range(3)}
    rep, quality = upstream.TD(reports, list(range(3)), upstream.getError([reports]),
                               rep, {}, [reports], [list(range(3))])
    # Two identical standardized features have the same cosine as one feature.
    value = torch.tensor(values, dtype=torch.float64).reshape(1, 3, 1).repeat(1, 1, 2)
    mask = torch.ones_like(value, dtype=torch.bool)
    state = model._new_state(value, 1)
    scores = model._step(value, torch.ones_like(value), mask, state)
    np.testing.assert_allclose(scores[0].numpy(), [1 - quality[i] for i in range(3)], atol=1e-10)
    np.testing.assert_allclose(state["reputation"][0].numpy(), [rep[i] for i in range(3)], atol=1e-10)


def test_bad_shape_is_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="Expected"):
        model.loss(torch.zeros(2, 3, 3, 2), torch.ones(2, 3, 3, 2))
