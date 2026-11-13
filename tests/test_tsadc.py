"""Behavioral tests for source-level TSAD-C; no benchmark claims."""
import io

import pytest
import torch

from uav_abnormality.models.tsadc import DenseGINE, TSADC


def tiny_model(**kwargs):
    torch.set_num_threads(1)
    config = dict(n_sources=3, n_features=2, window_size=8, hidden_dim=4,
                  state_dim=4, num_res_layers=1, res_channels=4,
                  skip_channels=4, diffusion_steps=2, score_samples=1)
    config.update(kwargs)
    return TSADC(**config)


def sample():
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(2, 8, 3, 2, generator=generator)
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[0, :3, 1] = False
    observed[1, :, 2] = False
    return values, observed


def test_all_three_training_components_and_gradients():
    torch.manual_seed(1)
    model = tiny_model()
    x, mask = sample()
    parts = model.loss_components(x, mask)
    assert set(parts) == {"noise", "graph", "reconstruction"}
    loss = sum(parts.values())
    assert torch.isfinite(loss)
    loss.backward()
    for branch in (model.decontaminator, model.temporal, model.graph_learner, model.gnn):
        gradients = [p.grad for p in branch.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()
    assert torch.isfinite(model.loss(x, mask))


def test_scores_reproducible_batch_invariant_and_rng_unchanged():
    model = tiny_model().eval()
    x, mask = sample()
    rng = torch.random.get_rng_state().clone()
    score = model.score(x, mask)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert score.shape == (2, 3)
    assert torch.isfinite(score).all()
    assert (score >= 0).all()
    assert score[1, 2] == 0
    assert torch.allclose(score, model.score(x, mask), atol=1e-6)
    separate = torch.cat([model.score(x[i:i+1], mask[i:i+1]) for i in range(2)])
    assert torch.allclose(score, separate, atol=1e-5, rtol=1e-5)
    assert torch.allclose(score.flip(0), model.score(x.flip(0), mask.flip(0)), atol=1e-5)


def test_missing_values_are_not_training_targets_or_scores():
    model = tiny_model().eval()
    x, mask = sample()
    changed = x.clone()
    changed[~mask] = float("nan")
    assert torch.allclose(model.score(x, mask), model.score(changed, mask), atol=1e-6)
    torch.manual_seed(19)
    before = model.loss(x, mask)
    torch.manual_seed(19)
    after = model.loss(changed, mask)
    assert torch.allclose(before, after, atol=1e-6)
    empty_mask = torch.zeros_like(mask)
    assert model.loss(x, empty_mask) == 0
    assert torch.equal(model.score(x, empty_mask), torch.zeros(2, 3))


def test_checkpoint_reproduces_scores():
    model = tiny_model()
    x, mask = sample()
    expected = model.score(x, mask)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = tiny_model()
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    assert torch.allclose(expected, restored.score(x, mask), atol=1e-6)
    assert model.training and restored.training


def test_dense_gine_matches_sparse_edge_message_definition():
    layer = DenseGINE(3, train_eps=True)
    nodes = torch.randn(2, 4, 3, requires_grad=True)
    adjacency = torch.tensor([[[1., .7, 0., .1], [0., 1., .3, 0.],
                               [.3, 0., 1., 0.], [0., 0., .4, 1.]]]).repeat(2, 1, 1)
    expected_batches = []
    for batch in range(2):
        targets = []
        for target in range(4):
            aggregate = torch.zeros(3)
            for source in range(4):
                weight = adjacency[batch, source, target]
                if weight != 0:
                    aggregate = aggregate + torch.relu(nodes[batch, source] + layer.edge_encoder(weight.reshape(1)))
            targets.append(layer.network((1 + layer.eps) * nodes[batch, target] + aggregate))
        expected_batches.append(torch.stack(targets))
    expected = torch.stack(expected_batches)
    actual = layer(nodes, adjacency)
    assert torch.allclose(actual, expected, atol=1e-6)
    gradient_actual = torch.autograd.grad(actual.square().sum(), nodes, retain_graph=True)[0]
    gradient_expected = torch.autograd.grad(expected.square().sum(), nodes)[0]
    assert torch.allclose(gradient_actual, gradient_expected, atol=1e-6)


@pytest.mark.parametrize("masking", ["rm", "rbm", "bom"])
def test_sparse_mask_always_supplies_observed_training_target(masking):
    model = tiny_model(masking=masking)
    observed = torch.zeros(2, 6, 8, dtype=torch.bool)
    observed[0, 0, 0] = True
    observed[1, 5, 7] = True
    conditioning = model._conditioning_mask(observed)
    assert (observed & ~conditioning).flatten(1).any(-1).all()
    assert not (conditioning & ~observed).any()


def test_invalid_observed_values_rejected():
    model = tiny_model()
    x, mask = sample()
    x[0, -1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.loss(x, mask)
