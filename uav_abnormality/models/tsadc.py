# SPDX-License-Identifier: CC-BY-NC-SA-3.0
"""Source-level adaptation of TSAD-C (Ho and Armanfard, UAI 2025).

Derived from Armanfard-Lab/TSADC, commit
7013d7d2053044acdd3aa980bcf309d48f94de0d. See docs/tsadc.md and the
retained upstream license/provenance in third_party/TSADC.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._tsadc_s4 import Decontaminator, GraphLearner, S4Model


class DenseGINE(nn.Module):
    """Dense implementation of PyG GINEConv with scalar edge attributes.

    Adjacency indexes are [source, target], matching dense_to_sparse.
    A nonzero edge is present, and its value is a scalar attribute, NOT a
    multiplicative message weight. This distinction matters for GINE.
    """

    def __init__(self, hidden_dim, train_eps=False):
        super().__init__()
        self.edge_encoder = nn.Linear(1, hidden_dim)
        self.network = nn.Linear(hidden_dim, hidden_dim)
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("eps", torch.zeros(()))

    def forward(self, x, adjacency):
        # x: [..., sources, hidden]; adjacency: [..., sources, targets]
        edge_features = self.edge_encoder(adjacency.unsqueeze(-1))
        messages = F.relu(x.unsqueeze(-2) + edge_features)
        messages = messages * (adjacency != 0).unsqueeze(-1)
        aggregate = messages.sum(dim=-3)
        return self.network((1.0 + self.eps) * x + aggregate)


class TSADC(nn.Module):
    """S4 conditional diffusion + temporal S4 + learned spatial GINE.

    Inputs and observation masks: [batch, window, source, feature].
    score() returns nonnegative [batch, source] scores for the last timestamp.
    Completely unobserved sources return zero and MUST be excluded by the
    caller's observation mask. Labels are never an input to this model.
    """

    def __init__(
        self, n_sources, n_features, window_size, hidden_dim=16, state_dim=8,
        num_temporal_layers=1, num_gnn_layers=1, num_res_layers=2,
        res_channels=16, skip_channels=16, diffusion_steps=10,
        beta_start=0.0001, beta_end=0.02, diffusion_embedding_dim=16,
        mask_ratio=0.3, masking="rbm", graph_interval=4, graph_knn=3,
        residual_weight=0.5, edge_top_perc=0.5, dropout=0.0,
        feature_smoothing_weight=0.05, degree_weight=0.05,
        sparse_weight=0.05, lambda_1=0.01, lambda_2=1.2,
        score_samples=2, score_seed=1729, train_eps=False,
    ):
        super().__init__()
        if min(n_sources, n_features, window_size) < 1:
            raise ValueError("source, feature and window dimensions must be positive")
        if state_dim < 2 or state_dim % 2:
            raise ValueError("S4 state_dim must be an even integer >= 2")
        if diffusion_steps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("invalid diffusion steps or beta schedule")
        if not 0 < mask_ratio < 1 or masking not in ("rm", "rbm", "bom"):
            raise ValueError("mask_ratio must lie in (0,1); masking is rm/rbm/bom")
        if not 0 <= residual_weight < 1 or not 0 < edge_top_perc <= 1:
            raise ValueError("invalid graph residual weight or pruning fraction")
        if min(graph_interval, graph_knn, score_samples, num_gnn_layers,
               num_temporal_layers, num_res_layers) < 1:
            raise ValueError("layer, graph and sampling counts must be positive")
        if diffusion_embedding_dim < 4 or diffusion_embedding_dim % 2:
            raise ValueError("diffusion_embedding_dim must be even and >= 4")
        self.n_sources = int(n_sources)
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.mask_ratio = float(mask_ratio)
        self.masking = masking
        self.graph_interval = int(graph_interval)
        self.graph_knn = min(int(graph_knn), self.n_sources)
        self.residual_weight = float(residual_weight)
        self.edge_top_perc = float(edge_top_perc)
        self.feature_smoothing_weight = float(feature_smoothing_weight)
        self.degree_weight = float(degree_weight)
        self.sparse_weight = float(sparse_weight)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.score_samples = int(score_samples)
        self.score_seed = int(score_seed)
        self.diffusion_steps = int(diffusion_steps)

        beta = torch.linspace(beta_start, beta_end, diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim=0)
        posterior_beta = beta.clone()
        posterior_beta[1:] *= (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sigma", posterior_beta.sqrt())

        channels = self.n_sources * self.n_features
        self.decontaminator = Decontaminator(
            in_channels=channels, out_channels=channels,
            res_channels=res_channels, skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            diffusion_step_embed_dim_in=diffusion_embedding_dim,
            diffusion_step_embed_dim_mid=diffusion_embedding_dim,
            diffusion_step_embed_dim_out=diffusion_embedding_dim,
            s4_lmax=window_size, s4_d_state=state_dim,
            s4_dropout=dropout, s4_bidirectional=True, s4_layernorm=True,
        )
        self.temporal = S4Model(
            d_input=n_features, d_model=hidden_dim, d_state=state_dim,
            n_layers=num_temporal_layers, dropout=dropout,
            l_max=window_size, bidirectional=False,
            add_decoder=False, pool=False,
        )
        self.graph_learner = GraphLearner(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_nodes=n_sources, metric_type="self_attention",
        )
        self.gnn = nn.ModuleList([
            DenseGINE(hidden_dim, train_eps=train_eps)
            for _ in range(num_gnn_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, n_features)

    def _validate(self, x, mask):
        expected = (self.window_size, self.n_sources, self.n_features)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected or mask.shape != x.shape:
            raise ValueError("expected matching [batch, %s] input and mask" % (
                ", ".join(str(v) for v in expected),))
        mask = mask.bool()
        if not torch.isfinite(x[mask]).all():
            raise ValueError("observed TSAD-C values must be finite")
        return torch.where(mask, x, torch.zeros_like(x)), mask

    @staticmethod
    def _masked_mean(value, mask):
        weights = mask.to(value.dtype)
        return (torch.where(mask, value, torch.zeros_like(value))).sum() / weights.sum().clamp_min(1)

    def _to_channels(self, x):
        return x.reshape(x.shape[0], self.window_size, -1).transpose(1, 2)

    def _from_channels(self, x):
        return x.transpose(1, 2).reshape(-1, self.window_size, self.n_sources, self.n_features)

    def _conditioning_mask(self, observed, generator=None, last_hidden=False):
        """Generate upstream RandM/RandBM/BoM masks without using labels.

        A supplied CPU generator uses common random numbers across batch rows,
        so evaluation is independent of dataloader ordering and batch size.
        """
        b, channels, length = observed.shape
        count = max(1, min(length, round(length * self.mask_ratio)))
        shared = generator is not None
        rows = 1 if shared else b
        device = torch.device("cpu") if shared else observed.device
        hidden = torch.zeros(rows, channels, length, dtype=torch.bool, device=device)
        if self.masking == "rm":
            ranks = torch.rand(rows, channels, length, device=device, generator=generator)
            indices = ranks.topk(count, dim=-1).indices
            hidden.scatter_(-1, indices, True)
        else:
            n_segments = math.ceil(length / count)
            n_mask_channels = 1 if self.masking == "bom" else channels
            segments = torch.randint(n_segments, (rows, n_mask_channels, 1),
                                     device=device, generator=generator)
            times = torch.arange(length, device=device).view(1, 1, -1)
            hidden = ((times >= segments * count) & (times < (segments + 1) * count))
            hidden = hidden.expand(rows, channels, length).clone()
        if last_hidden:
            hidden[..., -1] = True
        hidden = hidden.to(observed.device).expand_as(observed)
        condition = observed & ~hidden
        # Sparse windows may have no observed targets in the sampled block.
        # Force one observed value to be hidden in each such window.
        no_target = ~(observed & ~condition).flatten(1).any(-1)
        if no_target.any():
            condition = condition.clone(memory_format=torch.contiguous_format)
            for row in torch.nonzero(no_target, as_tuple=False).flatten().tolist():
                available = torch.nonzero(observed[row].flatten(), as_tuple=False)
                if available.numel():
                    condition[row].view(-1)[available[-1, 0]] = False
        return condition

    def _adjacency(self, nodes, present):
        """Learn symmetric pruned graph over physical sources."""
        valid_edges = present.unsqueeze(-1) & present.unsqueeze(-2)
        attention = self.graph_learner(nodes, attn_mask=valid_edges)
        attention = 0.5 * (attention + attention.transpose(-1, -2))
        attention = attention * valid_edges
        normalized = F.normalize(nodes, dim=-1)
        cosine = (normalized @ normalized.transpose(-1, -2)).clamp_min(0)
        cosine = cosine * valid_edges
        knn_values, knn_indices = cosine.topk(self.graph_knn, dim=-1)
        knn = torch.zeros_like(cosine).scatter(-1, knn_indices, knn_values)
        knn = 0.5 * (knn + knn.transpose(-1, -2))
        eye = torch.eye(self.n_sources, device=nodes.device, dtype=torch.bool)
        knn = torch.where(eye, present.unsqueeze(-1).to(nodes.dtype), knn)
        adjacency = self.residual_weight * knn + (1 - self.residual_weight) * attention
        # Keep the strongest undirected edges, including tied edge weights.
        # The upstream strict > cutoff can disconnect every identical source.
        keep = max(1, math.ceil(self.n_sources ** 2 * self.edge_top_perc))
        threshold = adjacency.flatten(-2).topk(keep, dim=-1).values[..., -1]
        adjacency = adjacency * (adjacency >= threshold[..., None, None]) * valid_edges
        return adjacency

    def _graph_regularization(self, nodes, adjacency, present):
        # Retain the normalized-Laplacian implementation in upstream utils.
        degree = adjacency.sum(-1)
        inv = degree.clamp_min(1e-12).rsqrt() * present
        normalized = adjacency * inv.unsqueeze(-1) * inv.unsqueeze(-2)
        eye = torch.eye(self.n_sources, device=nodes.device, dtype=nodes.dtype)
        laplacian = eye * present.unsqueeze(-1) - normalized
        smooth = (nodes * (laplacian @ nodes)).sum((-1, -2))
        n_present = present.sum(-1).clamp_min(1).to(nodes.dtype)
        smooth = smooth / (n_present.square() * self.hidden_dim ** 2)
        connectivity = -(degree.clamp_min(1e-8).log() * present).sum(-1) / n_present
        sparse = adjacency.square().sum((-1, -2)) / n_present.square()
        valid = present.any(-1)
        total = (self.feature_smoothing_weight * smooth +
                 self.degree_weight * connectivity + self.sparse_weight * sparse)
        return self._masked_mean(total, valid)

    def _dependency_reconstruction(self, x, observed):
        b, length, sources, features = x.shape
        temporal_input = x.permute(0, 2, 1, 3).reshape(b * sources, length, features)
        hidden = self.temporal(temporal_input)
        hidden = hidden.reshape(b, sources, length, self.hidden_dim).transpose(1, 2)
        observed_sources = observed.any(-1)
        output = []
        regularization = []
        for start in range(0, length, self.graph_interval):
            stop = min(start + self.graph_interval, length)
            chunk = hidden[:, start:stop]
            active = observed_sources[:, start:stop]
            weights = active.to(hidden.dtype).unsqueeze(-1)
            pooled = (chunk * weights).sum(1) / weights.sum(1).clamp_min(1)
            present = active.any(1)
            adjacency = self._adjacency(pooled, present)
            regularization.append(self._graph_regularization(pooled, adjacency, present))
            # Upstream removes then adds self-loops with scalar edge value 1.
            eye = torch.eye(sources, device=x.device, dtype=torch.bool)
            edges = torch.where(eye, present.unsqueeze(-1).to(x.dtype), adjacency)
            edges = edges[:, None].expand(-1, stop - start, -1, -1)
            for layer in self.gnn:
                chunk = self.dropout(F.leaky_relu(layer(chunk, edges)))
            output.append(self.decoder(chunk))
        return torch.cat(output, dim=1), torch.stack(regularization).mean()

    def loss_components(self, x, mask):
        x, mask = self._validate(x, mask)
        signal, observed = self._to_channels(x), self._to_channels(mask)
        condition = self._conditioning_mask(observed)
        timestep = torch.randint(self.diffusion_steps, (x.shape[0],), device=x.device)
        cumulative = self.alpha_bar[timestep].view(-1, 1, 1)
        noise = torch.randn_like(signal)
        # Preserve the reference implementation's noising convention: observed
        # conditioning values replace noise at kept locations; the noise loss
        # is evaluated only at artificially hidden, observed locations.
        mixed_noise = torch.where(condition, signal, noise)
        noisy = cumulative.sqrt() * signal + (1 - cumulative).sqrt() * mixed_noise
        predicted_noise = self.decontaminator((noisy, signal, condition, timestep[:, None]))
        recovered = (noisy - (1 - cumulative).sqrt() * predicted_noise) / cumulative.sqrt()
        noise_loss = self._masked_mean((predicted_noise - noise).square(), observed & ~condition)
        recovered = self._from_channels(recovered)
        recovered = torch.where(mask, recovered, torch.zeros_like(recovered))
        reconstruction, graph_loss = self._dependency_reconstruction(recovered, mask)
        reconstruction_loss = self._masked_mean((reconstruction - recovered).square(), mask)
        return {"noise": noise_loss, "graph": graph_loss, "reconstruction": reconstruction_loss}

    def loss(self, x, mask):
        """Unsupervised training loss; callers may train on contaminated data."""
        parts = self.loss_components(x, mask)
        return parts["noise"] + parts["graph"] + parts["reconstruction"]

    @staticmethod
    def _fixed_noise(signal, generator):
        noise = torch.randn((1,) + tuple(signal.shape[1:]), generator=generator,
                            dtype=torch.float32, device="cpu")
        return noise.to(device=signal.device, dtype=signal.dtype).expand_as(signal)

    def _sample(self, signal, condition, generator):
        sample = self._fixed_noise(signal, generator)
        for step in range(self.diffusion_steps - 1, -1, -1):
            sample = torch.where(condition, signal, sample)
            timestep = torch.full((signal.shape[0], 1), step, device=signal.device, dtype=torch.long)
            predicted_noise = self.decontaminator((sample, signal, condition, timestep))
            sample = (sample - self.beta[step] / (1 - self.alpha_bar[step]).sqrt() * predicted_noise) / self.alpha[step].sqrt()
            if step > 0:
                sample = sample + self.sigma[step] * self._fixed_noise(signal, generator)
        return torch.where(condition, signal, sample)

    @torch.no_grad()
    def score(self, x, mask):
        """Eq. 11 weighted diffusion/dependency RMSE, localized to last time.

        Complete reverse diffusion is retained. The last timestamp is held out
        from diffusion conditioning so every observed source has a score.
        Fixed CPU generators leave the caller's RNG untouched.
        """
        was_training = self.training
        self.eval()
        try:
            x, mask = self._validate(x, mask)
            signal, observed = self._to_channels(x), self._to_channels(mask)
            last_mask = mask[:, -1]
            counts = last_mask.sum(-1).clamp_min(1)
            reconstruction, _ = self._dependency_reconstruction(x, mask)
            reconstruction_error = torch.where(last_mask, (reconstruction[:, -1] - x[:, -1]).square(), 0.0)
            second_score = (reconstruction_error.sum(-1) / counts).sqrt()
            first_score = torch.zeros_like(second_score)
            for sample_index in range(self.score_samples):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.score_seed + sample_index)
                condition = self._conditioning_mask(observed, generator, last_hidden=True)
                recovered = self._from_channels(self._sample(signal, condition, generator))
                error = torch.where(last_mask, (recovered[:, -1] - x[:, -1]).square(), 0.0)
                first_score += (error.sum(-1) / counts).sqrt() / self.score_samples
            return self.lambda_1 * first_score + self.lambda_2 * second_score
        finally:
            self.train(was_training)
