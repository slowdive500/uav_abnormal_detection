"""PRBTD: CFSTTN prediction, signed implications, and reputation truth discovery.

This is a documented multivariate UAV-report adaptation; the original TaxiBJ
paper's complete predictor and experimental protocol are not reproduced.
The predictor implementation originates in the pinned upstream repository.
"""
from collections import deque

import torch
from torch import nn

from .prbtd_cfsttn import Transformer


class PRBTD(nn.Module):
    """Input [batch, window, physical_source, feature]; output [batch, source].

    ``stream_score`` is the benchmark interface: call in temporal order and reset
    at independent sequence/task boundaries. ``score`` is side-effect-free and
    replays each window with its own reputation state, for offline diagnostics.
    Neither method consumes fault labels or clean reference measurements.
    """

    supports_streaming = True

    def __init__(self, n_sources, n_features, window_size, d_model=16,
                 d_inner=32, layers=1, nheads_spatial=2, nheads_temporal=2,
                 dk_s=8, dk_t=8, growth_rate=4, drop_rate=0.0,
                 alpha=0.018, beta=0.06, gamma=0.5, rho=0.002,
                 implication_history=5, environment_history=100,
                 max_iter=30, tolerance=1e-4, residual_mode="standardized",
                 residual_floor=1e-3):
        super().__init__()
        if min(n_sources, n_features) < 1 or window_size < 2:
            raise ValueError("PRBTD needs positive dimensions and window_size >= 2")
        if not (0 < alpha <= 1 and 0 < beta <= 1 and 0 < gamma < 1):
            raise ValueError("alpha/beta must be in (0,1], gamma in (0,1)")
        if rho < 0 or min(implication_history, environment_history, max_iter) < 1:
            raise ValueError("rho must be nonnegative and histories/iterations positive")
        if residual_mode not in {"standardized", "relative"}:
            raise ValueError("residual_mode must be standardized or relative")
        if residual_floor <= 0:
            raise ValueError("residual_floor must be positive")
        self.n_sources, self.n_features = int(n_sources), int(n_features)
        self.window_size = int(window_size)
        self.alpha, self.beta, self.gamma, self.rho = alpha, beta, gamma, rho
        self.implication_history = int(implication_history)
        self.environment_history = int(environment_history)
        self.max_iter, self.tolerance = int(max_iter), tolerance
        self.residual_mode, self.residual_floor = residual_mode, residual_floor
        # Each standardized physical feature is processed by the same scalar
        # CFSTTN. Source order is a fixed index lattice, not a geographic grid.
        self.predictor = Transformer(
            (1, window_size - 1, 1, 1, n_sources), [], [],
            growth_rate=growth_rate, num_init_features=growth_rate,
            d_model=d_model, d_inner=d_inner, layers=layers,
            nheads_spatial=nheads_spatial, nheads_temporal=nheads_temporal,
            dk_s=dk_s, dk_t=dk_t, drop_rate=drop_rate, bounded_output=False)
        self._stream_state = None
        self.stream_diagnostics = None

    def _validate(self, x, mask):
        expected = (self.window_size, self.n_sources, self.n_features)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected or mask.shape != x.shape:
            raise ValueError("Expected x and mask [B, %d, %d, %d]" % expected)
        if not torch.isfinite(x[mask.bool()]).all():
            raise ValueError("Observed values must be finite")
        return torch.where(mask.bool(), x, torch.zeros_like(x)), mask.bool()

    def _predict_history(self, history):
        b, t, s, f = history.shape
        # The historical tensor never includes the value being predicted.
        inp = history.permute(0, 3, 1, 2).reshape(b * f, t, 1, 1, s)
        prediction, _ = self.predictor(inp)
        return prediction[:, 0, 0, :].reshape(b, f, s).transpose(1, 2)

    def forward(self, x, mask):
        x, _ = self._validate(x, mask)
        return self._predict_history(x[:, :-1])

    def loss(self, x, mask):
        """Upstream L1 next-step prediction objective over observed targets."""
        x, mask = self._validate(x, mask)
        prediction = self._predict_history(x[:, :-1])
        valid = mask[:, -1].to(x.dtype)
        return ((prediction - x[:, -1]).abs() * valid).sum() / valid.sum().clamp_min(1)

    def reset_state(self):
        """Clear online reputations and both report buffers before a new task."""
        self._stream_state = None
        self.stream_diagnostics = None

    def _new_state(self, x, batch_size):
        return {
            "reputation": x.new_full((batch_size, self.n_sources), 0.5),
            "quality": x.new_full((batch_size, self.n_sources), 0.5),
            "implications": deque(maxlen=self.implication_history),
            "environment": deque(maxlen=self.environment_history),
        }

    def _residual(self, value, prediction):
        residual = value - prediction
        if self.residual_mode == "relative":
            sign = torch.where(prediction < 0, -1.0, 1.0)
            denominator = sign * prediction.abs().clamp_min(self.residual_floor)
            residual = residual / denominator
        return residual

    def _implication_matrix(self, error, mask, environmental_error, history):
        b, s, f = error.shape
        result = error.new_zeros((b, s, s))
        env = environmental_error[:, None, None, :]
        for past_error, past_mask in history:
            shared = mask[:, :, None, :] & past_mask[:, None, :, :]
            current_user = error[:, :, None, :] - env
            past_user = past_error[:, None, :, :] - env
            current_norm = ((current_user.square() + env.square()) * shared).sum(-1)
            past_norm = ((past_user.square() + env.square()) * shared).sum(-1)
            dot = ((current_user * past_user + env.square()) * shared).sum(-1)
            cosine = dot / (current_norm * past_norm).sqrt().clamp_min(1e-12)
            cosine = torch.where(cosine.abs() < 1e-4, 0.0, cosine).clamp(-1, 1)
            result += cosine
        # A source cannot corroborate itself, including its past reports.
        result.diagonal(dim1=1, dim2=2).zero_()
        return result

    def _update_reputation(self, reputation, quality):
        positive = quality >= self.gamma
        rate = torch.where(positive, self.alpha, self.beta)
        scale = torch.where(positive, 1.0 / (1 - self.gamma), 1.0 / self.gamma)
        return 0.5 + (rate * (quality - self.gamma) * scale
                      + (1 - rate) * (2 * reputation - 1)) * 0.5

    def _step(self, value, prediction, mask, state):
        error = torch.where(mask, self._residual(value, prediction), 0.0)
        state["environment"].append((error, mask))
        state["implications"].append((error, mask))
        errors = torch.stack([item[0] for item in state["environment"]], dim=1)
        masks = torch.stack([item[1] for item in state["environment"]], dim=1)
        env = errors.sum(dim=(1, 2)) / masks.sum(dim=(1, 2)).clamp_min(1)
        # Preserve the upstream signed epsilon for zero environmental residuals.
        env = torch.where(env >= 0, env.clamp_min(0.001), env.clamp_max(-0.001))
        implication = self._implication_matrix(error, mask, env, state["implications"])
        active = mask.any(-1)
        reputation, quality = state["reputation"], state["quality"]
        quality = torch.where(active, reputation, quality)
        done = ~active.any(-1)
        eps = torch.finfo(value.dtype).eps
        for iteration in range(self.max_iter):
            updating = active & ~done[:, None]
            previous = reputation
            reputation = torch.where(updating,
                                     self._update_reputation(reputation, quality),
                                     reputation).clamp(eps, 1 - eps)
            # Upstream uses sequential (Gauss-Seidel) quality updates.
            for source in range(self.n_sources):
                theta = -torch.log1p(-reputation[:, source])
                support = -torch.log1p(-quality.clamp(eps, 1 - eps))
                theta += self.rho * (support * implication[:, source, :]).sum(-1)
                updated = -torch.expm1(-theta.clamp(min=0, max=30))
                quality[:, source] = torch.where(updating[:, source],
                                                updated.clamp(0, 1 - eps),
                                                quality[:, source])
            change = (reputation - previous).abs().amax(-1)
            if iteration >= 1:
                done = done | (change <= self.tolerance)
            if done.all():
                break
        state["reputation"], state["quality"] = reputation, quality
        # Missing reports are excluded by the evaluator; zero here is neutral.
        return torch.where(active, 1 - quality, torch.zeros_like(quality))

    @torch.no_grad()
    def stream_score(self, x, mask):
        """Score consecutive windows while retaining original online state.

        Rows must be temporal successors within ONE sequence/task. Batching
        changes no update ordering. Call eval() before scoring.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before PRBTD scoring")
        x, mask = self._validate(x, mask)
        predictions = self._predict_history(x[:, :-1])
        if self._stream_state is None:
            self._stream_state = self._new_state(x, 1)
        if self._stream_state["reputation"].device != x.device:
            raise RuntimeError("Reset PRBTD state after changing model device")
        scores, prior, reputations = [], [], []
        for row in range(x.shape[0]):
            prior.append(self._stream_state["reputation"].clone())
            scores.append(self._step(x[row:row + 1, -1], predictions[row:row + 1],
                                     mask[row:row + 1, -1], self._stream_state))
            reputations.append(self._stream_state["reputation"].clone())
        result = torch.cat(scores, dim=0)
        self.stream_diagnostics = {"quality_native": 1 - result,
                                   "reputation_prior": torch.cat(prior, dim=0),
                                   "reputation": torch.cat(reputations, dim=0)}
        return result

    @torch.no_grad()
    def score(self, x, mask):
        """Pure window-local scoring; this does not alter streaming state.

        For each prefix, predict using only preceding observations, padded with
        standardized zeros on the left. The first observation has no preceding
        history and is therefore not used as a scored reference.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before PRBTD scoring")
        x, mask = self._validate(x, mask)
        histories = []
        for time in range(1, self.window_size):
            prefix = torch.zeros_like(x[:, :-1])
            prefix[:, -time:] = x[:, :time]
            histories.append(prefix)
        predictions = self._predict_history(torch.cat(histories, dim=0))
        predictions = predictions.reshape(self.window_size - 1, x.shape[0],
                                          self.n_sources, self.n_features)
        state = self._new_state(x, x.shape[0])
        for time in range(1, self.window_size):
            scores = self._step(x[:, time], predictions[time - 1], mask[:, time], state)
        return scores

    @property
    def reputation(self):
        """Latest stream reputation for reporting, or None before stream start."""
        if self._stream_state is None:
            return None
        return self._stream_state["reputation"][0].detach().clone()
