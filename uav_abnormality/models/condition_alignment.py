"""Leave-one-source-out condition alignment and report quality.

Inputs use the fixed feature and condition scales fitted on training reports.
Scalar reports use exact medians and mean absolute distance over common time
positions. Vector reports use geometric medians and mean Euclidean distance
(normalized by sqrt(feature count)); a vector is observed when all coordinates
are present. Reference selection depends only on conditions and availability.
"""
import math

import torch
from torch import nn


def weighted_median(values, weights, dim=-1):
    """Weighted absolute-distance minimizer; midpoint for a tied median interval.

    ``weights`` broadcasts to ``values``. Zero-weight/nonfinite observations are
    omitted. An empty set returns NaN. Sorting and the selected values retain
    gradients, which permits the same robust reference during training.
    """
    values, weights = torch.broadcast_tensors(values, weights)
    if values.shape[dim] == 0:
        raise ValueError("weighted_median needs a nonempty observation axis")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    valid = torch.isfinite(values) & (weights > 0)
    clean = torch.where(valid, values, torch.zeros_like(values))
    weight = torch.where(valid, weights, torch.zeros_like(weights))
    order = torch.argsort(torch.where(valid, clean, float("inf")), dim=dim,
                          stable=True)
    ordered = clean.gather(dim, order)
    ordered_weights = weight.gather(dim, order)
    cumulative = ordered_weights.cumsum(dim)
    total = ordered_weights.sum(dim, keepdim=True)
    half = total / 2
    lower = (cumulative < half).sum(dim, keepdim=True)
    upper = (cumulative <= half).sum(dim, keepdim=True)
    last = values.shape[dim] - 1
    lower = lower.clamp(max=last)
    upper = upper.clamp(max=last)
    median = (ordered.gather(dim, lower) + ordered.gather(dim, upper)) / 2
    median = torch.where(total > 0, median, float("nan"))
    return median.squeeze(dim)


def weighted_geometric_median(points, weights, dim=-2, max_iter=128,
                              tolerance=1e-6):
    """Weighted Euclidean-distance minimizer by stabilized Weiszfeld iterations.

    The final dimension contains coordinates; ``dim`` is the observation axis.
    ``weights`` has the shape of ``points`` without the coordinate dimension.
    Scalar inputs take the exact weighted-median path. All-empty inputs return
    NaN. The finite iterations are differentiable with respect to observations.
    """
    if points.ndim < 2 or points.shape[-1] < 1:
        raise ValueError("Expected points [..., observation, feature]")
    dim = dim % points.ndim
    if dim == points.ndim - 1:
        raise ValueError("Observation and feature axes must differ")
    weight_shape = points.shape[:-1]
    weights = torch.broadcast_to(weights, weight_shape)
    points = points.movedim(dim, -2)
    weights = weights.movedim(dim, -1)
    if points.shape[-1] == 1:
        return weighted_median(points[..., 0], weights).unsqueeze(-1)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("Iteration count and tolerance must be positive")
    valid = torch.isfinite(points).all(-1) & (weights > 0)
    clean = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
    weight = torch.where(valid, weights, torch.zeros_like(weights))
    total = weight.sum(-1, keepdim=True)
    center = (clean * weight.unsqueeze(-1)).sum(-2) / total.clamp_min(1e-12)
    for _ in range(max_iter):
        distance = torch.linalg.vector_norm(clean - center.unsqueeze(-2), dim=-1)
        effective_weight = weight / distance.clamp_min(tolerance)
        proposal = (clean * effective_weight.unsqueeze(-1)).sum(-2)
        proposal = proposal / effective_weight.sum(-1, keepdim=True).clamp_min(1e-12)
        change = (proposal - center).abs().amax()
        center = proposal
        if change.detach() <= tolerance:
            break
    return torch.where(total > 0, center, float("nan"))


class ConditionAlignment(nn.Module):
    """Additive condition model from proposal sections 2.1, 2.2 and 3.1.

    ``x, mask`` have shape [B,T,S,F], ``conditions`` [B,T,S,C], and
    ``condition_mask`` [B,T,S]. Report scores correspond to the final position.
    ``reference_count`` counts eligible endpoint peers after radius/top-k rules.
    Reputations and labels are deliberately absent from the scoring interface.
    """

    def __init__(self, n_features, n_conditions, hidden_dim=32,
                 architecture="mlp", min_references=2, max_neighbors=None,
                 neighbor_radius=None, huber_delta=1.0, tau0=0.1):
        super().__init__()
        if min(n_features, n_conditions, hidden_dim, min_references) < 1:
            raise ValueError("Dimensions and min_references must be positive")
        if architecture not in {"linear", "mlp"}:
            raise ValueError("architecture must be linear or mlp")
        if max_neighbors is not None and max_neighbors < min_references:
            raise ValueError("max_neighbors must be at least min_references")
        if neighbor_radius is not None and (not math.isfinite(neighbor_radius)
                                             or neighbor_radius <= 0):
            raise ValueError("neighbor_radius must be finite and positive")
        if not all(math.isfinite(v) and v > 0 for v in (huber_delta, tau0)):
            raise ValueError("huber_delta and tau0 must be finite and positive")
        self.n_features, self.n_conditions = int(n_features), int(n_conditions)
        self.min_references = int(min_references)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.neighbor_radius = neighbor_radius
        self.huber_delta, self.tau0 = float(huber_delta), float(tau0)
        self.architecture = architecture
        # An output bias cancels from h(c')-h(c), so it is omitted.
        if architecture == "linear":
            self.condition_model = nn.Linear(n_conditions, n_features, bias=False)
            nn.init.zeros_(self.condition_model.weight)
        else:
            self.condition_model = nn.Sequential(
                nn.Linear(n_conditions, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, n_features, bias=False))
            nn.init.zeros_(self.condition_model[-1].weight)

    def correction(self, conditions):
        """Return h_theta(c), in fixed standardized feature units."""
        if conditions.shape[-1] != self.n_conditions:
            raise ValueError("Incorrect condition dimension")
        return self.condition_model(conditions)

    def align(self, x, source_conditions, target_conditions):
        """Phi(x,c,c') = x+h(c')-h(c), with broadcasting."""
        return x + (self.correction(target_conditions)
                    - self.correction(source_conditions))

    def _validate(self, x, mask, conditions, condition_mask):
        if x.ndim != 4 or mask.shape != x.shape or x.shape[-1] != self.n_features:
            raise ValueError("Expected x and mask [B,T,S,n_features]")
        if min(x.shape[:3]) < 1:
            raise ValueError("Batch, time and source dimensions must be positive")
        if conditions.shape != x.shape[:-1] + (self.n_conditions,):
            raise ValueError("Expected conditions [B,T,S,n_conditions]")
        if condition_mask.shape != x.shape[:-1]:
            raise ValueError("Expected condition_mask [B,T,S]")
        mask, condition_mask = mask.bool(), condition_mask.bool()
        if not torch.isfinite(x[mask]).all():
            raise ValueError("Observed feature values must be finite")
        if not torch.isfinite(conditions[condition_mask]).all():
            raise ValueError("Observed conditions must be finite")
        x = torch.where(mask, x, torch.zeros_like(x))
        conditions = torch.where(condition_mask.unsqueeze(-1), conditions,
                                 torch.zeros_like(conditions))
        observed = mask.all(-1) & condition_mask
        return x, conditions, observed, condition_mask

    def _reference(self, x, mask, conditions, condition_mask):
        x, conditions, observed, condition_mask = self._validate(
            x, mask, conditions, condition_mask)
        sources = x.shape[2]
        correction = self.correction(conditions)
        # Reference target axis I and donor axis J are separate. A target value
        # never enters its own center, dispersion, or neighborhood selection.
        distance = torch.linalg.vector_norm(
            conditions[:, -1, :, None] - conditions[:, -1, None, :], dim=-1)
        selected = (observed[:, -1, :, None] & observed[:, -1, None, :]
                    & ~torch.eye(sources, device=x.device, dtype=torch.bool)[None])
        if self.neighbor_radius is not None:
            selected = selected & (distance <= self.neighbor_radius)
        if self.max_neighbors is not None and self.max_neighbors < sources:
            ordered = torch.argsort(torch.where(selected, distance, float("inf")),
                                    dim=-1, stable=True)
            rank = torch.empty_like(ordered)
            rank.scatter_(-1, ordered, torch.arange(sources, device=x.device)
                          .expand_as(ordered))
            selected = selected & (rank < self.max_neighbors)
        aligned = (x[:, :, None, :, :] + correction[:, :, :, None, :]
                   - correction[:, :, None, :, :])
        peer_valid = (selected[:, None, :, :] & observed[:, :, None, :]
                      & condition_mask[:, :, :, None])
        if self.n_features == 1:
            reference = weighted_median(aligned[..., 0], peer_valid.to(x.dtype))
            reference = reference.unsqueeze(-1)
        else:
            reference = weighted_geometric_median(aligned, peer_valid.to(x.dtype))
        reference_valid = peer_valid.sum(-1) >= self.min_references
        # Empty medians remain externally visible as NaN but never participate
        # in arithmetic that could generate NaN gradients for a valid report.
        reference = torch.nan_to_num(reference)
        common = observed & reference_valid
        evaluated = common[:, -1]
        return x, aligned, reference, peer_valid, reference_valid, common, evaluated

    def _distance(self, residual):
        if self.n_features == 1:
            return residual[..., 0].abs()
        return torch.linalg.vector_norm(residual, dim=-1) / math.sqrt(self.n_features)

    def diagnostics(self, x, mask, conditions, condition_mask):
        """Return report error e, quality q, reference MAD sigma and score s."""
        (x, aligned, reference, peer_valid, reference_valid, common,
         evaluated) = self._reference(x, mask, conditions, condition_mask)
        residual = self._distance(x - reference)
        error = (residual * common).sum(1) / common.sum(1).clamp_min(1)
        peer_common = peer_valid & common.unsqueeze(-1)
        peer_distance = self._distance(aligned - reference.unsqueeze(-2))
        peer_error = (peer_distance * peer_common).sum(1)
        peer_error = peer_error / peer_common.sum(1).clamp_min(1)
        dispersion = weighted_median(peer_error,
                                     peer_common.any(1).to(x.dtype))
        dispersion = torch.nan_to_num(dispersion)
        quality = 1 / (1 + error)
        score = error / dispersion.clamp_min(self.tau0)
        result = {"error": error, "quality": quality,
                  "dispersion": dispersion, "score": score}
        result = {key: torch.where(evaluated, value, float("nan"))
                  for key, value in result.items()}
        result.update({"evaluated": evaluated,
                       "reference_count": peer_valid[:, -1].sum(-1),
                       "reference": torch.where(reference_valid.unsqueeze(-1),
                                                reference, float("nan")),
                       "reference_valid": reference_valid})
        return result

    def loss(self, x, mask, conditions, condition_mask):
        """Mean Huber loss of the fixed-distance, leave-one-source-out errors."""
        (x, _, reference, _, _, common,
         evaluated) = self._reference(x, mask, conditions, condition_mask)
        distance = self._distance(x - reference)
        error = (distance * common).sum(1) / common.sum(1).clamp_min(1)
        delta = self.huber_delta
        huber = torch.where(error <= delta, 0.5 * error.square(),
                            delta * (error - 0.5 * delta))
        return (huber * evaluated).sum() / evaluated.sum().clamp_min(1)

    def score(self, x, mask, conditions, condition_mask):
        return self.diagnostics(x, mask, conditions, condition_mask)["score"]
