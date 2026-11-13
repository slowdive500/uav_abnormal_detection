"""Condition-based report scoring, environment estimates and quality reputation."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data.schema import KEYS, feature_columns
from .condition_data import ConditionScaler

ROUND = ['sequence_id', 'task_id', 'timestamp']


@torch.no_grad()
def score_condition_windows(model, dataset, device, batch_size=64):
    model.eval()
    rows = []
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in indices]
        batch = {key: torch.stack([item[key] for item in items]).to(device)
                 for key in ('x', 'mask', 'conditions', 'condition_mask')}
        result = model.diagnostics(**batch)
        references = result['reference'][:, -1].detach().cpu().numpy()
        diagnostics = {key: value.detach().cpu().numpy() for key, value in result.items()
                       if key in ('error', 'quality', 'dispersion', 'score', 'evaluated', 'reference_count')}
        for b, index in enumerate(indices):
            _, _, row_ids = dataset.metadata(index)
            for source, row_id in enumerate(row_ids):
                if row_id < 0 or not diagnostics['evaluated'][b, source]:
                    continue
                row = dataset.frame.iloc[row_id].to_dict()
                for key, values in diagnostics.items():
                    row[key] = bool(values[b, source]) if key == 'evaluated' else float(values[b, source])
                for feature in range(len(dataset.scaler.columns)):
                    row['aligned_reference_' + str(feature)] = float(
                        references[b, source, feature] * dataset.scaler.scale[feature] + dataset.scaler.mean[feature])
                if not all(np.isfinite(row[k]) for k in ('error', 'quality', 'dispersion', 'score')):
                    raise FloatingPointError('Nonfinite condition-alignment diagnostics')
                rows.append(row)
    return pd.DataFrame(rows, columns=list(dataset.frame.columns) +
                        ['error', 'quality', 'dispersion', 'score', 'evaluated', 'reference_count'] +
                        ['aligned_reference_' + str(i) for i in range(len(dataset.scaler.columns))])


def reputation_update(prior, quality, eta_up=.05, eta_down=.25):
    if not 0 < eta_up < eta_down <= 1:
        raise ValueError('Require 0 < eta_up < eta_down <= 1')
    return prior + (eta_up if quality >= prior else eta_down) * (quality - prior)


def _query_table(raw, condition_columns, query_path):
    if query_path:
        queries = pd.read_csv(query_path, dtype={'sequence_id': str, 'task_id': str, 'query_id': str})
        missing = set(ROUND + condition_columns) - set(queries.columns)
        if missing:
            raise ValueError('Query table missing: ' + str(sorted(missing)))
        if not pd.api.types.is_numeric_dtype(queries.timestamp):
            queries['timestamp'] = pd.to_datetime(queries.timestamp, utc=True).astype('int64') / 1e9
        if 'query_id' not in queries:
            queries['query_id'] = queries.index.astype(str)
        if 'raw_node_id' in raw and 'raw_node_id' in queries:
            if set(raw.raw_node_id) & set(queries.raw_node_id):
                raise ValueError('Query measurements must use held-out sensors')
        return queries
    rows = []
    for key, group in raw.groupby(ROUND, sort=True):
        row = dict(zip(ROUND, key), query_id='task_center')
        for column in condition_columns:
            finite = group[column].to_numpy(float)
            finite = finite[np.isfinite(finite)]
            row[column] = float(np.median(finite)) if len(finite) else np.nan
        # References are attached for evaluation only; no estimator reads them.
        for column in feature_columns(raw):
            reference = 'truth_' + column[len('value_'):]
            if reference in group:
                values = group[reference].dropna().to_numpy(float)
                if len(values) and np.allclose(values, values[0]):
                    row['reference_' + column[len('value_'):]] = float(values[0])
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def estimate_query(model, group, query, scaler, conditions, device, config):
    """Align scored endpoint reports to a query; references never enter this API."""
    from .models.condition_alignment import weighted_median, weighted_geometric_median
    minimum = int(config.get('min_sources', 3))
    neighbors = config.get('max_neighbors', 8)
    radius = config.get('radius')
    if minimum < 1 or (neighbors is not None and int(neighbors) < minimum):
        raise ValueError('Environment requires max_neighbors >= min_sources >= 1')
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError('Environment radius must be positive and finite')
    n_features = len(scaler.columns)
    unavailable = np.full(n_features, np.nan)
    target = conditions.transform(np.asarray(query, float))
    if not len(group) or not np.isfinite(target).all():
        return unavailable, 0, 0.
    origin = conditions.transform(group[conditions.columns].to_numpy(float))
    values = scaler.transform(group[scaler.columns].to_numpy(float))
    distance = np.linalg.norm(origin - target, axis=-1)
    valid = np.isfinite(origin).all(-1) & np.isfinite(values).all(-1)
    if radius is not None:
        valid &= distance <= float(radius)
    selected = np.flatnonzero(valid)
    # Ties use stable source ID order, independent of measured values or quality.
    selected = sorted(selected, key=lambda i: (distance[i], str(group.iloc[i].source_id)))
    if neighbors is not None:
        selected = selected[:int(neighbors)]
    retained = group.iloc[selected]
    weights = retained.reputation_prior.to_numpy(float) * retained.quality.to_numpy(float)
    weights *= (retained.prediction.to_numpy(int) == 0)
    positive = weights > 0
    count, mass = int(positive.sum()), float(weights.sum())
    if count < minimum or mass <= 0:
        return unavailable, count, mass
    selected = np.asarray(selected, dtype=int)[positive]
    x = torch.as_tensor(values[selected], dtype=torch.float32, device=device)
    source_c = torch.as_tensor(origin[selected], dtype=torch.float32, device=device)
    target_c = torch.as_tensor(target, dtype=torch.float32, device=device).expand_as(source_c)
    aligned = model.align(x, source_c, target_c)
    weight = torch.as_tensor(weights[positive], dtype=torch.float32, device=device)
    if n_features == 1:
        estimate = weighted_median(aligned[:, 0], weight).reshape(1)
    else:
        estimate = weighted_geometric_median(aligned, weight)
    result = estimate.detach().cpu().numpy() * scaler.scale + scaler.mean
    return result, count, mass


def apply_reputation_and_estimate(model, predictions, raw, scaler, condition_scaler, device,
                                   method_config, query_path=None):
    settings = method_config.get('reputation', {})
    initial = float(settings.get('initial', .5))
    eta_up, eta_down = float(settings.get('eta_up', .05)), float(settings.get('eta_down', .25))
    if not 0 <= initial <= 1:
        raise ValueError('Initial reputation must be in [0,1]')
    reputation_update(initial, initial, eta_up, eta_down)
    predictions = predictions.copy()
    for column in ('reputation_prior', 'reputation'):
        predictions[column] = np.nan
    groups = {key: group.index for key, group in predictions.groupby(ROUND, sort=False)}
    queries = _query_table(raw, condition_scaler.columns, query_path)
    query_groups = {key: group for key, group in queries.groupby(ROUND, sort=False)}
    state, last_sequence, estimates = {}, None, []
    keys = sorted(set(map(tuple, raw[ROUND].to_numpy())), key=lambda k: (k[0], k[2], k[1]))
    for key in keys:
        if key[0] != last_sequence:
            state, last_sequence = {}, key[0]
        indices = groups.get(key, [])
        for index in indices:
            source = str(predictions.at[index, 'source_id'])
            predictions.at[index, 'reputation_prior'] = state.get(source, initial)
        group = predictions.loc[indices]
        for _, query in query_groups.get(key, pd.DataFrame()).iterrows():
            values, count, mass = estimate_query(model, group,
                query[condition_scaler.columns].to_numpy(float), scaler, condition_scaler, device,
                method_config.get('environment', {}))
            row = query.to_dict()
            row.update(contributing_sources=count, effective_weight=mass,
                       status='estimated' if np.isfinite(values).all() else 'insufficient_contributions')
            row.update({'estimate_' + str(i): float(value) for i, value in enumerate(values)})
            estimates.append(row)
        # Each task uses the old reputation for estimation, then updates once.
        for index in indices:
            source = str(predictions.at[index, 'source_id'])
            prior, quality = predictions.at[index, 'reputation_prior'], predictions.at[index, 'quality']
            updated = reputation_update(prior, quality, eta_up, eta_down)
            state[source] = updated
            predictions.at[index, 'reputation'] = updated
    return predictions, pd.DataFrame(estimates)


def environment_metrics(estimates, n_features):
    from .evaluation import _errors
    result = {'method': 'condition_aligned_quality_reputation_weighted_median',
              'query_rounds': len(estimates), 'estimated_rounds': 0, 'coverage': 0.}
    if not len(estimates):
        return result
    count = int((estimates.status == 'estimated').sum())
    result.update(estimated_rounds=count, coverage=count / len(estimates))
    for i in range(n_features):
        reference = 'reference_' + str(i)
        if reference in estimates:
            result[str(i)] = _errors(estimates['estimate_' + str(i)], estimates[reference])
    return result
