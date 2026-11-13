"""Reject unusable report protocols using masks and identities before training.

No anomaly scores, reference temperatures or held-out detection performance are
used to plan the calendar. Calibration counts respect identity sampling and the
configured temporal separation, including unavailable selected reports.
"""
import numpy as np

from .calibration import fit_label_trim
from .condition_data import ConditionScaler
from .protocol import SPLITS, Standardizer, Windows


def _eligible_registry(frame, sources, scaler, condition_scaler, config):
    window = dict(config.get('window', {}))
    window.setdefault('window_size', 16)
    window.setdefault('min_sources', 3)
    dataset = Windows(frame, sources, scaler, allow_empty=True, **window)
    registry = frame.reset_index(drop=True).copy()
    registry['score'] = np.nan
    registry['baseline_eligible'] = False
    model = config.get('models', {}).get('condition_alignment', {})
    if config.get('algorithm') == 'condition_alignment':
        model = config.get('model', model)
    minimum = int(model.get('min_references', 2))
    radius, neighbors = model.get('neighbor_radius'), model.get('max_neighbors')
    conditions = condition_scaler.transform(registry[condition_scaler.columns].to_numpy(float))
    for gid, end in dataset.index:
        group = dataset.groups[gid]
        row_ids = group['rows'][end]
        baseline = (row_ids >= 0) & group['mask'][end].any(-1)
        registry.loc[row_ids[baseline], 'baseline_eligible'] = True
        present = (row_ids >= 0) & group['mask'][end].all(-1)
        ids = row_ids[present]
        ids = ids[np.isfinite(conditions[ids]).all(-1)]
        if not len(ids):
            continue
        peers = np.ones((len(ids), len(ids)), dtype=bool)
        np.fill_diagonal(peers, False)
        if radius is not None:
            distance = np.linalg.norm(conditions[ids, None] - conditions[ids][None], axis=-1)
            peers &= distance <= float(radius)
        counts = peers.sum(-1)
        if neighbors is not None:
            counts = np.minimum(counts, int(neighbors))
        # A constant placeholder denotes structural availability, not a score.
        registry.loc[ids[counts >= minimum], 'score'] = 1.
    return registry, len(dataset)


def validate_experiment_feasibility(splits, config):
    """Return compact counts, or fail before fitting an unusable benchmark."""
    from .condition_experiment import plan_calibration_registry

    policy = config.get('protocol_feasibility', {})
    minimum_positive = (0 if config.get('attack_mode') == 'none' else
                        int(policy.get('min_scored_attacks_per_split', 1)))
    minimum_pool = int(policy.get('min_retained_calibration_units', 19))
    sources = sorted(splits['train'].source_id.unique().tolist())
    scaler = Standardizer().fit(splits['train'])
    condition_scaler = ConditionScaler().fit(splits['train'], config['method']['condition_columns'])
    result, registries = {}, {}
    for name in SPLITS:
        registry, windows = _eligible_registry(splits[name], sources, scaler, condition_scaler, config)
        registries[name] = registry
        available = registry.score.notna() & registry.baseline_eligible
        positive = registry.injection_label.eq(1) if 'injection_label' in registry else np.zeros(len(registry), bool)
        result[name] = {
            'reports': len(registry), 'time_slots': int(registry.timestamp.nunique()),
            'eligible_windows': windows, 'common_scoreable_reports': int(available.sum()),
            'injected_reports': int(positive.sum()),
            'common_scoreable_attacks': int((available & positive).sum()),
        }
        if not windows or not result[name]['common_scoreable_reports']:
            raise ValueError('Protocol preflight: %s has no common scoreable reports. '
                             'Adjust calendar, warmup, conditions and real source participation before training.' % name)
        if result[name]['common_scoreable_attacks'] < minimum_positive:
            raise ValueError('Protocol preflight: %s has %d scoreable attacked reports; requires %d. '
                             'Adjust calendar, warmup and real source participation before training.' %
                             (name, result[name]['common_scoreable_attacks'], minimum_positive))
    registry = plan_calibration_registry(registries['calibration'], config['window'])
    calibration_config = dict(config['method'].get('calibration', {}))
    calibration_config.setdefault('alpha', config.get('calibration', {}).get('alpha', .05))
    calibration_config.update(verification='none', budget=0)
    calibration, _ = fit_label_trim(registry[registry.calibration_planned], calibration_config)
    n = calibration['n']
    original = config['method'].get('calibration', {})
    budget = original.get('budget', 'auto')
    budget = calibration['budget_limit'] if budget in (None, 'auto') else int(budget)
    if original.get('verification', original.get('verification_mode', 'none')) == 'none':
        budget = 0
    guaranteed = n - budget
    result['calibration'].update(candidate_calibration_units=calibration['candidate_tasks'],
        available_calibration_units=n, worst_case_retained_units=guaranteed,
        worst_case_min_p=1. / (guaranteed + 1))
    if guaranteed < minimum_pool or 1. / (guaranteed + 1) > calibration['alpha']:
        raise ValueError('Protocol preflight: calibration retains at least %d units after the review '
                         'budget; requires %d and attainable alpha=%g.' %
                         (guaranteed, minimum_pool, calibration['alpha']))
    return result
