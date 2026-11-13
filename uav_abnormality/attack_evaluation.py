"""Offline, counterfactual audits of attempted report attacks.

Attack identities and feasibility come from the generator, never detector scores.
All raw task rounds (including warm-up rounds) enter the undefended audit. A
missing defended estimate is unavailable, not evidence of successful protection.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import ROUND, retained_mask


KEYS = ROUND + ['source_id']
TARGETS = ('target_bias', 'change_suppression', 'false_trend')


def _bool(series):
    return series.map(lambda x: str(x).lower() in ('true', '1', '1.0'))


def _frame(value):
    if value is None:
        return None
    result = pd.read_csv(value) if isinstance(value, (str, Path)) else value.copy()
    for key in ('sequence_id', 'task_id', 'source_id', 'query_id'):
        if key in result:
            result[key] = result[key].astype(str)
    if 'timestamp' in result and not pd.api.types.is_numeric_dtype(result.timestamp):
        result['timestamp'] = pd.to_datetime(result.timestamp, utc=True, errors='raise').astype('int64') / 1e9
    return result


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _number(value):
    return float(value) if _finite(value) else None


def _direction(value):
    if str(value).lower() in ('up', 'positive', 'increase'):
        return 1.
    if str(value).lower() in ('down', 'negative', 'decrease'):
        return -1.
    return float(np.sign(float(value))) if _finite(value) else 0.


def _groups(frame):
    return {} if frame is None or frame.empty else dict(tuple(frame.groupby(ROUND, sort=False)))


def _estimate(group, value, query, settings, threshold=None, filtered=False):
    if group is None or group.empty or value not in group:
        return np.nan
    if query is None:
        chosen = group.loc[retained_mask(group, threshold)] if filtered else group
        return float(chosen[value].median()) if len(chosen) else np.nan
    distance = np.hypot(group.condition_x - query['condition_x'], group.condition_y - query['condition_y'])
    valid = np.isfinite(group[value]) & np.isfinite(distance)
    radius = settings.get('query_radius')
    if radius is not None:
        valid &= distance <= float(radius)
    chosen = group.loc[valid].copy()
    chosen['_distance'] = distance.loc[valid]
    chosen = chosen.sort_values(['_distance', 'source_id'], kind='stable').head(int(settings.get('query_neighbors', 8)))
    if filtered:
        chosen = chosen.loc[retained_mask(chosen, threshold)]
    if chosen.source_id.nunique() < int(settings.get('query_min_sources', 3)):
        return np.nan
    return float(np.average(chosen[value], weights=1 / np.maximum(chosen._distance.to_numpy(float), 1e-6)))


def _native_map(frame, value):
    if frame is None or frame.empty:
        return {}
    frame = frame.copy()
    if 'query_id' not in frame:
        frame['query_id'] = 'task_center'
    column = 'estimate_' + value[len('value_'):]
    if column not in frame:
        return {}
    if frame.duplicated(ROUND + ['query_id']).any():
        raise ValueError('Duplicate native estimate identities')
    return {tuple(row[k] for k in ROUND + ['query_id']): row[column]
            for _, row in frame.iterrows()}


def _round_estimates(raw, predictions, baseline_predictions, native, baseline_native, threshold, config, value):
    settings = config.get('evaluation', config)
    query_path = settings.get('queries_path', settings.get('query_path'))
    queries = _frame(query_path) if query_path else None
    if queries is not None:
        required = ROUND + ['condition_x', 'condition_y', 'reference_0']
        if not set(required).issubset(queries):
            raise ValueError('Query sidecar is missing required columns')
        if value != 'value_0':
            raise ValueError('Fixed IDW query evaluation supports value_0')
        if 'query_id' not in queries:
            queries['query_id'] = queries.index.astype(str)
        if 'raw_node_id' in queries and 'raw_node_id' in raw:
            if set(queries.raw_node_id) & set(raw.raw_node_id):
                raise ValueError('Query sensors must be held out from reports')
        minimum, neighbors = int(settings.get('query_min_sources', 3)), int(settings.get('query_neighbors', 8))
        radius = settings.get('query_radius')
        if minimum < 1 or neighbors < minimum or (radius is not None and float(radius) <= 0):
            raise ValueError('Invalid query neighborhood')
    query_groups = _groups(queries)
    pred_groups, baseline_groups = _groups(predictions), _groups(baseline_predictions)
    native_values = _native_map(native, value)
    baseline_native_values = _native_map(baseline_native, value)
    baseline_column = 'baseline_' + value[len('value_'):]
    reference_column = 'truth_' + value[len('value_'):]
    records = []
    for key, group in raw.groupby(ROUND, sort=True):
        if queries is None:
            reference = np.nan
            if reference_column in group:
                refs = group[reference_column].dropna().to_numpy(float)
                if len(refs) and np.allclose(refs, refs[0]):
                    reference = float(refs[0])
            points = [{'query_id': 'task_center', 'reference_0': reference}]
        else:
            points = query_groups[key].to_dict('records') if key in query_groups else [
                {'query_id': '__missing_query__', 'reference_0': np.nan}]
        for point in points:
            spatial = point if queries is not None and 'condition_x' in point else None
            missing_query = queries is not None and spatial is None
            row = dict(zip(ROUND, key), query_id=str(point['query_id']), reference=point.get('reference_0', np.nan))
            row['undefended_clean'] = np.nan if missing_query else _estimate(group, baseline_column, spatial, settings)
            row['undefended_attack'] = np.nan if missing_query else _estimate(group, value, spatial, settings)
            for name, groups in (('defended_attack', pred_groups), ('defended_clean', baseline_groups)):
                row[name] = np.nan if missing_query else _estimate(groups.get(key), value, spatial, settings, threshold, True)
            full_key = key + (str(point['query_id']),)
            row['native_attack'] = native_values.get(full_key, np.nan)
            row['native_clean'] = baseline_native_values.get(full_key, np.nan)
            records.append(row)
    columns = ROUND + ['query_id', 'reference'] + [c + '_' + p for c in ('undefended', 'defended', 'native') for p in ('clean', 'attack')]
    return pd.DataFrame(records, columns=columns), ('fixed_inverse_distance' if queries is not None else 'task_median')


def _assess(row, anchor, channel, kind, direction, harm_threshold, stable_tolerance,
            objective='mechanism_specific'):
    clean, attack, reference = (row[channel + '_clean'], row[channel + '_attack'], row['reference'])
    result = {'available': _finite(clean) and _finite(attack), 'assessed': False,
              'displacement': _number(attack - clean), 'error_increase': None,
              'reference_change': None, 'response_loss': None, 'harm': None,
              'displacement_exceeds_threshold': None}
    if result['available'] and harm_threshold is not None:
        result['displacement_exceeds_threshold'] = bool(abs(attack - clean) >= harm_threshold)
    if not result['available'] or not _finite(reference):
        return result
    error_increase = abs(attack - reference) - abs(clean - reference)
    result['error_increase'] = float(error_increase)
    has_anchor = (anchor is not None and _finite(anchor.get(channel + '_clean'))
                  and _finite(anchor.get('reference')))
    if has_anchor:
        reference_change = reference - anchor['reference']
        clean_response = clean - anchor[channel + '_clean']
        attack_response = attack - anchor[channel + '_clean']
        result['reference_change'] = float(reference_change)
        if kind == 'change_suppression':
            result['response_loss'] = float(np.sign(reference_change) * (clean_response - attack_response))
    if harm_threshold is None or kind not in TARGETS:
        return result
    if objective == 'instantaneous_error':
        # The sensing objective concerns the current estimate. Attack shape,
        # direction, and temporal response remain diagnostics, not harm gates.
        result.update(assessed=True, harm=bool(error_increase >= harm_threshold))
        return result
    if kind == 'target_bias':
        if direction == 0:
            return result
        result.update(assessed=True, harm=bool(direction * (attack - clean) >= harm_threshold
                                              and error_increase >= harm_threshold))
        return result
    if not has_anchor:
        return result
    result['assessed'] = True
    if kind == 'change_suppression':
        response_loss = result['response_loss']
        result['harm'] = bool(abs(reference_change) >= harm_threshold
                              and np.sign(reference_change) * clean_response >= harm_threshold
                              and abs(attack_response) < abs(clean_response)
                              and response_loss >= harm_threshold and error_increase >= harm_threshold)
    else:
        if direction == 0:
            result['assessed'] = False
            return result
        result['harm'] = bool(abs(reference_change) <= stable_tolerance
                              and direction * attack_response >= harm_threshold
                              and direction * (attack - clean) >= harm_threshold
                              and error_increase >= harm_threshold)
    return result


def _decisions(predictions, threshold, template):
    empty = template[KEYS].iloc[:0].copy()
    empty['_alarm'] = pd.Series(dtype=bool)
    if predictions is None or predictions.empty:
        return empty
    result = predictions.copy()
    if result.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction report identities')
    if 'prediction' in result:
        available = result.prediction.isin([0, 1, False, True])
        result['_alarm'] = result.prediction == 1
    elif 'score' in result and threshold is not None:
        available = np.isfinite(result.score)
        result['_alarm'] = result.score > threshold
    else:
        return empty
    if 'evaluated' in result:
        available &= _bool(result.evaluated)
    return result.loc[available, KEYS + ['_alarm']]


def _fpr(raw, decisions):
    joined = raw.merge(decisions, on=KEYS, how='inner', validate='one_to_one')
    if 'label' in joined:
        normal = joined.label.eq(0)
    elif 'original_label' in joined:
        normal = joined.original_label.eq(0)
    else:
        normal = pd.Series(False, index=joined.index)
    if 'injection_label' in joined:
        normal &= joined.injection_label.eq(0)
    if 'original_label' in joined:
        normal &= ~joined.original_label.eq(1)
    recovery = _bool(joined.window_contains_attack) if 'window_contains_attack' in joined else pd.Series(False, index=joined.index)
    result = {}
    for name, mask in (('background', normal & ~recovery), ('recovery', normal & recovery)):
        n, fp = int(mask.sum()), int(joined.loc[mask, '_alarm'].sum())
        result[name] = {'n': n, 'fp': fp, 'fpr': fp / n if n else None}
    result['semantics'] = 'normal endpoint label, no injected change or known original anomaly; recovery means window_contains_attack; scored decisions only'
    return result


def _summary(events):
    n = len(events)
    result = {'attempt_events': n, 'feasible_events': sum(e['feasible'] for e in events),
              'assessed_events': sum(e['undefended_effective'] is not None for e in events),
              'effective_events': sum(e['undefended_effective'] is True for e in events),
              'unscored_events': sum(e['scored_attempt_reports'] == 0 for e in events)}
    effective = [e for e in events if e['undefended_effective'] is True]
    result['unassessed_events'] = n - result['assessed_events']
    result['effective_attempt_rate'] = (result['effective_events'] / n
                                         if n and result['assessed_events'] == n else None)
    result['effective_assessed_event_rate'] = (result['effective_events'] / result['assessed_events']
                                               if result['assessed_events'] else None)
    for channel in ('defended', 'native'):
        provided = [e for e in events if e[channel + '_effective'] is not None]
        result[channel] = {'assessed_events': len(provided),
                           'effective_events': sum(e[channel + '_effective'] is True for e in events),
                           'protected_effective_events': sum(e[channel + '_protected'] is True for e in effective),
                           'protection_rate_on_effective_attempts': (
                               sum(e[channel + '_protected'] is True for e in effective) / len(effective) if effective else None)}
    detected = [e for e in events if e['detected']]
    result['detection'] = {'detected_events': len(detected),
                           'detected_event_rate': len(detected) / n if n else None,
                           'mean_delay_seconds': float(np.mean([e['detection_delay_seconds'] for e in detected])) if detected else None,
                           'delay_semantics': 'first alarm on an actually changed event report; undetected events remain in rate denominator'}
    feasible = [e for e in events if e['feasible']]
    result['detection']['detected_feasible_event_rate'] = (
        sum(e['detected'] for e in feasible) / len(feasible) if feasible else None)
    result['displacement_proxy'] = {
        'is_truth_error': False,
        'semantics': 'absolute estimate displacement from the uninjected counterfactual; '
                     'uses harm_threshold and min_harm_rounds; does not establish increased error',
        'channels': {},
    }
    for channel in ('undefended', 'defended', 'native'):
        assessed = [e for e in events if e[channel + '_proxy_shifted'] is not None]
        shifted = sum(e[channel + '_proxy_shifted'] is True for e in events)
        result['displacement_proxy']['channels'][channel] = {
            'assessed_events': len(assessed), 'shifted_events': shifted,
            'shifted_assessed_event_rate': shifted / len(assessed) if assessed else None,
        }
    return result


def evaluate_attacks(raw, predictions=None, threshold=None, output=None, config=None,
                     baseline_predictions=None, native_estimates=None, baseline_native_estimates=None,
                     attempt_metadata=None):
    """Return JSON-safe event metrics; optionally write event/round audit CSVs.

    ``config.harm_threshold`` is in ``value_column`` units (default value_0).
    ``min_harm_rounds`` counts task rounds, not sources or query points.
    Query settings are read from ``config.evaluation`` or the config itself.
    ``attempt_metadata['events']`` preserves scheduled attempts with no submitted
    reports; callers should supply the generator's per-split metadata.
    ``harm_objective`` / ``evaluation.attack_harm_objective`` can select
    ``instantaneous_error``: current absolute estimation error must increase by
    at least the harm threshold. The default ``mechanism_specific`` preserves
    legacy checkpoints' gates on direction and temporal response. Without an
    independent reference, harm stays unknown; estimate displacement is reported
    separately as a proxy.
    """
    if 'attack_id' not in raw:
        return {'status': 'not_enabled'}
    config = dict(config or {})
    raw, predictions, baseline_predictions = _frame(raw), _frame(predictions), _frame(baseline_predictions)
    native, baseline_native = _frame(native_estimates), _frame(baseline_native_estimates)
    value = config.get('value_column', 'value_0')
    if value not in raw or 'baseline_' + value[len('value_'):] not in raw:
        raise ValueError('Attack evaluation requires observed and baseline values')
    if raw.duplicated(KEYS).any():
        raise ValueError('Duplicate raw report identities')
    evaluation = config.get('evaluation', {})
    objective = config.get('harm_objective', config.get('attack_harm_objective',
                          evaluation.get('attack_harm_objective',
                                         evaluation.get('harm_objective', 'mechanism_specific'))))
    if objective not in ('instantaneous_error', 'mechanism_specific'):
        raise ValueError('harm_objective must be instantaneous_error or mechanism_specific')
    harm_threshold = config.get('harm_threshold', config.get('attack_harm_threshold',
                                evaluation.get('attack_harm_threshold', evaluation.get('harm_threshold'))))
    if harm_threshold is not None:
        harm_threshold = float(harm_threshold)
        if not np.isfinite(harm_threshold) or harm_threshold <= 0:
            raise ValueError('harm_threshold must be positive and finite')
    minimum = int(config.get('min_harm_rounds', config.get('attack_min_harm_rounds',
                      evaluation.get('attack_min_harm_rounds', evaluation.get('min_harm_rounds', 1)))))
    if minimum < 1:
        raise ValueError('min_harm_rounds must be positive')
    stable = float(config.get('stable_tolerance', evaluation.get('attack_stable_tolerance',
                   harm_threshold / 2 if harm_threshold is not None else 0.)))
    if not np.isfinite(stable) or stable < 0:
        raise ValueError('stable_tolerance must be finite and nonnegative')
    estimates, estimator = _round_estimates(raw, predictions, baseline_predictions, native, baseline_native,
                                            threshold, config, value)
    decisions = _decisions(predictions, threshold, raw)
    ids = raw.attack_id.notna() & raw.attack_id.astype(str).str.strip().ne('')
    observed = {str(key): group for key, group in raw.loc[ids].groupby('attack_id', sort=True)}
    registry = attempt_metadata.get('events', []) if isinstance(attempt_metadata, dict) else (attempt_metadata or [])
    metadata = {str(event.get('event_id', event.get('attack_id'))): event for event in registry}
    events, rounds = [], []
    for attack_id in sorted(set(observed) | set(metadata)):
        attempted = observed.get(attack_id, raw.iloc[:0])
        specification = metadata.get(attack_id, {})
        kind = str(specification.get('kind', attempted.attack_type.iloc[0]
                    if len(attempted) and 'attack_type' in attempted else 'unknown'))
        if 'feasible' in specification:
            feasible = bool(specification['feasible'])
        elif 'attack_event_feasible' in attempted:
            feasible = bool(_bool(attempted.attack_event_feasible).any())
        else:
            feasible = bool(_bool(attempted.attack_feasible).any()) if 'attack_feasible' in attempted else bool(len(attempted))
        direction = _direction(specification.get('direction', attempted.attack_direction.iloc[0]
                                 if len(attempted) and 'attack_direction' in attempted else 0.))
        scheduled = attempted
        if all(k in specification for k in ('sequence_id', 'task_id', 'time_start', 'time_end')):
            scheduled = raw.loc[(raw.sequence_id == str(specification['sequence_id']))
                                & (raw.task_id == str(specification['task_id']))
                                & (raw.timestamp >= specification['time_start'])
                                & (raw.timestamp <= specification['time_end'])]
        affected = set(map(tuple, scheduled[ROUND].to_numpy()))
        subset = estimates.loc[[tuple(row) in affected for row in estimates[ROUND].to_numpy()]].copy()
        event_rounds = []
        for _, row in subset.iterrows():
            series = estimates.loc[(estimates.sequence_id == row.sequence_id) & (estimates.task_id == row.task_id)
                                   & (estimates.query_id == row.query_id)]
            first_time = float(specification.get('time_start', scheduled.loc[
                (scheduled.sequence_id == row.sequence_id) & (scheduled.task_id == row.task_id), 'timestamp'].min()))
            previous = series.loc[series.timestamp < first_time].sort_values('timestamp')
            anchor = previous.iloc[-1] if len(previous) else None
            record = dict(row, attack_id=str(attack_id), attack_type=kind, feasible=feasible,
                          harm_objective=objective,
                          anchor_timestamp=float(anchor.timestamp) if anchor is not None else None)
            for channel in ('undefended', 'defended', 'native'):
                diagnostic = _assess(row, anchor, channel, kind, direction, harm_threshold, stable, objective)
                record.update({channel + '_' + key: val for key, val in diagnostic.items()})
            event_rounds.append(record)
        changed = attempted.loc[attempted.injection_label.eq(1)] if 'injection_label' in attempted else attempted
        scored_attempts = attempted[KEYS].merge(decisions, on=KEYS, how='inner', validate='one_to_one')
        detections = changed[KEYS].merge(decisions, on=KEYS, how='inner', validate='one_to_one')
        alarm_times = detections.loc[detections._alarm, 'timestamp']
        event = {'attack_id': str(attack_id), 'attack_type': kind, 'feasible': feasible,
                 'harm_objective': objective,
                 'infeasible_reason': '|'.join(sorted((set(attempted.get('attack_infeasible_reason', pd.Series(dtype=str)).dropna().astype(str))
                                                       | set(specification.get('reason_counts', {}))) - {''})),
                 'attempted_reports': len(attempted), 'changed_reports': len(changed),
                 'feasible_reports': int(_bool(attempted.attack_feasible).sum()) if 'attack_feasible' in attempted else len(attempted),
                 'registry_only_attempt': attack_id not in observed,
                 'task_rounds': len(affected), 'query_rounds': len(event_rounds),
                 'scored_attempt_reports': len(scored_attempts),
                 'scored_attack_reports': len(detections), 'detected': bool(len(alarm_times)),
                 'detection_delay_seconds': float(alarm_times.min() - specification.get('time_start', attempted.timestamp.min())) if len(alarm_times) else None}
        for channel in ('undefended', 'defended', 'native'):
            assessed = [r for r in event_rounds if r[channel + '_assessed']]
            harmed = set(tuple(r[k] for k in ROUND) for r in assessed if r[channel + '_harm'])
            complete = bool(event_rounds) and len(assessed) == len(event_rounds)
            # Sufficient observed harm establishes success despite missing rounds;
            # missing rounds never establish that a defense prevented harm.
            effective = True if len(harmed) >= minimum else False if complete else None
            event[channel + '_effective'] = effective
            event[channel + '_harm_rounds'] = len(harmed)
            event[channel + '_assessed_query_rounds'] = len(assessed)
            event[channel + '_available_query_rounds'] = sum(r[channel + '_available'] for r in event_rounds)
            displacements = [abs(r[channel + '_displacement']) for r in event_rounds if r[channel + '_displacement'] is not None]
            event[channel + '_mean_absolute_displacement'] = float(np.mean(displacements)) if displacements else None
            increases = [r[channel + '_error_increase'] for r in event_rounds if r[channel + '_error_increase'] is not None]
            event[channel + '_mean_error_increase'] = float(np.mean(increases)) if increases else None
            event[channel + '_complete'] = complete
            proxy_assessed = [r for r in event_rounds if r[channel + '_displacement_exceeds_threshold'] is not None]
            shifted = {tuple(r[k] for k in ROUND) for r in proxy_assessed
                       if r[channel + '_displacement_exceeds_threshold']}
            proxy_complete = bool(event_rounds) and len(proxy_assessed) == len(event_rounds)
            event[channel + '_proxy_shifted'] = (True if len(shifted) >= minimum
                                                else False if proxy_complete else None)
            event[channel + '_proxy_shift_rounds'] = len(shifted)
        for channel in ('defended', 'native'):
            event[channel + '_protected'] = bool(event['undefended_effective'] is True
                                                and event[channel + '_effective'] is False
                                                and event[channel + '_complete'])
        events.append(event)
        rounds.extend(event_rounds)
    result = _summary(events)
    result.update(status='ok' if harm_threshold is not None else 'harm_threshold_not_configured',
                  harm_objective=objective, harm_threshold=harm_threshold,
                  min_harm_rounds=minimum, stable_tolerance=stable,
                  value_column=value, undefended_estimator=estimator,
                  reference_semantics='independent_reference_required_for_harm; absent_reference_reports_displacement_only',
                  target_semantics=({kind: 'current absolute-error increase meets harm_threshold; '
                                          'attack shape is not an additional success criterion'
                                     for kind in TARGETS} if objective == 'instantaneous_error' else {
                      'target_bias': 'directional displacement and absolute-error increase both meet harm_threshold',
                      'change_suppression': 'true change and clean response meet threshold; attack reduces response amplitude and increases error',
                      'false_trend': 'directional apparent change from causal pre-event clean anchor while reference is stable; not fitted slope'}),
                  false_positive_rate=_fpr(raw, decisions),
                  per_kind={kind: _summary([e for e in events if e['attack_type'] == kind])
                            for kind in sorted(set(e['attack_type'] for e in events))})
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(events, columns=None if events else ['attack_id', 'attack_type', 'feasible']).to_csv(
            destination / 'attack_events.csv', index=False)
        pd.DataFrame(rounds, columns=None if rounds else ['attack_id'] + ROUND + ['query_id']).to_csv(
            destination / 'attack_rounds.csv', index=False)
    return result
