"""Attack success requires task harm; coverage and counterfactuals stay explicit."""
import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.attack_evaluation import evaluate_attacks


def reports(times=(0, 1, 2)):
    rows = []
    for time in times:
        for source, value in enumerate((0., 2., 4.)):
            rows.append(dict(sequence_id='s', task_id='t', timestamp=float(time), source_id=str(source),
                             value_0=value, baseline_0=value, truth_0=2., label=0, original_label=0, injection_label=0,
                             attack_id='', attack_type='', attack_attempted=False, attack_feasible=False,
                             attack_direction=1., attack_infeasible_reason='', window_contains_attack=False))
    return pd.DataFrame(rows)


def attack(frame, time=1, kind='target_bias', value=9., identifier='event', feasible=True):
    index = frame.index[(frame.timestamp == time) & (frame.source_id == '0')]
    frame.loc[index, ['attack_id', 'attack_type', 'attack_attempted', 'attack_feasible']] = [identifier, kind, True, feasible]
    if feasible:
        frame.loc[index, 'value_0'] = value
        frame.loc[index, 'injection_label'] = 1
    else:
        frame.loc[index, 'attack_infeasible_reason'] = 'constraint_failure'
    return frame


def decisions(frame, alarm=None):
    result = frame.copy()
    result['prediction'] = 0
    if alarm is not None:
        result.loc[alarm, 'prediction'] = 1
    return result


def test_unscored_and_infeasible_attempts_stay_in_denominator(tmp_path):
    raw = attack(reports((0, 1, 2, 3)))
    attack(raw, time=3, identifier='infeasible', feasible=False)
    raw.loc[raw.timestamp == 2, 'window_contains_attack'] = True
    pred = decisions(raw.loc[raw.timestamp >= 2])
    pred.loc[(pred.timestamp == 2) & (pred.source_id == '0'), 'prediction'] = 1
    pred.loc[(pred.timestamp == 3) & (pred.source_id == '1'), 'prediction'] = 1
    result = evaluate_attacks(raw, pred, output=tmp_path,
                              config={'evaluation': {'attack_harm_threshold': 1., 'attack_min_harm_rounds': 1}})
    assert result['attempt_events'] == 2
    assert result['feasible_events'] == 1
    assert result['effective_events'] == 1
    assert result['unscored_events'] == 1
    assert result['detection']['detected_event_rate'] == 0.
    assert result['defended']['protected_effective_events'] == 0
    assert result['false_positive_rate']['recovery'] == {'n': 3, 'fp': 1, 'fpr': 1 / 3}
    assert result['false_positive_rate']['background'] == {'n': 3, 'fp': 1, 'fpr': 1 / 3}
    events = pd.read_csv(tmp_path / 'attack_events.csv')
    assert events.set_index('attack_id').loc['event', 'undefended_effective']
    assert pd.isna(events.set_index('attack_id').loc['event', 'defended_effective'])
    assert (tmp_path / 'attack_rounds.csv').exists()
    json.dumps(result, allow_nan=False)


def test_displacement_without_truth_is_not_claimed_as_harm(tmp_path):
    raw = attack(reports()).drop(columns='truth_0')
    result = evaluate_attacks(raw, config={'harm_threshold': 1,
                              'evaluation': {'attack_harm_objective': 'instantaneous_error'}}, output=tmp_path)
    assert result['assessed_events'] == result['effective_events'] == 0
    assert result['effective_attempt_rate'] is None
    events = pd.read_csv(tmp_path / 'attack_events.csv')
    assert events.undefended_mean_absolute_displacement.iloc[0] == 2.
    assert pd.isna(events.undefended_effective.iloc[0])
    assert pd.isna(events.undefended_mean_error_increase.iloc[0])
    assert 'displacement_only' in result['reference_semantics']
    proxy = result['displacement_proxy']
    assert proxy['is_truth_error'] is False
    assert proxy['channels']['undefended'] == {
        'assessed_events': 1, 'shifted_events': 1, 'shifted_assessed_event_rate': 1.}
    assert proxy['channels']['defended']['shifted_assessed_event_rate'] is None
    assert events.undefended_proxy_shifted.iloc[0]
    rounds = pd.read_csv(tmp_path / 'attack_rounds.csv')
    assert pd.isna(rounds.undefended_harm.iloc[0])
    assert rounds.undefended_displacement_exceeds_threshold.iloc[0]
    assert result['harm_objective'] == rounds.harm_objective.iloc[0] == events.harm_objective.iloc[0] == 'instantaneous_error'
    json.dumps(result, allow_nan=False)


def test_direction_and_error_increase_both_required(tmp_path):
    raw = attack(reports())
    raw['attack_direction'] = -1
    result = evaluate_attacks(raw, config={'harm_threshold': 1})
    assert result['harm_objective'] == 'mechanism_specific'  # old checkpoints retain their original rule
    assert result['assessed_events'] == 1 and result['effective_events'] == 0
    raw['attack_direction'] = 1
    raw['truth_0'] = 4.  # the same malicious displacement now improves accuracy
    assert evaluate_attacks(raw, config={'harm_threshold': 1})['effective_events'] == 0


@pytest.mark.parametrize('kind', ['target_bias', 'change_suppression', 'false_trend'])
def test_instantaneous_harm_does_not_require_direction_or_temporal_anchor(kind):
    raw = attack(reports((1,)), kind=kind)
    raw['attack_direction'] = -1
    settings = {'harm_threshold': 1., 'harm_objective': 'instantaneous_error'}
    result = evaluate_attacks(raw, config=settings)
    assert result['assessed_events'] == result['effective_events'] == 1
    # A malicious change that improves the current estimate is still not harm.
    raw['truth_0'] = 4.
    result = evaluate_attacks(raw, config=settings)
    assert result['assessed_events'] == 1 and result['effective_events'] == 0
    assert result['displacement_proxy']['channels']['undefended']['shifted_events'] == 1


def test_instantaneous_harm_keeps_temporal_response_as_diagnostic_only(tmp_path):
    raw = attack(reports(), kind='change_suppression')
    # The reference has no temporal change, yet pushing the estimate away is harmful.
    result = evaluate_attacks(raw, output=tmp_path,
                              config={'harm_threshold': 1., 'attack_harm_objective': 'instantaneous_error'})
    assert result['effective_events'] == 1
    rounds = pd.read_csv(tmp_path / 'attack_rounds.csv')
    assert rounds.undefended_reference_change.iloc[0] == 0.
    assert rounds.undefended_response_loss.iloc[0] == 0.
    legacy = evaluate_attacks(raw, config={'harm_threshold': 1., 'harm_objective': 'mechanism_specific'})
    assert legacy['assessed_events'] == 1 and legacy['effective_events'] == 0


def test_invalid_harm_objective_is_rejected():
    with pytest.raises(ValueError, match='harm_objective'):
        evaluate_attacks(attack(reports()), config={'harm_objective': 'displacement_is_truth'})


def test_counterfactual_defense_and_missing_estimates_are_distinct(tmp_path):
    raw = attack(reports())
    clean = raw.copy()
    clean['value_0'] = clean.baseline_0
    # Filtering the manipulated low source yields median 3, versus clean 2.
    pred = decisions(raw, (raw.timestamp == 1) & (raw.source_id == '0'))
    result = evaluate_attacks(raw, pred, config={'harm_threshold': 1.5},
                              baseline_predictions=decisions(clean), output=tmp_path)
    assert result['effective_events'] == 1
    assert result['defended']['protected_effective_events'] == 1
    assert result['detection']['detected_event_rate'] == 1
    assert result['detection']['mean_delay_seconds'] == 0
    missing = decisions(raw, raw.timestamp == 1)  # no retained output during attack
    missing_result = evaluate_attacks(raw, missing, config={'harm_threshold': 1.5},
                                      baseline_predictions=decisions(clean))
    assert missing_result['defended']['protected_effective_events'] == 0
    assert missing_result['defended']['assessed_events'] == 0


def test_change_suppression_uses_preevent_reference_and_response(tmp_path):
    raw = reports()
    raw.loc[raw.timestamp > 0, ['value_0', 'baseline_0']] += 5
    raw.loc[raw.timestamp > 0, 'truth_0'] += 5
    index = (raw.timestamp == 1) & (raw.source_id == '2')
    raw.loc[index, ['value_0', 'injection_label', 'attack_id', 'attack_type', 'attack_attempted', 'attack_feasible']] = [0., 1, 'suppress', 'change_suppression', True, True]
    result = evaluate_attacks(raw, config={'harm_threshold': 1.5}, output=tmp_path)
    assert result['effective_events'] == 1
    rounds = pd.read_csv(tmp_path / 'attack_rounds.csv')
    assert rounds.undefended_reference_change.iloc[0] == 5.
    assert rounds.undefended_response_loss.iloc[0] == 2.
    assert rounds.anchor_timestamp.iloc[0] == 0.
    # Without a causal anchor this is not assessed by borrowing later observations.
    no_anchor = evaluate_attacks(raw.loc[raw.timestamp > 0], config={'harm_threshold': 1.5})
    assert no_anchor['assessed_events'] == 0


def test_false_trend_requires_stable_reference_and_round_duration(tmp_path):
    raw = attack(reports(), kind='false_trend')
    result = evaluate_attacks(raw, config={'harm_threshold': 1., 'min_harm_rounds': 2})
    assert result['effective_events'] == 0
    result = evaluate_attacks(raw, config={'harm_threshold': 1.})
    assert result['effective_events'] == 1
    raw.loc[raw.timestamp == 1, 'truth_0'] = 5.
    assert evaluate_attacks(raw, config={'harm_threshold': 1.})['effective_events'] == 0


def test_idw_matches_common_query_rule_and_query_count_not_duration(tmp_path):
    raw = attack(reports())
    raw['condition_x'] = raw.source_id.astype(float) + 1.
    raw['condition_y'] = 0.
    queries = pd.DataFrame([dict(sequence_id='s', task_id='t', timestamp=float(t),
                                 query_id=q, condition_x=0., condition_y=0., reference_0=14 / 11)
                            for t in (0, 1, 2) for q in ('q1', 'q2')])
    path = tmp_path / 'queries.csv'
    queries.to_csv(path, index=False)
    config = {'evaluation': {'queries_path': str(path), 'query_min_sources': 3, 'query_neighbors': 3},
              'harm_threshold': 1., 'min_harm_rounds': 2}
    result = evaluate_attacks(raw, config=config, output=tmp_path)
    rows = pd.read_csv(tmp_path / 'attack_rounds.csv')
    assert rows.undefended_clean.to_numpy() == pytest.approx([14 / 11, 14 / 11])
    assert rows.undefended_attack.to_numpy() == pytest.approx([68 / 11, 68 / 11])
    assert result['effective_events'] == 0  # two queries, but just one attacked time
    assert result['displacement_proxy']['channels']['undefended']['shifted_events'] == 0


def test_native_estimates_use_separate_clean_counterfactual(tmp_path):
    raw = attack(reports())
    clean = pd.DataFrame([dict(sequence_id='s', task_id='t', timestamp=float(t), query_id='task_center', estimate_0=2.) for t in (0, 1, 2)])
    native = clean.copy()
    native.loc[native.timestamp == 1, 'estimate_0'] = 2.5
    result = evaluate_attacks(raw, config={'harm_threshold': 1}, native_estimates=native,
                              baseline_native_estimates=clean)
    assert result['native']['protected_effective_events'] == 1
    assert result['defended']['protected_effective_events'] == 0


def test_missing_threshold_and_disabled_mode():
    assert evaluate_attacks(reports().drop(columns='attack_id')) == {'status': 'not_enabled'}
    result = evaluate_attacks(attack(reports()))
    assert result['status'] == 'harm_threshold_not_configured'
    assert result['assessed_events'] == 0
    json.dumps(result, allow_nan=False)


def test_known_natural_anomalies_and_unknown_labels_are_not_normal_fpr():
    raw = attack(reports())
    raw.loc[(raw.timestamp == 0) & (raw.source_id == '0'), ['original_label', 'label']] = [1, 1]
    raw.loc[(raw.timestamp == 0) & (raw.source_id == '1'), ['original_label', 'label']] = [-1, -1]
    pred = decisions(raw, raw.timestamp == 0)
    result = evaluate_attacks(raw, pred, config={'harm_threshold': 1})
    assert result['false_positive_rate']['background'] == {'n': 6, 'fp': 1, 'fpr': 1 / 6}


def test_empty_attempt_registry_writes_readable_empty_artifacts(tmp_path):
    result = evaluate_attacks(reports(), output=tmp_path, config={'harm_threshold': 1})
    assert result['attempt_events'] == 0
    assert pd.read_csv(tmp_path / 'attack_events.csv').empty
    assert pd.read_csv(tmp_path / 'attack_rounds.csv').empty
    json.dumps(result, allow_nan=False)


def test_metadata_only_attempt_is_preserved_when_no_controlled_source_reports(tmp_path):
    raw = reports()
    metadata = {'events': [dict(event_id='no_source', kind='target_bias', feasible=False,
                               sequence_id='s', task_id='t', time_start=1., time_end=2.,
                               direction=1, source_ids=[], reason_counts={'no_controlled_source_present': 2})]}
    result = evaluate_attacks(raw, config={'harm_threshold': 1}, output=tmp_path, attempt_metadata=metadata)
    assert result['attempt_events'] == 1 and result['feasible_events'] == 0
    assert result['unscored_events'] == 1
    assert result['detection']['detected_event_rate'] == 0
    event = pd.read_csv(tmp_path / 'attack_events.csv').iloc[0]
    assert event.registry_only_attempt and event.attempted_reports == 0
    assert event.task_rounds == 2
    assert event.infeasible_reason == 'no_controlled_source_present'
    json.dumps(result, allow_nan=False)


def test_event_feasibility_is_not_all_of_report_feasibility():
    raw = attack(reports(), time=1, feasible=False)
    attack(raw, time=2)
    raw['attack_event_feasible'] = raw.attack_id.eq('event')
    result = evaluate_attacks(raw, config={'harm_threshold': 1})
    assert result['attempt_events'] == result['feasible_events'] == result['effective_events'] == 1


def test_idw_filtered_estimates_equal_existing_reference_evaluator(tmp_path):
    from uav_abnormality.evaluation import evaluate_queries
    raw = attack(reports())
    raw['condition_x'], raw['condition_y'] = raw.source_id.astype(float) + 1, 0.
    queries = pd.DataFrame([dict(sequence_id='s', task_id='t', timestamp=float(t), query_id='q',
                                 condition_x=0., condition_y=0., reference_0=14 / 11) for t in (0, 1, 2)])
    path = tmp_path / 'queries.csv'
    queries.to_csv(path, index=False)
    settings = {'queries_path': str(path), 'query_min_sources': 2, 'query_neighbors': 3}
    pred = decisions(raw.loc[raw.timestamp > 0])
    pred.loc[(pred.timestamp == 1) & (pred.source_id == '0'), 'prediction'] = 1
    evaluate_queries(pred, None, path, tmp_path, settings)
    expected = pd.read_csv(tmp_path / 'query_estimates.csv')
    evaluate_attacks(raw, pred, output=tmp_path, config={'harm_threshold': 1, 'evaluation': settings})
    actual = pd.read_csv(tmp_path / 'attack_rounds.csv')
    assert actual.defended_attack.iloc[0] == pytest.approx(expected.loc[expected.timestamp == 1, 'filtered_estimate'].iloc[0])
    assert actual.undefended_attack.iloc[0] == pytest.approx(expected.loc[expected.timestamp == 1, 'all_estimate'].iloc[0])
