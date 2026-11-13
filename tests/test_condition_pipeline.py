"""Behavioral checks for the proposal's complete condition/report pipeline."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from uav_abnormality.condition_data import ConditionScaler, ConditionWindows
from uav_abnormality.condition_runtime import (
    apply_reputation_and_estimate, estimate_query, reputation_update,
    score_condition_windows)
from uav_abnormality.data.schema import KEYS, validate_table
from uav_abnormality.models.condition_alignment import ConditionAlignment
from uav_abnormality.protocol import Standardizer


def reports(sequences=1, steps=8, sources=4):
    rows = []
    for sequence in range(sequences):
        for t in range(steps):
            for source in range(sources):
                condition = -1. + 2 * source / (sources - 1)
                clean = 10 + np.sin(t / 3) + .7 * condition
                rows.append(dict(sequence_id='day_%02d' % sequence, task_id='environment',
                    timestamp=float(100 * sequence + t), source_id='uav_%d' % source,
                    value_0=clean, condition_x=condition, truth_0=clean, label=0))
    return validate_table(pd.DataFrame(rows))


def identity_scalers():
    features = Standardizer.from_dict({'columns': ['value_0'], 'mean': [0.], 'scale': [1.]})
    conditions = ConditionScaler.from_dict({'columns': ['condition_x'], 'mean': [0.], 'scale': [1.]})
    return features, conditions


def test_condition_windows_keep_unseen_sources_and_mask_missing_conditions():
    train = reports()
    scaler = Standardizer().fit(train)
    conditions = ConditionScaler().fit(train, ['condition_x'])
    test = train.copy()
    test.loc[test.source_id == 'uav_0', 'source_id'] = 'new_uav'
    test.loc[(test.source_id == 'new_uav') & (test.timestamp == 7), 'condition_x'] = np.nan
    data = ConditionWindows(test, sorted(train.source_id.unique()), scaler, conditions,
                            window_size=3, min_sources=3)
    assert set(data.sources) == set(train.source_id) | {'new_uav'}
    index = data.sources.index('new_uav')
    item = data[-1]
    assert item['mask'][-1, index].all()
    assert not item['condition_mask'][-1, index]
    assert torch.isfinite(item['conditions']).all()
    model = ConditionAlignment(1, 1)
    scored = score_condition_windows(model, data, 'cpu')
    assert 'new_uav' in set(scored.source_id)
    assert not ((scored.source_id == 'new_uav') & (scored.timestamp == 7)).any()
    assert np.isfinite(scored.score).all()


def test_condition_scaling_is_training_fitted_and_excludes_noncondition_columns():
    train = reports()
    scaler = ConditionScaler().fit(train, ['condition_x'])
    restored = ConditionScaler.from_dict(scaler.to_dict())
    np.testing.assert_array_equal(scaler.transform([[100.]]), restored.transform([[100.]]))
    with pytest.raises(ValueError, match='condition_'):
        ConditionScaler().fit(train, ['truth_0'])
    with pytest.raises(ValueError, match='condition_'):
        ConditionScaler().fit(train, ['label'])
    train['condition_x'] = np.nan
    with pytest.raises(ValueError, match='finite training'):
        ConditionScaler().fit(train, ['condition_x'])


def controlled_predictions():
    rows = []
    for t in (0., 1., 2.):
        for source, value, quality in [('a', 0., .75), ('b', 10., .4), ('c', 20., .4)]:
            rows.append(dict(sequence_id='one', task_id='task', timestamp=t,
                source_id=source, value_0=value, condition_x=0., quality=quality,
                prediction=0, score=1., error=1., dispersion=1., label=0))
    return pd.DataFrame(rows)


def test_environment_uses_old_reputation_times_quality_before_quality_update():
    raw = controlled_predictions()
    scaler, conditions = identity_scalers()
    config = {'reputation': {'initial': .5, 'eta_up': .05, 'eta_down': .25},
              'environment': {'min_sources': 3}}
    scored, estimates = apply_reputation_and_estimate(ConditionAlignment(1, 1), raw, raw,
        scaler, conditions, 'cpu', config)
    # Old weights [.375,.2,.2] select 10. Updating R first would select 0,
    # because [.5125*.75,.475*.4,.475*.4] give source a a majority.
    first = estimates[estimates.timestamp == 0].iloc[0]
    assert first.estimate_0 == 10
    assert first.effective_weight == pytest.approx(.775)
    first_reports = scored[scored.timestamp == 0].set_index('source_id')
    assert first_reports.loc['a', 'reputation_prior'] == .5
    assert first_reports.loc['a', 'reputation'] == pytest.approx(.5125)
    assert first_reports.loc['b', 'reputation'] == pytest.approx(.475)
    assert estimates[estimates.timestamp == 1].iloc[0].estimate_0 == 0


def test_absent_and_unassessed_reports_hold_reputation_and_alarm_does_not_drive_update():
    raw = controlled_predictions()
    # At t1 a submitted an unassessed report, b was absent, and only c scored.
    raw = raw[~((raw.timestamp == 1) & (raw.source_id == 'b'))].copy()
    predictions = raw[~((raw.timestamp == 1) & (raw.source_id == 'a'))].copy()
    predictions.loc[(predictions.timestamp == 0) & (predictions.source_id == 'a'), 'prediction'] = 1
    scaler, conditions = identity_scalers()
    scored, estimates = apply_reputation_and_estimate(ConditionAlignment(1, 1), predictions,
        raw, scaler, conditions, 'cpu', {'environment': {'min_sources': 3}})
    at_two = scored[scored.timestamp == 2].set_index('source_id')
    assert at_two.loc['a', 'reputation_prior'] == pytest.approx(.5125)
    assert at_two.loc['b', 'reputation_prior'] == pytest.approx(.475)
    assert at_two.loc['c', 'reputation_prior'] == pytest.approx(.45625)
    assert estimates[estimates.timestamp == 1].iloc[0].status == 'insufficient_contributions'
    assert np.isnan(estimates[estimates.timestamp == 1].iloc[0].estimate_0)
    assert reputation_update(.5, .75) == pytest.approx(.5125)


def test_query_aligns_to_requested_condition_and_ignores_truth_and_labels():
    scaler, conditions = identity_scalers()
    model = ConditionAlignment(1, 1, architecture='linear')
    with torch.no_grad():
        model.condition_model.weight.fill_(2.)
    group = pd.DataFrame({'source_id': ['a', 'b', 'c'], 'value_0': [8., 10., 12.],
        'condition_x': [-1., 0., 1.], 'quality': [.8, .8, .8],
        'reputation_prior': [.5, .5, .5], 'prediction': [0, 0, 0],
        'truth_0': [100., 200., 300.], 'label': [1, 1, 1], 'injection_label': [1, 1, 1]})
    estimate, count, mass = estimate_query(model, group, [3.], scaler, conditions,
                                            'cpu', {'min_sources': 3})
    np.testing.assert_allclose(estimate, [16.])
    assert count == 3 and mass == pytest.approx(1.2)
    changed = group.copy()
    changed[['truth_0', 'label', 'injection_label']] = -99999
    repeated = estimate_query(model, changed, [3.], scaler, conditions,
                              'cpu', {'min_sources': 3})
    np.testing.assert_array_equal(estimate, repeated[0])
    group.loc[0, 'prediction'] = 1
    unavailable, count, _ = estimate_query(model, group, [3.], scaler, conditions,
                                          'cpu', {'min_sources': 3})
    assert count == 2 and np.isnan(unavailable).all()


def test_scoring_and_reputation_are_independent_of_future_reports_and_metadata():
    raw = reports(steps=10)
    scaler, conditions = Standardizer().fit(raw), ConditionScaler().fit(raw, ['condition_x'])
    model = ConditionAlignment(1, 1, architecture='linear')
    with torch.no_grad():
        model.condition_model.weight.fill_(.3)
    def evaluate(frame):
        data = ConditionWindows(frame, sorted(raw.source_id.unique()), scaler, conditions,
                                window_size=3, min_sources=3)
        result = score_condition_windows(model, data, 'cpu')
        result['prediction'] = 0
        return apply_reputation_and_estimate(model, result, frame, scaler, conditions,
                                             'cpu', {'environment': {'min_sources': 3}})
    original_scores, original_estimates = evaluate(raw)
    changed = raw.copy()
    future = changed.timestamp > 5
    changed.loc[future, 'value_0'] += 1000
    changed.loc[future, 'condition_x'] += 1000
    changed.loc[future & (changed.source_id == 'uav_0'), 'source_id'] = 'future_new_uav'
    changed['truth_0'] = -9000
    changed['label'] = 1
    scores, estimates = evaluate(changed)
    columns = KEYS + ['error', 'quality', 'dispersion', 'score', 'reputation_prior', 'reputation']
    pd.testing.assert_frame_equal(original_scores.loc[original_scores.timestamp <= 5, columns].reset_index(drop=True),
                                   scores.loc[scores.timestamp <= 5, columns].reset_index(drop=True))
    selected = ['sequence_id', 'task_id', 'timestamp', 'estimate_0', 'effective_weight', 'status']
    pd.testing.assert_frame_equal(original_estimates.loc[original_estimates.timestamp <= 5, selected].reset_index(drop=True),
                                   estimates.loc[estimates.timestamp <= 5, selected].reset_index(drop=True))


def test_sequence_reset_and_later_source_start_use_initial_reputation():
    raw = controlled_predictions()
    raw2 = raw.copy()
    raw2['sequence_id'] = 'two'
    raw2['timestamp'] += 10
    raw2.loc[raw2.source_id == 'a', 'source_id'] = 'unseen_source'
    raw = pd.concat([raw, raw2], ignore_index=True)
    scaler, conditions = identity_scalers()
    scored, _ = apply_reputation_and_estimate(ConditionAlignment(1, 1), raw, raw,
        scaler, conditions, 'cpu', {'environment': {'min_sources': 3}})
    assert (scored.loc[scored.timestamp == 10, 'reputation_prior'] == .5).all()


def test_condition_pipeline_checkpoint_roundtrip_and_no_oracle_leakage(tmp_path):
    from uav_abnormality.condition_experiment import train_condition_experiment
    from uav_abnormality.experiment import evaluate_checkpoint
    raw = reports(sequences=4, steps=12)
    anomaly = ((raw.sequence_id == 'day_03') & (raw.timestamp == 307)
               & (raw.source_id == 'uav_0'))
    raw.loc[anomaly, 'value_0'] += 3
    raw.loc[anomaly, 'label'] = 1
    source = tmp_path / 'reports.csv'
    raw.to_csv(source, index=False)
    config = {'algorithm': 'condition_alignment', 'seed': 17,
        'data': {'path': str(source)}, 'split': {'mode': 'sequence', 'ratios': [.25] * 4},
        'window': {'window_size': 2, 'min_sources': 3},
        'model': {'architecture': 'linear', 'min_references': 2, 'tau0': .1},
        'method': {'condition_columns': ['condition_x'],
                   'calibration': {'verification': 'none', 'alpha': .2},
                   'environment': {'min_sources': 3}},
        'training': {'epochs': 2, 'batch_size': 8, 'device': 'cpu', 'threads': 1}}
    run = tmp_path / 'run'
    result = train_condition_experiment(config, run)
    assert result['algorithm'] == 'condition_alignment'
    assert result['scored_reports'] == 44
    assert result['native_environment']['query_rounds'] == 12
    assert (run / 'calibration_audit.csv').exists()
    test_path = run / 'splits' / 'test.csv'
    replay = tmp_path / 'replay'
    evaluate_checkpoint(run / 'checkpoint.pt', test_path, replay, device='cpu', metrics=False)
    first = pd.read_csv(run / 'predictions.csv')
    pd.testing.assert_frame_equal(first, pd.read_csv(replay / 'predictions.csv'))
    pd.testing.assert_frame_equal(pd.read_csv(run / 'native_environment.csv'),
                                   pd.read_csv(replay / 'native_environment.csv'))
    changed = pd.read_csv(test_path)
    changed['truth_0'] = -1000000
    changed['label'] = 1 - changed['label']
    changed['injection_label'] = 1
    changed_path = tmp_path / 'changed_oracle.csv'
    changed.to_csv(changed_path, index=False)
    altered = tmp_path / 'altered'
    evaluate_checkpoint(run / 'checkpoint.pt', changed_path, altered, device='cpu', metrics=False)
    columns = KEYS + ['error', 'quality', 'dispersion', 'score', 'p_value', 'prediction',
                      'reputation_prior', 'reputation']
    pd.testing.assert_frame_equal(first[columns], pd.read_csv(altered / 'predictions.csv')[columns])
    np.testing.assert_array_equal(pd.read_csv(run / 'native_environment.csv').estimate_0,
                                  pd.read_csv(altered / 'native_environment.csv').estimate_0)

    # Deployment can receive reports that cannot be assessed. It still writes
    # explicit coverage and missing environmental estimates, with valid JSON.
    missing_conditions = changed.copy()
    missing_conditions['condition_x'] = np.nan
    missing_path = tmp_path / 'no_conditions.csv'
    missing_conditions.to_csv(missing_path, index=False)
    unavailable_run = tmp_path / 'no_conditions_output'
    missing_result = evaluate_checkpoint(run / 'checkpoint.pt', missing_path,
                                         unavailable_run, device='cpu', metrics=True)
    assert missing_result['scored_reports'] == 0
    assert missing_result['scoring_coverage'] == 0
    assert len(pd.read_csv(unavailable_run / 'unassessed_reports.csv')) == len(changed)
    assert pd.read_csv(unavailable_run / 'native_environment.csv').estimate_0.isna().all()

    # A whole deployment file can lack quorum. Abstention should not abort the
    # exported run and does not require retraining the fixed condition model.
    two_sources = changed[changed.source_id.isin(['uav_0', 'uav_1'])].copy()
    two_path = tmp_path / 'two_sources.csv'
    two_sources.to_csv(two_path, index=False)
    no_quorum = evaluate_checkpoint(run / 'checkpoint.pt', two_path,
        tmp_path / 'two_sources_output', device='cpu', metrics=True)
    assert no_quorum['scored_reports'] == 0
    assert no_quorum['warmup_or_ineligible_reports'] == len(two_sources)

    # A small calibration pool at stringent alpha has no attainable alarm.
    # None is the serialized threshold, and explicit p-value decisions prevail.
    from uav_abnormality.calibration import fit_label_trim
    checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    checkpoint['calibration'], _ = fit_label_trim(pd.read_csv(run / 'calibration_scores.csv'),
                                                   {'verification': 'none', 'alpha': .001})
    checkpoint['threshold'] = checkpoint['calibration']['threshold']
    assert checkpoint['threshold'] is None
    small_path = tmp_path / 'small_calibration.pt'
    torch.save(checkpoint, small_path)
    small_result = evaluate_checkpoint(small_path, test_path,
        tmp_path / 'small_calibration', device='cpu', metrics=True)
    assert small_result['threshold'] is None
    assert small_result['fp'] == small_result['tp'] == 0


def test_training_parameters_and_calibration_ignore_label_and_truth_columns(tmp_path):
    from uav_abnormality.condition_experiment import train_condition_experiment
    raw = reports(sequences=4, steps=8)
    config = {'algorithm': 'condition_alignment', 'seed': 29,
        'split': {'mode': 'sequence', 'ratios': [.25] * 4},
        'window': {'window_size': 2, 'min_sources': 3},
        'model': {'architecture': 'linear'},
        'method': {'condition_columns': ['condition_x'],
                   'calibration': {'verification': 'none', 'alpha': .2}},
        'training': {'epochs': 2, 'batch_size': 8, 'device': 'cpu', 'threads': 1}}
    checkpoints = []
    for name in ('original', 'changed'):
        frame = raw.copy()
        if name == 'changed':
            frame['label'] = 1
            frame['truth_0'] = 123456
            frame['injection_label'] = 1
        path = tmp_path / (name + '.csv')
        frame.to_csv(path, index=False)
        run_config = copy.deepcopy(config)
        run_config['data'] = {'path': str(path)}
        output = tmp_path / name
        train_condition_experiment(run_config, output, evaluate_test=False)
        checkpoints.append(torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False))
    first, second = checkpoints
    assert first['scaler'] == second['scaler']
    assert first['condition_scaler'] == second['condition_scaler']
    assert first['calibration'] == second['calibration']
    assert first['best_epoch'] == second['best_epoch']
    for key in first['state_dict']:
        assert torch.equal(first['state_dict'][key], second['state_dict'][key])


def test_training_rejects_conditions_with_no_eligible_reference_neighbors(tmp_path):
    from uav_abnormality.condition_experiment import train_condition_experiment
    raw = reports(sequences=4, steps=4)
    path = tmp_path / 'spatially_separated.csv'
    raw.to_csv(path, index=False)
    config = {'algorithm': 'condition_alignment', 'seed': 11, 'data': {'path': str(path)},
        'split': {'mode': 'sequence', 'ratios': [.25] * 4},
        'window': {'window_size': 2, 'min_sources': 3},
        'model': {'architecture': 'linear', 'neighbor_radius': 1e-6, 'min_references': 2},
        'method': {'condition_columns': ['condition_x']},
        'training': {'epochs': 2, 'batch_size': 8, 'device': 'cpu', 'threads': 1}}
    output = tmp_path / 'rejected_run'
    with pytest.raises(ValueError, match='No evaluable condition-alignment reports in train'):
        train_condition_experiment(config, output)
    assert not (output / 'checkpoint.pt').exists()
    assert not (output / 'history.json').exists()
