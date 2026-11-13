"""All-split contamination, counterfactual replay and window phase contracts."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from uav_abnormality.attack_pipeline import (
    annotate_attack_windows, prepare_corrupted_splits, restore_report_values)
from uav_abnormality.data.schema import read_table, validate_table
from uav_abnormality.protocol import SPLITS, Standardizer, split_table


def fixture():
    rows = []
    for day in range(4):
        for t in range(48):
            for source in range(8):
                value = 10 + 2 * np.sin(t / 4) + source * .1
                rows.append(dict(sequence_id=str(day), task_id='field', timestamp=100 * day + t,
                    source_id=str(source), value_0=value, truth_0=10 + 2 * np.sin(t / 4),
                    condition_x=float(source), label=-1))
    frame = validate_table(pd.DataFrame(rows))
    attack = {'protocol': 'targeted_report_attack', 'assume_unlabelled_normal': True,
        'source_fraction': .25, 'source_selection_seed': 6113, 'report_fraction': 1.,
        'duration': 6, 'coalition_sizes': [1, 2], 'max_local_fraction': .49,
        'kinds': ['target_bias', 'change_suppression', 'false_trend'],
        'constraints': {'target_delta_sigma': .75, 'max_delta_sigma': 1.,
            'max_delta_step_sigma': .25, 'min_effect_sigma': .1}}
    config = {'algorithm': 'condition_alignment', 'seed': 42,
        'split': {'mode': 'sequence', 'ratios': [.25] * 4},
        'window': {'window_size': 3, 'min_sources': 3, 'max_gap_seconds': 2},
        'corruption': {name: copy.deepcopy(attack) for name in SPLITS},
        'training': {'epochs': 1, 'device': 'cpu', 'threads': 1, 'batch_size': 32},
        'model': {'architecture': 'linear', 'min_references': 2, 'tau0': .1},
        'method': {'condition_columns': ['condition_x'],
            'calibration': {'alpha': .05, 'budget': 'auto', 'verification': 'injected_fault_oracle'},
            'environment': {'min_sources': 3}},
        'evaluation': {'attack_harm_threshold': .1, 'attack_min_harm_rounds': 2}}
    return frame, config


def test_all_splits_are_polluted_and_heldout_data_cannot_fit_attack_profile():
    raw, config = fixture()
    splits, _ = split_table(raw, config['split'])
    attacked, metadata, profile = prepare_corrupted_splits(splits, config, 42)
    for name in SPLITS:
        assert attacked[name].injection_label.sum() > 0, name
        np.testing.assert_allclose(restore_report_values(attacked[name]).value_0, splits[name].value_0)
        assert metadata[name]['protocol'] == 'targeted_report_attack'
    changed = copy.deepcopy(splits)
    for name in SPLITS[1:]:
        changed[name]['value_0'] += 10000
        changed[name]['truth_0'] = -10000
    second, _, same_profile = prepare_corrupted_splits(changed, config, 42)
    assert profile == same_profile
    pd.testing.assert_frame_equal(attacked['train'], second['train'])
    with pytest.raises(ValueError, match='all four splits'):
        bad = copy.deepcopy(config)
        del bad['corruption']['train']
        prepare_corrupted_splits(splits, bad, 42)
    with pytest.raises(ValueError, match='No realized training attacks'):
        impossible = copy.deepcopy(config)
        for stage in SPLITS:
            impossible['corruption'][stage]['duration'] = 1
        prepare_corrupted_splits(splits, impossible, 42)


def test_recovery_uses_task_clock_and_stops_at_observation_gap():
    raw = pd.DataFrame({'sequence_id': ['s'] * 7, 'task_id': ['t'] * 7,
        'source_id': ['a', 'b', 'b', 'b', 'a', 'a', 'b'],
        'timestamp': [0., 0., 1., 2., 2., 10., 10.],
        'injection_label': [1, 0, 0, 0, 0, 0, 0]})
    phase = annotate_attack_windows(raw, {'window_size': 2, 'max_gap_seconds': 2})
    assert phase.loc[0, 'evaluation_phase'] == 'active'
    assert phase.loc[4, 'evaluation_phase'] == 'background'  # a's own missing t1 still occupies a round
    assert not phase.loc[5, 'window_contains_attack']
    phase = annotate_attack_windows(raw, {'window_size': 3, 'max_gap_seconds': 2})
    assert phase.loc[4, 'evaluation_phase'] == 'recovery'
    assert not phase.loc[5, 'window_contains_attack']


def test_two_objectives_are_shared_across_all_contaminated_splits():
    raw, config = fixture()
    for stage in SPLITS:
        config['corruption'][stage]['kinds'] = ['target_bias', 'change_suppression']
    splits, _ = split_table(raw, config['split'])
    attacked, _, _ = prepare_corrupted_splits(splits, config, 42)
    for frame in attacked.values():
        assert frame.injection_label.sum() > 0
        assert 'false_trend' not in set(frame.attack_type)
    config['corruption']['test']['kinds'] = ['target_bias']
    with pytest.raises(ValueError, match='shared across splits'):
        prepare_corrupted_splits(splits, config, 42)


def test_no_attacks_preserves_every_split_value_and_unknown_background_labels(monkeypatch):
    raw, config = fixture()
    raw.loc[0, 'label'] = 1  # Known natural labels must not be relabelled clean.
    raw.loc[1, 'value_0'] = np.nan
    splits, _ = split_table(raw, config['split'])
    config['attack_mode'] = 'none'

    def unexpected_injection(*args, **kwargs):
        raise AssertionError('No-attack mode must not call any fault generator')

    monkeypatch.setattr('uav_abnormality.attack_pipeline.inject_faults', unexpected_injection)
    prepared, metadata, profile = prepare_corrupted_splits(splits, config, 42)
    assert profile is None
    for name in SPLITS:
        pd.testing.assert_frame_equal(prepared[name][['value_0', 'truth_0', 'condition_x']],
                                      splits[name][['value_0', 'truth_0', 'condition_x']])
        pd.testing.assert_series_equal(prepared[name].original_label, splits[name].label, check_names=False)
        pd.testing.assert_series_equal(prepared[name].baseline_0, splits[name].value_0, check_names=False)
        pd.testing.assert_series_equal(prepared[name].label, splits[name].label.replace(-1, 0))
        assert prepared[name].injection_label.eq(0).all()
        assert prepared[name].evaluation_phase.eq('background').all()
        assert not prepared[name].window_contains_attack.any()
        assert metadata[name]['protocol'] == 'none'
        assert metadata[name]['events'] == []
        assert metadata[name]['injected_reports'] == 0
    assert prepared['train'].loc[0, 'label'] == 1
    splits['test']['injection_label'] = 1
    with pytest.raises(ValueError, match='without prior injected attacks'):
        prepare_corrupted_splits(splits, config, 42)


def test_no_attack_preflight_still_requires_scoreable_reports_and_calibration():
    from uav_abnormality.preflight import validate_experiment_feasibility
    raw, config = fixture()
    splits, _ = split_table(raw, config['split'])
    config['attack_mode'] = 'none'
    config['protocol_feasibility'] = {'enabled': True, 'min_scored_attacks_per_split': 1,
                                     'min_retained_calibration_units': 40}
    prepared, metadata, _ = prepare_corrupted_splits(splits, config, 42)
    assert metadata['calibration']['preflight']['worst_case_retained_units'] >= 40
    assert all(metadata[name]['preflight']['common_scoreable_attacks'] == 0 for name in SPLITS)
    short = copy.deepcopy(prepared)
    first = short['calibration'].timestamp.min()
    short['calibration'] = short['calibration'][short['calibration'].timestamp < first + 12]
    with pytest.raises(ValueError, match='calibration retains'):
        validate_experiment_feasibility(short, config)
    unavailable = copy.deepcopy(prepared)
    unavailable['test']['condition_x'] = np.nan
    with pytest.raises(ValueError, match='test has no common scoreable reports'):
        validate_experiment_feasibility(unavailable, config)


def test_preflight_rejects_warmup_only_attacks_and_short_calibration():
    from uav_abnormality.preflight import validate_experiment_feasibility
    raw, config = fixture()
    splits, _ = split_table(raw, config['split'])
    attacked, _, _ = prepare_corrupted_splits(splits, config, 42)
    config['protocol_feasibility'] = {'enabled': True, 'min_retained_calibration_units': 40}
    result = validate_experiment_feasibility(attacked, config)
    assert result['calibration']['worst_case_retained_units'] >= 40
    assert all(result[name]['common_scoreable_attacks'] > 0 for name in SPLITS)
    short = copy.deepcopy(attacked)
    first = short['calibration'].timestamp.min()
    short['calibration'] = short['calibration'][short['calibration'].timestamp < first + 12]
    with pytest.raises(ValueError, match='calibration retains'):
        validate_experiment_feasibility(short, config)
    warmup = copy.deepcopy(attacked)
    warmup['test']['injection_label'] = warmup['test'].timestamp.eq(warmup['test'].timestamp.min()).astype(int)
    with pytest.raises(ValueError, match='test has 0 scoreable attacked reports'):
        validate_experiment_feasibility(warmup, config)


def test_training_and_evaluation_persist_attack_audits_and_rescore_counterfactual(tmp_path):
    from uav_abnormality.experiment import train_experiment, evaluate_checkpoint
    raw, config = fixture()
    reports = tmp_path / 'reports.csv'
    raw.to_csv(reports, index=False)
    config['data'] = {'path': str(reports)}
    output = tmp_path / 'run'
    result = train_experiment(config, output)
    assert 'attack_evaluation' in result
    for name in SPLITS:
        assert read_table(output / 'splits' / (name + '.csv')).injection_label.sum() > 0
        assert (output / 'attack_audits' / name / 'attack_events.csv').exists()
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=False)
    polluted_train = read_table(output / 'splits/train.csv')
    assert checkpoint['scaler'] == Standardizer().fit(polluted_train).to_dict()
    test = read_table(output / 'splits/test.csv')
    clean_path = tmp_path / 'counterfactual.csv'
    clean = restore_report_values(test).drop(columns=[c for c in test if c.startswith('attack_')])
    clean.to_csv(clean_path, index=False)
    evaluate_checkpoint(output / 'checkpoint.pt', clean_path, tmp_path / 'clean', device='cpu')
    direct = pd.read_csv(tmp_path / 'clean/predictions.csv')
    paired = pd.read_csv(output / 'baseline_predictions.csv')
    np.testing.assert_allclose(direct.score, paired.score, rtol=0, atol=0)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'clean/native_environment.csv'),
                                  pd.read_csv(output / 'baseline_native_environment.csv'))
    json.loads((output / 'metrics.json').read_text())


def test_no_attack_training_keeps_oracle_budget_and_reports_background_fpr(tmp_path):
    from uav_abnormality.experiment import train_experiment
    raw, config = fixture()
    config['attack_mode'] = 'none'
    reports = tmp_path / 'reports.csv'
    raw.to_csv(reports, index=False)
    config['data'] = {'path': str(reports)}
    output = tmp_path / 'no_attacks'
    result = train_experiment(config, output)
    assert result['status'] == 'single_label_class'
    assert result['labelled_reports'] > 0
    assert result['unknown_or_invalid_reports'] == 0
    assert result['false_positive_rate'] is not None
    assert result['injected_fault_detection']['tp'] == result['injected_fault_detection']['fn'] == 0
    assert result['injected_fault_detection']['false_positive_rate'] is not None
    assert 'attack_evaluation' not in result
    assert not (output / 'attack_audits').exists()
    assert not (output / 'attack_profile.json').exists()
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=False)
    assert checkpoint['config']['method']['calibration'] == config['method']['calibration']
    assert checkpoint['scaler'] == Standardizer().fit(raw[raw.sequence_id.eq('0')]).to_dict()
    calibration = json.loads((output / 'calibration.json').read_text())
    assert calibration['verification'] == 'injected_fault_oracle'
    assert calibration['m'] == calibration['budget_limit'] > 0
    assert calibration['removed'] == 0
    assert calibration['retained'] == calibration['n']
    manifest = json.loads((output / 'split_manifest.json').read_text())
    for name in SPLITS:
        assert manifest['corruptions'][name]['protocol'] == 'none'
        assert read_table(output / 'splits' / (name + '.csv')).injection_label.eq(0).all()
