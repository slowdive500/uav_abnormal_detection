import numpy as np
import pandas as pd
import pytest
import torch
from uav_abnormality.corruption import inject_faults
from uav_abnormality.evaluation import evaluate_references, evaluate_queries
from uav_abnormality.experiment import score_windows
from uav_abnormality.models.prbtd import PRBTD
from uav_abnormality.protocol import Standardizer, Windows, split_table


def frame():
    return pd.DataFrame([dict(sequence_id='campaign', task_id=task, timestamp=float(t),
                              source_id=str(source), value_0=float(2 * t + source), label=-1,
                              truth_0=float(2 * t), condition_x=float(source), condition_y=0.)
                         for t in range(24) for task in ('a', 'b') for source in range(3)])


@pytest.mark.parametrize('kind', ['offset', 'drift', 'spike', 'stuck', 'replay', 'replacement'])
def test_fault_audit_preserves_reference_and_missingness(kind):
    data = frame()
    data.loc[5, 'value_0'] = np.nan
    damaged, audit = inject_faults(data, {'assume_unlabelled_normal': True,
        'source_fraction': 1., 'report_fraction': .4, 'duration': 3, 'kinds': [kind]}, [1.], 3)
    np.testing.assert_array_equal(data.truth_0, damaged.truth_0)
    np.testing.assert_array_equal(data.value_0, damaged.baseline_0)
    assert data.value_0.isna().equals(damaged.value_0.isna())
    assert (damaged.original_label == -1).all()
    changed = ((data.value_0 - damaged.value_0).abs() > 1e-12).fillna(False)
    assert (changed.astype(int) == damaged.injection_label).all()
    assert audit['injected_reports'] > 0
    assert sum(e['changed_reports'] for e in audit['events']) == int(changed.sum())


def test_faults_do_not_read_across_split_or_modify_environment():
    parts, _ = split_table(frame(), {'mode': 'chronological'})
    config = {'assume_unlabelled_normal': True, 'source_fraction': 1., 'report_fraction': .6,
              'duration': 3, 'kinds': ['replay']}
    damaged, _ = inject_faults(parts['test'], config, [1.], 19)
    for (task, source), group in damaged.groupby(['task_id', 'source_id']):
        first = group.sort_values('timestamp').iloc[0]
        assert first.injection_label == 0  # no train/calibration history for replay
        valid = parts['test'][(parts['test'].task_id == task) & (parts['test'].source_id == source)]
        assert set(group.value_0).issubset(set(valid.value_0))


def test_reference_only_columns_never_enter_model_windows():
    raw = frame()
    scaler = Standardizer().fit(raw)
    changed = raw.copy()
    changed['truth_0'] = 1e20
    changed['label'] = 1
    changed['baseline_0'] = -1e20
    a = Windows(raw, ['0', '1', '2'], scaler, 3)
    b = Windows(changed, ['0', '1', '2'], scaler, 3)
    torch.testing.assert_close(a[4]['x'], b[4]['x'])
    assert scaler.columns == ['value_0']


def test_source_reputation_uses_same_past_for_simultaneous_tasks():
    data = frame().query('timestamp < 8').copy()
    torch.manual_seed(7)
    model = PRBTD(3, 1, 3, d_model=4, d_inner=8, dk_s=2, dk_t=2, growth_rate=2).eval()
    windows = Windows(data, ['0', '1', '2'], Standardizer().fit(data), 3)
    scored = score_windows(model, windows, torch.device('cpu'), reputation_scope='source')
    for _, group in scored.groupby(['timestamp', 'source_id']):
        assert group.reputation_prior.nunique() == 1
        assert group.reputation.nunique() == 1
    for _, group in scored[scored.task_id == 'a'].groupby('source_id'):
        group = group.sort_values('timestamp')
        np.testing.assert_allclose(group.reputation_prior.iloc[1:], group.reputation.iloc[:-1])
    repeated = score_windows(model, windows, torch.device('cpu'), batch_size=1, reputation_scope='source')
    pd.testing.assert_frame_equal(scored, repeated)
    future = data.copy()
    future.loc[future.timestamp == 7, 'value_0'] += 1000
    future_windows = Windows(future, ['0', '1', '2'], windows.scaler, 3)
    rescored = score_windows(model, future_windows, torch.device('cpu'), reputation_scope='source')
    np.testing.assert_allclose(scored[scored.timestamp < 7].score, rescored[rescored.timestamp < 7].score)


def test_downstream_quality_and_all_rejected_abstention(tmp_path):
    data = frame().query('timestamp == 0 and task_id == "a"').copy()
    data['score'] = [0., .1, 10.]
    result = evaluate_references(data, .5, tmp_path)
    assert result['all_reports']['mae'] == 1.
    assert result['retained_reports']['mae'] == .5
    assert result['score_error_spearman'] == 1.
    result = evaluate_references(data, -1., tmp_path)
    assert result['environment']['0']['filtered']['n'] == 0
    assert result['retained_reports']['mae'] is None


def test_query_value_is_only_an_evaluation_reference(tmp_path):
    data = frame().query('timestamp == 0 and task_id == "a"').copy()
    data['score'] = [0., .1, 10.]
    query = pd.DataFrame([dict(sequence_id='campaign', task_id='a', timestamp=0.,
                               query_id='heldout', condition_x=.5, condition_y=0., reference_0=1.)])
    path = tmp_path / 'queries.csv'
    query.to_csv(path, index=False)
    result = evaluate_queries(data, .5, path, tmp_path, {'query_min_sources': 3})
    before = pd.read_csv(tmp_path / 'query_estimates.csv')
    assert result['all']['coverage'] == 1.
    assert result['filtered']['coverage'] == 0.
    query['reference_0'] = 1e8
    query.to_csv(path, index=False)
    evaluate_queries(data, .5, path, tmp_path, {'query_min_sources': 3})
    after = pd.read_csv(tmp_path / 'query_estimates.csv')
    np.testing.assert_array_equal(before.all_estimate, after.all_estimate)
    assert before.reference_0.iloc[0] != after.reference_0.iloc[0]


def test_iso_query_timestamps_match_report_seconds(tmp_path):
    data = frame().query('timestamp == 0 and task_id == "a"').copy()
    data['score'] = 0.
    query = pd.DataFrame([dict(sequence_id='campaign', task_id='a', timestamp='1970-01-01T00:00:00Z',
                               query_id='q', condition_x=1., condition_y=0., reference_0=1.)])
    path = tmp_path / 'q.csv'
    query.to_csv(path, index=False)
    assert evaluate_queries(data, 1., path, tmp_path)['query_rounds'] == 1


@pytest.mark.parametrize('magnitude', [float('inf'), float('nan')])
def test_nonfinite_fault_amplitude_is_rejected(magnitude):
    with pytest.raises(ValueError, match='magnitude'):
        inject_faults(frame(), {'magnitude': magnitude, 'assume_unlabelled_normal': True}, [1.], 0)


def test_scoring_stride_does_not_create_a_false_observation_gap():
    from uav_abnormality.experiment import has_observation_gap
    raw = frame()
    windows = Windows(raw, ['0', '1', '2'], Standardizer().fit(raw), 3, stride=3, max_gap_seconds=1.1)
    assert not has_observation_gap(windows, 0, 2., 5.)
    interrupted = raw.copy()
    interrupted.loc[interrupted.timestamp >= 4, 'timestamp'] += 100
    windows = Windows(interrupted, ['0', '1', '2'], Standardizer().fit(raw), 3, stride=3, max_gap_seconds=1.1)
    assert has_observation_gap(windows, 0, 2., 105.)


def test_unverified_partial_download_is_not_promoted_to_complete(tmp_path):
    from uav_abnormality.data.download import download_file
    original = tmp_path / 'original'
    original.write_bytes(b'complete source content')
    destination = tmp_path / 'download'
    destination.mkdir()
    (destination / 'file.part').write_bytes(b'truncated')
    result = download_file({'path': 'file', 'url': original.as_uri()}, destination)
    assert result.read_bytes() == original.read_bytes()


@pytest.mark.parametrize('algorithm', ['prbtd', 'tsadc'])
def test_direct_cli_resolves_same_model_with_or_without_override(tmp_path, monkeypatch, algorithm):
    from uav_abnormality.cli import main
    import uav_abnormality.experiment as experiment
    import yaml
    captured = []
    def record(config, output, evaluate_test):
        captured.append(config)
        return {'status': 'captured'}
    monkeypatch.setattr(experiment, 'train_experiment', record)
    for command in ['run', 'train']:
        config = {'algorithm': algorithm, 'models': {
            'prbtd': {'d_model': 12}, 'tsadc': {'hidden_dim': 20}}}
        path = tmp_path / 'experiment.yaml'
        path.write_text(yaml.safe_dump(config))
        argv = [command, '--config', str(path), '--output', str(tmp_path / 'run')]
        main(argv)
        main(argv + ['--algorithm', algorithm])
        assert captured[-1] == captured[-2]
        assert captured[-1]['model'] == config['models'][algorithm]
        other = 'tsadc' if algorithm == 'prbtd' else 'prbtd'
        main(argv + ['--algorithm', other])
        assert captured[-1]['algorithm'] == other
        assert captured[-1]['model'] == config['models'][other]
