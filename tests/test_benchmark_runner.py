"""Runner keeps failed seeds visible, resumes safely and always aggregates."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_report_benchmarks.py'
    spec = importlib.util.spec_from_file_location('report_benchmark_runner', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'configs/experiments'
    folder.mkdir(parents=True)
    reports = tmp_path / 'reports.csv'
    reports.write_text('unchanged input')
    (folder / 'crowd_temperature.yaml').write_text('data:\n  path: reports.csv\ntraining:\n  epochs: 20\n')


def test_failure_is_recorded_and_resume_preserves_completed_runs(tmp_path, monkeypatch):
    runner = load_runner()
    setup_config(tmp_path, monkeypatch)
    calls, summaries = [], []
    fail = [True]

    def fake_train(config, output):
        calls.append((config['algorithm'], config['seed']))
        output.mkdir(parents=True)
        (output / 'checkpoint.pt').write_text('partial or complete training')
        if fail[0] and calls[-1] == ('condition_alignment', 42):
            raise RuntimeError('deliberate training failure')
        (output / 'metrics.json').write_text(json.dumps({'algorithm': config['algorithm'], 'f1': .5}))

    def fake_analysis(root, **kwargs):
        summaries.append(json.loads((root / 'run_status.json').read_text()))
        assert kwargs['expected_seeds'] == [0, 42]
        (root / 'analysis').mkdir(exist_ok=True)
        (root / 'analysis/REPORT.md').write_text('summary')

    monkeypatch.setattr(runner, 'train_experiment', fake_train)
    monkeypatch.setitem(sys.modules, 'uav_abnormality.benchmark_analysis',
                        types.SimpleNamespace(analyze_benchmarks=fake_analysis))
    args = ['--datasets', 'crowd_temperature', '--algorithms', 'condition_alignment', 'prbtd',
            '--seeds', '0', '42', '--output', 'runs/check']
    assert runner.main(args) == 1
    root = tmp_path / 'runs/check'
    status = json.loads((root / 'run_status.json').read_text())
    assert status['counts']['completed'] == 3 and status['counts']['failed'] == 1
    assert len(calls) == 4 and len(summaries) == 1
    failed = next(r for r in status['runs'] if r['status'] == 'failed')
    assert 'deliberate training failure' in failed['error']
    assert 'Traceback' in (root / failed['log']).read_text()
    complete = root / 'crowd_temperature/prbtd/42/metrics.json'
    saved = complete.stat().st_mtime_ns
    fail[0] = False
    assert runner.main(args + ['--resume']) == 0
    assert calls[-1] == ('condition_alignment', 42) and len(calls) == 5
    assert complete.stat().st_mtime_ns == saved
    assert len(list((root / 'failed_attempts').rglob('checkpoint.pt'))) == 1
    assert summaries[-1]['counts']['completed'] == 4
    assert summaries[-1]['counts']['failed'] == 0
    # A data change must not reuse stale completed results.
    (tmp_path / 'reports.csv').write_text('changed input')
    with pytest.raises(ValueError, match='inputs changed'):
        runner.main(args + ['--resume'])


def test_duplicate_seed_or_incompatible_resume_is_rejected(tmp_path, monkeypatch):
    runner = load_runner()
    setup_config(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        runner.main(['--seeds', '42', '42', '--output', 'unused'])
    with pytest.raises(ValueError, match='identical'):
        runner.main(['--datasets', 'crowd_temperature', '--output', 'runs/absent', '--resume'])


def test_no_attack_plan_resolves_all_splits_and_rejects_cross_mode_resume(tmp_path, monkeypatch):
    runner = load_runner()
    setup_config(tmp_path, monkeypatch)
    original = {
        'data': {'path': 'reports.csv'}, 'training': {'epochs': 20},
        'corruption': {name: {'protocol': 'targeted_report_attack'}
                       for name in ('train', 'validation', 'calibration', 'test')},
        'protocol_feasibility': {'enabled': True, 'min_scored_attacks_per_split': 1,
                                 'min_retained_calibration_units': 40},
        'method': {'calibration': {'budget': 'auto', 'verification': 'injected_fault_oracle',
                                   'separation_seconds': 3600}},
    }
    import yaml
    path = tmp_path / 'configs/experiments/crowd_temperature.yaml'
    path.write_text(yaml.safe_dump(original))
    original_bytes = path.read_bytes()
    calls = []

    def fake_train(config, output):
        calls.append(config)
        output.mkdir(parents=True)
        (output / 'metrics.json').write_text('{}')

    monkeypatch.setattr(runner, 'train_experiment', fake_train)
    monkeypatch.setitem(sys.modules, 'uav_abnormality.benchmark_analysis',
                        types.SimpleNamespace(analyze_benchmarks=lambda *a, **kw: None))
    args = ['--datasets', 'crowd_temperature', '--algorithms', 'condition_alignment',
            '--seeds', '0', '--output', 'runs/no_attacks']
    assert runner.main(args + ['--no-attacks']) == 0
    resolved = calls[0]
    assert resolved['attack_mode'] == 'none'
    assert all(value == {'protocol': 'none', 'assume_unlabelled_normal': True}
               for value in resolved['corruption'].values())
    assert resolved['protocol_feasibility'] == {
        'enabled': True, 'min_scored_attacks_per_split': 0, 'min_retained_calibration_units': 40}
    assert resolved['method'] == original['method']
    assert resolved['training'] == original['training']
    assert path.read_bytes() == original_bytes
    root = tmp_path / 'runs/no_attacks'
    assert json.loads((root / 'run_plan.json').read_text())['attack_mode'] == 'none'
    assert runner.main(args + ['--attack-mode', 'none', '--resume']) == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match='attack mode'):
        runner.main(args + ['--resume'])


def test_legacy_configured_plan_remains_resumable(tmp_path, monkeypatch):
    runner = load_runner()
    setup_config(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'cli_main', lambda args: None)
    args = ['--datasets', 'crowd_temperature', '--output', 'runs/legacy', '--prepare-only']
    assert runner.main(args) == 0
    path = tmp_path / 'runs/legacy/run_plan.json'
    legacy = json.loads(path.read_text())
    del legacy['attack_mode']
    path.write_text(json.dumps(legacy))
    assert runner.main(args + ['--resume']) == 0
    with pytest.raises(ValueError, match='attack mode'):
        runner.main(args + ['--no-attacks', '--resume'])
