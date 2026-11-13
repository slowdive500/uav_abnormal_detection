"""Checkpointed pipeline tests use synthetic fixtures, never benchmark claims."""
import numpy as np
import pandas as pd
import pytest
from uav_abnormality.experiment import train_experiment, evaluate_checkpoint
from uav_abnormality.fixtures import make_fixture

@pytest.mark.parametrize('algorithm', ['prbtd', 'tsadc'])
def test_full_pipeline_and_checkpoint_inference(tmp_path, algorithm):
    reports = tmp_path / 'fixture.csv'
    make_fixture(reports, sequences=8, steps=16, sources=3)
    config = {'algorithm': algorithm, 'seed': 7, 'data': {'path': str(reports)},
              'split': {'mode': 'sequence', 'ratios': [.5, .125, .125, .25]},
              'window': {'window_size': 4, 'min_sources': 3, 'max_gap_seconds': 5.1},
              'training': {'device': 'cpu', 'threads': 1, 'epochs': 1, 'patience': 1, 'batch_size': 32},
              'calibration': {'alpha': .1}}
    run = tmp_path / 'run'
    result = train_experiment(config, run)
    assert result['status'] == 'no_ground_truth_labels'
    assert 'f1' not in result
    assert result['warmup_or_ineligible_reports'] == 18
    before = pd.read_csv(run / 'predictions.csv')
    after_dir = tmp_path / 'inference'
    evaluate_checkpoint(run / 'checkpoint.pt', run / 'splits/test.csv', after_dir, 'cpu', metrics=False)
    after = pd.read_csv(after_dir / 'predictions.csv')
    np.testing.assert_allclose(before.score, after.score, rtol=1e-6, atol=1e-6)
    assert not (after_dir / 'metrics.json').exists()
    assert (run / 'calibration.json').exists()
    assert (run / 'split_manifest.json').exists()
