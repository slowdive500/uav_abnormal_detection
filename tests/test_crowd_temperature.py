"""Parser-contract tests only: these fixtures are NOT the downloaded dataset."""
import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.data.crowd_temperature import (OFFICIAL_MD5, prepare_crowd_temperature,
                                                   acquire_crowd_temperature)
from uav_abnormality.data.download import file_digest


def _write(tmp_path, data=None, **extra):
    raw = data if data is not None else pd.DataFrame({
        "Taxi ID": ["001", "002", "003", "001", "003"],
        "Grid": ["A", "A", "A", "B", "B"],
        "Date": ["2014-02-01"] * 5, "Period": ["first"] * 3 + ["third"] * 2,
        "Temperature": [10., 12., 8., 14., 15.],
        "Ground Truth": [10., 10., 10., 14., 14.],
        "Class": ["honest", "dishonest", "misleading", "honest", "misleading"],
        "Sigma": [.1, .3, .2, .1, .2],
    })
    path = tmp_path / "parser_fixture.csv"
    raw.to_csv(path, index=False)
    config = {"input": str(path), "expected_checksum": "sha256:" + file_digest(path),
              "period_to_hours": {"first": 0, "third": 12}, "timezone": "Europe/Rome"}
    config.update(extra)
    return config


def test_identity_clock_sparse_coverage_and_unknown_labels_are_preserved(tmp_path):
    frame = prepare_crowd_temperature(_write(tmp_path, virtual_uav=True))
    assert set(frame.source_id) == {"001", "002", "003"}
    assert set(frame.task_id) == {"A", "B"}
    assert set(frame.timestamp) == {pd.Timestamp("2014-01-31T23:00:00Z").timestamp(),
                                    pd.Timestamp("2014-02-01T11:00:00Z").timestamp()}
    assert len(frame) == 5  # No invented second-period or missing-source reports.
    assert (frame.label == -1).all()
    assert frame.attrs["rounds_with_three_sources"] == 1
    assert frame.attrs["median_step_seconds"] == 43200
    assert frame.virtual_uav.all()
    assert set(frame.source_role) == {"virtual_uav_from_taxi"}
    assert frame.attrs["official_original_checksum_verified"] is False
    assert not frame.attrs["adapter_generated_observations"]


def test_truth_noise_and_honesty_are_audit_only_not_observations_or_labels(tmp_path):
    frame = prepare_crowd_temperature(_write(tmp_path))
    assert [c for c in frame if c.startswith("value_")] == ["value_0"]
    assert not [c for c in frame if c.startswith("condition_")]
    assert frame.loc[frame.source_id == "002", "value_0"].tolist() == [12.]
    assert frame.loc[frame.source_id == "002", "truth_0"].tolist() == [10.]
    assert (frame.label == -1).all()  # Even a dishonest source can report accurately.
    assert {"audit_raw_class", "audit_noise", "audit_source_class"} <= set(frame.columns)


def test_no_truth_or_label_columns_are_fabricated(tmp_path):
    raw = pd.DataFrame({"source_id": ["a"], "task_id": ["r"], "timestamp": [1.5], "temperature": [5.]})
    frame = prepare_crowd_temperature(_write(tmp_path, raw, timestamp_unit="s"))
    assert "truth_0" not in frame
    assert frame.label.tolist() == [-1]
    assert frame.timestamp.tolist() == [1.5]


def test_missing_measurement_is_retained_without_an_invented_value(tmp_path):
    raw = pd.DataFrame({"source_id": ["a", "b"], "task_id": ["r", "r"],
                        "timestamp": [0, 0], "temperature": [5., np.nan]})
    frame = prepare_crowd_temperature(_write(tmp_path, raw, timestamp_unit="s"))
    assert len(frame) == 2
    assert np.isnan(frame.loc[frame.source_id == "b", "value_0"].iloc[0])
    assert (frame.label == -1).all()


def test_reject_conflicting_truth_and_duplicate_reports(tmp_path):
    config = _write(tmp_path)
    raw = pd.read_csv(config["input"], dtype=str)
    raw.loc[0, "Ground Truth"] = "11"
    with pytest.raises(ValueError, match="Conflicting reference truth"):
        prepare_crowd_temperature(_write(tmp_path, raw))
    raw.loc[0, "Ground Truth"] = "10"
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_crowd_temperature(_write(tmp_path, pd.concat([raw, raw.iloc[[0]]])))


def test_honesty_class_cannot_be_imported_as_per_report_labels(tmp_path):
    config = _write(tmp_path, columns={"label": "Class"}, label_semantics="per_report_anomaly",
                    label_map={"honest": 0, "dishonest": 1, "misleading": 1})
    with pytest.raises(ValueError, match="honesty/noise"):
        prepare_crowd_temperature(config)


def test_explicit_report_labels_require_a_complete_mapping(tmp_path):
    raw = pd.DataFrame({"source_id": ["a", "b", "c"], "task_id": ["r"] * 3,
                        "timestamp": [0] * 3, "temperature": [10, 11, 9],
                        "Verified report status": ["good", "bad", None]})
    config = _write(tmp_path, raw, timestamp_unit="s", columns={"label": "Verified report status"},
                    label_semantics="per_report_anomaly", label_map={"good": 0, "bad": 1})
    assert prepare_crowd_temperature(config).label.tolist() == [0, 1, -1]
    config["label_map"] = {"good": 0}
    with pytest.raises(ValueError, match="Unmapped report labels"):
        prepare_crowd_temperature(config)


def test_time_interpretations_are_never_silently_guessed(tmp_path):
    config = _write(tmp_path)
    config.pop("period_to_hours")
    with pytest.raises(ValueError, match="explicit period_to_hours"):
        prepare_crowd_temperature(config)
    config = _write(tmp_path)
    config.pop("timezone")
    with pytest.raises(ValueError, match="explicit timezone"):
        prepare_crowd_temperature(config)
    raw = pd.DataFrame({"source_id": ["a"], "task_id": ["r"], "Date": [20140201],
                        "Period": [1], "temperature": [10.]})
    config = _write(tmp_path, raw, date_format="%Y%m%d", period_to_hours={1: 0})
    assert prepare_crowd_temperature(config).timestamp.iloc[0] == pd.Timestamp("2014-01-31T23:00:00Z").timestamp()


def test_checksum_guard_and_missing_file_provide_actionable_failure(tmp_path):
    with pytest.raises(FileNotFoundError, match="IEEE login"):
        prepare_crowd_temperature({"input": str(tmp_path / "missing.csv")})
    config = _write(tmp_path)
    config.pop("expected_checksum")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_crowd_temperature(config)
    with pytest.raises(ValueError, match="checksum mismatch"):
        acquire_crowd_temperature(tmp_path / "download", input_path=config["input"])
    assert not (tmp_path / "download" / "crowd_temperature.csv").exists()
    assert not (tmp_path / "download" / "download_receipt.json").exists()


def test_source_subset_is_deterministic_and_never_pads_reports(tmp_path):
    config = _write(tmp_path, profile="smoke", profiles={"full": {}, "smoke": {"max_sources": 2}})
    frame = prepare_crowd_temperature(config)
    assert set(frame.source_id) == {"001", "002"}
    assert len(frame) == 3
    assert frame.attrs["original_rows"] == 5
    assert frame.attrs["selected_rows"] == 3


def test_manifest_pins_original_file_checksum():
    from pathlib import Path
    manifest = json.loads((Path(__file__).parents[1] / "data/manifests/crowd_temperature.json").read_text())
    assert manifest["required_files"][0]["checksum"] == "md5:" + OFFICIAL_MD5
    assert manifest["downloaded_data_files"] == 1


def test_verified_layout_mixed_dates_timezone_and_causal_buckets(tmp_path):
    # These minimal rows exercise the actual six-column schema, not benchmark results.
    raw = pd.DataFrame({'Taxi ID': ['1','1','2','3'], 'Date':['02-03-14',' 03/02/2014','02-03-14','02-04-14'],
        'Time':['01:00:00.1+01','02:00:00.123456+01','03:00:00+01','02:00:00+01'],
        'Latitude':[41.9]*4,'Longitude':[12.48]*4,'Temperature':[10.,14.,9.,100.]})
    config = _write(tmp_path, raw, layout='published_csv', grid_bounds=[41.78,42.,12.36,12.64],
                    time_bin_hours=6, source_selection_first_slots=1)
    result = prepare_crowd_temperature(config)
    assert set(result.source_id) == {'1','2'}  # future-only source is not part of the training cohort
    assert result.timestamp.unique().tolist() == [pd.Timestamp('2014-02-03T05:00:00Z').timestamp()]
    assert result.loc[result.source_id == '1','value_0'].item() == 12.
    assert result.loc[result.source_id == '1','sample_count'].item() == 2
    assert (result.sample_end < result.timestamp).all()
    assert 'truth_0' not in result and (result.label == -1).all()
    assert result.attrs['date_formats']['slash_rows'] == 1


def test_published_grid_does_not_silently_clip_outside_points(tmp_path):
    raw = pd.DataFrame({'Taxi ID':['1'], 'Date':['02-01-14'], 'Time':['01:00:00+01'],
                        'Latitude':[40.], 'Longitude':[12.48], 'Temperature':[10.]})
    config = _write(tmp_path, raw, layout='published_csv', grid_bounds=[41.78,42.,12.36,12.64])
    with pytest.raises(ValueError, match='outside'):
        prepare_crowd_temperature(config)


def test_published_minute_buckets_preserve_causal_aggregation_and_training_sources(tmp_path):
    raw = pd.DataFrame({
        'Taxi ID': ['1', '1', '2', '1', '3'],
        'Date': ['02-03-14'] * 5,
        'Time': ['01:01:00+01', '01:14:59+01', '01:05:00+01',
                 '01:15:00+01', '01:20:00+01'],
        'Latitude': [41.9] * 5, 'Longitude': [12.48] * 5,
        'Temperature': [10., 14., 9., 20., 100.],
    })
    config = _write(tmp_path, raw, layout='published_csv',
                    grid_bounds=[41.78, 42., 12.36, 12.64], grid_shape=[2, 2],
                    time_bin_minutes=15, source_selection_first_slots=1)
    result = prepare_crowd_temperature(config)
    assert set(result.source_id) == {'1', '2'}
    assert len(result) == 3
    first = pd.Timestamp('2014-02-03T00:15:00Z').timestamp()
    second = pd.Timestamp('2014-02-03T00:30:00Z').timestamp()
    assert set(result.timestamp) == {first, second}
    assert result.loc[(result.source_id == '1') & (result.timestamp == first), 'value_0'].item() == 12.
    assert result.loc[(result.source_id == '1') & (result.timestamp == second), 'value_0'].item() == 20.
    assert result.sample_count.sum() == 4
    assert (result.sample_end < result.timestamp).all()
    assert result.attrs['time_bin_minutes'] == 15
    assert result.attrs['time_bin_hours'] == .25
    assert result.attrs['source_selection']['first_slots'] == 1


@pytest.mark.parametrize('minutes', [0, -15, 7, 15.5, True])
def test_published_rejects_invalid_minute_bucket_width(tmp_path, minutes):
    raw = pd.DataFrame({'Taxi ID': ['1'], 'Date': ['02-01-14'],
                        'Time': ['01:00:00+01'], 'Latitude': [41.9],
                        'Longitude': [12.48], 'Temperature': [10.]})
    config = _write(tmp_path, raw, layout='published_csv',
                    grid_bounds=[41.78, 42., 12.36, 12.64], time_bin_minutes=minutes)
    with pytest.raises(ValueError, match='time_bin_minutes'):
        prepare_crowd_temperature(config)


def test_published_rejects_ambiguous_bucket_units(tmp_path):
    raw = pd.DataFrame({'Taxi ID': ['1'], 'Date': ['02-01-14'],
                        'Time': ['01:00:00+01'], 'Latitude': [41.9],
                        'Longitude': [12.48], 'Temperature': [10.]})
    config = _write(tmp_path, raw, layout='published_csv',
                    grid_bounds=[41.78, 42., 12.36, 12.64],
                    time_bin_minutes=30, time_bin_hours=6)
    with pytest.raises(ValueError, match='only one'):
        prepare_crowd_temperature(config)
