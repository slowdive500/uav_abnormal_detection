import h5py
import numpy as np
import pandas as pd
import pytest
import hashlib
import zipfile

from uav_abnormality.data.schema import feature_columns
from uav_abnormality.data.taxibj import decode_timestamps, prepare_taxibj, download_taxibj


def _field_file(tmp_path, dates, values=None):
    path = tmp_path / "BJ15_M32x32_T30_InOut.h5"
    if values is None:
        values = np.ones((len(dates), 2, 32, 32)) * 40
        values[:, 1] = 70
    with h5py.File(path, "w") as handle:
        handle["date"] = np.array(dates, dtype="S10")
        handle["data"] = values
    return {"raw_dir": str(tmp_path), "cells": [[3, 4], [3, 5]],
            "source_count": 8, "participation_probability": 0.6,
            "min_sources": 3, "seed": 12, "sequence_mode": "day"}


def test_beijing_slots_are_not_utc_or_seconds_since_midnight():
    stamps, days, slots = decode_timestamps([b"2016010101", b"2016010148", b"2016010201"])
    assert stamps[0] == pd.Timestamp("2015-12-31T16:30:00Z").timestamp()
    assert stamps[1] == pd.Timestamp("2016-01-01T16:00:00Z").timestamp()
    assert np.diff(stamps).tolist() == [47 * 1800, 1800]
    assert days.tolist() == ["20160101", "20160101", "20160102"]
    assert slots.tolist() == [1, 48, 1]


@pytest.mark.parametrize("date", [b"2016010100", b"2016010149", b"notadate", b"2016130101"])
def test_reject_invalid_dates(date):
    with pytest.raises(ValueError):
        decode_timestamps([date])


def test_real_field_is_eval_only_and_reports_vary_by_source(tmp_path):
    dates = ["20160101%02d" % slot for slot in range(1, 49)]
    config = _field_file(tmp_path, dates)
    config["channels"] = ["inflow", "outflow"]
    reports = prepare_taxibj(config)
    assert feature_columns(reports) == ["value_0", "value_1"]
    assert set(reports.truth_0) == {40}
    assert set(reports.truth_1) == {70}
    assert not np.array_equal(reports.value_0, reports.truth_0)
    assert reports.groupby(["task_id", "timestamp"]).value_0.nunique().min() >= 3
    assert set(reports.label) == {0}
    assert (reports.timestamp == reports.condition_bin_end).all()
    assert (reports.timestamp - reports.condition_bin_start == 1800).all()
    assert reports.reference_semantics.eq("traffic_field_ground_truth").all()
    assert reports.attrs["provenance"]["actual_uav_flights"] is False
    pd.testing.assert_frame_equal(reports, prepare_taxibj(config))


def test_incomplete_days_not_interpolated_and_invalid_cells_are_missing(tmp_path):
    dates = ["20160101%02d" % slot for slot in range(1, 49)] + ["2016010201"]
    values = np.ones((len(dates), 2, 32, 32)) * 12
    values[4, 0, 3, 4] = -1
    config = _field_file(tmp_path, dates, values)
    reports = prepare_taxibj(config)
    assert set(reports.sequence_id) == {"taxibj_20160101"}
    assert not ((reports.grid_column == 4) & (reports.timeslot == 5)).any()
    assert ((reports.grid_column == 5) & (reports.timeslot == 5)).any()


def test_participation_changes_but_source_identities_and_quorum_persist(tmp_path):
    dates = ["201601%02d%02d" % (day, slot) for day in (1, 2) for slot in range(1, 49)]
    config = _field_file(tmp_path, dates)
    reports = prepare_taxibj(config)
    counts = reports.groupby(["sequence_id", "task_id", "timestamp"]).source_id.nunique()
    assert counts.min() >= 3
    assert counts.nunique() > 1
    identities = reports.groupby("sequence_id").source_id.apply(set).tolist()
    assert identities[0] == identities[1]
    assert len(identities[0]) == 8


def test_task_selection_does_not_change_existing_task_reports(tmp_path):
    config = _field_file(tmp_path, ["20160101%02d" % slot for slot in range(1, 49)])
    full = prepare_taxibj(config)
    subset = prepare_taxibj(dict(config, cells=[[3, 5]]))
    pd.testing.assert_frame_equal(full[full.grid_column == 5].reset_index(drop=True), subset)


def test_reject_prediction_files_and_truth_copies(tmp_path):
    config = _field_file(tmp_path, ["20160101%02d" % slot for slot in range(1, 49)])
    with pytest.raises(ValueError, match="measurement noise"):
        prepare_taxibj(dict(config, normal_report_noise={"relative_std": 0, "absolute_std": 0}))
    with h5py.File(tmp_path / config.get("files", ["BJ15_M32x32_T30_InOut.h5"])[0], "w") as handle:
        handle["pred"] = np.zeros((2, 32, 32))
    with pytest.raises(ValueError, match="original TaxiBJ"):
        prepare_taxibj(config)


def test_campaign_identity_continues_across_day_boundary(tmp_path):
    dates = ["201601%02d%02d" % (day, slot) for day in (1, 2) for slot in range(1, 49)]
    config = _field_file(tmp_path, dates)
    config.pop("sequence_mode")
    reports = prepare_taxibj(config)
    assert reports.sequence_id.nunique() == 1
    assert reports.timestamp.nunique() == 96


def test_extending_future_days_does_not_change_existing_reports(tmp_path):
    dates = ["201601%02d%02d" % (day, slot) for day in (1, 2) for slot in range(1, 49)]
    config = _field_file(tmp_path, dates)
    short = prepare_taxibj(dict(config, max_days=1))
    full = prepare_taxibj(dict(config, max_days=2))
    pd.testing.assert_frame_equal(short, full[full.sequence_id == "taxibj_20160101"].reset_index(drop=True))


def test_compressed_acquisition_extracts_only_manifested_data_and_verifies_hash(tmp_path):
    archive = tmp_path / "source.zip"
    contents = b"verified original data"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("repo/data/TaxiBJ/example.h5", contents)
        handle.writestr("repo/unused.py", b"raise RuntimeError('do not run')")
    manifest = {"dataset": "test", "acquisition_archive": {
        "path": "archive.zip", "url": archive.as_uri(), "size": archive.stat().st_size,
        "checksum": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "member_prefix": "repo/data/TaxiBJ/"}, "files": [{
        "path": "example.h5", "url": "https://invalid.test/not-needed",
        "size": len(contents), "checksum": "md5:" + hashlib.md5(contents).hexdigest()}]}
    destination = tmp_path / "download"
    receipt = download_taxibj(manifest, destination)
    assert (destination / "example.h5").read_bytes() == contents
    assert receipt["files"][0]["sha256"] == hashlib.sha256(contents).hexdigest()
    assert "acquisition_archive" in receipt
    assert not (destination / "unused.py").exists()
    assert "acquisition_archive" in download_taxibj(manifest, destination)
