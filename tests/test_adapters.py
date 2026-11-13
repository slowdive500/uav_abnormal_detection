import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from uav_abnormality.data.download import safe_extract, verify_file, file_digest
from uav_abnormality.data.lapse_rate import prepare_lapse_rate
from uav_abnormality.data.uav3d import (prepare_uav3d, quaternion_matrix,
                                       load_uav3d_metadata, associate_reports)


def _nc(path, lat=38.01, asl=2400., agl=20., second=None):
    values = {"Year": [2018., 2018.], "Month": [7., 7.], "Day": [19., 19.],
              "Hour": [12., 12.], "Minute": [0., 0.], "Sec": second or [0., 1.],
              "Lat": [lat, lat], "Long": [-106., -106.], "Alt": [asl, asl], "z": [agl, agl],
              "u": [1., -9999.9], "v": [2., 4.]}
    xr.Dataset({name: ("time", value) for name, value in values.items()}).to_netcdf(path, engine="scipy")


def test_lapse_sources_missing_values_and_absolute_height(tmp_path):
    _nc(tmp_path / "UKY.SOLOW.b1.20180719.120000.cdf", asl=2400, agl=20)
    _nc(tmp_path / "UKY.S1000.b1.20180719.120000.cdf", asl=2400, agl=120)
    table = prepare_lapse_rate({"raw_dir": str(tmp_path), "time_bin_seconds": 5})
    assert len(table) == 2
    assert set(table.source_id) == {"SOLOW", "S1000"}
    assert table.task_id.nunique() == 1  # ASL, not different per-takeoff AGL.
    assert table.value_0.tolist() == [1., 1.]
    assert table.value_1.tolist() == [3., 3.]
    assert (table.label == -1).all()
    assert table.attrs["source_overlap"] == {"2": 1}


def test_lapse_distinct_flights_do_not_become_sources_and_seconds_roll(tmp_path):
    _nc(tmp_path / "UKY.SOLOW.b1.20180719.120000.cdf", second=[59.9, 60.])
    _nc(tmp_path / "UKY.SOLOW.b1.20180719.120100.cdf", second=[59.9, 60.])
    with pytest.warns(RuntimeWarning, match="no colocated"):
        table = prepare_lapse_rate({"raw_dir": str(tmp_path), "time_bin_seconds": 1})
    assert table.source_id.nunique() == 1
    assert len(table) == 2
    assert table.timestamp.max() == pd.Timestamp("2018-07-19T12:01:00Z")


def _metadata(root):
    tables = {
        "scene": [{"token": "scene", "name": "scene_0"}],
        "sample": [{"token": "sample", "scene_token": "scene", "timestamp": 2000000}],
        "sensor": [{"token": "sensor0", "channel": "CAMERA_BOTTOM_id_0"},
                   {"token": "sensor1", "channel": "CAMERA_BOTTOM_id_1"}],
        "calibrated_sensor": [{"token": "cal0", "sensor_token": "sensor0", "rotation": [1,0,0,0], "translation": [1,0,0]},
                              {"token": "cal1", "sensor_token": "sensor1", "rotation": [1,0,0,0], "translation": [1,0,0]}],
        "ego_pose": [{"token": "pose", "rotation": [2**-.5,0,0,2**-.5], "translation": [10,20,30]}],
        "sample_data": [{"token": "sd0", "sample_token": "sample", "ego_pose_token": "pose", "calibrated_sensor_token": "cal0", "timestamp": 2000000},
                        {"token": "sd1", "sample_token": "sample", "ego_pose_token": "pose", "calibrated_sensor_token": "cal1", "timestamp": 2000000}],
    }
    for name, records in tables.items():
        (root / (name + ".json")).write_text(json.dumps(records))


def test_uav3d_transforms_real_observations_without_gt(tmp_path):
    _metadata(tmp_path)
    reports = tmp_path / "reports.jsonl"
    reports.write_text(json.dumps({"sample_data_token": "sd0", "x": 1, "y": 0, "z": 0,
                                   "coordinate_frame": "sensor", "task_id": "associated_a"}) + "\n")
    frame = prepare_uav3d({"raw_dir": str(tmp_path), "reports_path": str(reports),
                            "metadata_coordinate_convention": "nuscenes_optical"})
    assert frame.source_id.tolist() == ["uav_0"]
    assert np.allclose(frame[["value_0", "value_1", "value_2"]], [[10,22,30]])
    assert frame.label.tolist() == [-1]
    assert not any(col.startswith("truth_") for col in frame)
    assert frame.timestamp.tolist() == [2.]
    assert not frame.attrs["ground_truth_used_to_generate_observations"]


def test_no_annotation_mode_without_explicit_acknowledgement(tmp_path):
    _metadata(tmp_path)
    with pytest.raises(ValueError, match="acknowledge_oracle"):
        prepare_uav3d({"raw_dir": str(tmp_path), "mode": "oracle_semisynthetic"})


def test_central_predictions_cannot_fake_multi_uav(tmp_path):
    path = tmp_path / "reports.csv"
    pd.DataFrame([{"timestamp": 0, "sequence_id": "s", "source_id": "uav_0", "task_id": "a", "x": 0, "y": 0, "z": 0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="physical UAVs"):
        prepare_uav3d({"reports_path": str(path), "min_physical_sources": 2})


def test_geometric_association_preserves_physical_source_limit():
    rows = []
    for t in range(3):
        for source, bias in [("uav_0", 0), ("uav_1", .1)]:
            for location in [0, 20]:
                rows.append({"timestamp": t, "sequence_id": "s", "source_id": source,
                             "value_0": location+t+bias, "value_1": 0., "value_2": 0., "class_name": "vehicle"})
    table = associate_reports(pd.DataFrame(rows), distance_gate_m=3)
    assert table.task_id.nunique() == 2
    assert not table.duplicated(["timestamp", "source_id", "task_id"]).any()
    assert (table.groupby(["timestamp", "task_id"]).source_id.nunique() == 2).all()


def test_safe_extract_rejects_traversal_before_writing(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ok.txt", "good")
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "output/ok.txt").exists()


def test_checksum_detects_same_size_corruption(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"1234")
    entry = {"size": 4, "checksum": "sha256:" + file_digest(path)}
    assert verify_file(path, entry)
    path.write_bytes(b"4321")
    assert not verify_file(path, entry)


def test_raw_uav3d_global_camera_translation_and_optical_axis(tmp_path):
    _metadata(tmp_path)
    reports = tmp_path / "reports.jsonl"
    reports.write_text(json.dumps({"sample_data_token": "sd0", "x": 0, "y": 0, "z": 2,
                                   "coordinate_frame": "sensor", "task_id": "associated_a"}) + "\n")
    # Raw UAV3D calibration translation is world position. Ego [10,20,30]
    # must not be added. Optical forward +z corresponds to CARLA body +x.
    frame = prepare_uav3d({"raw_dir": str(tmp_path), "reports_path": str(reports)})
    assert np.allclose(frame[["value_0", "value_1", "value_2"]], [[3, 0, 0]])
    assert np.allclose(frame[["condition_uav_x", "condition_uav_y", "condition_uav_z"]], [[1, 0, 0]])
    assert np.allclose(frame.condition_range_m, 2)


def test_frontend_semantics_preserved_and_mixing_rejected(tmp_path):
    path = tmp_path / "reports.jsonl"
    base = {"timestamp": 0, "sequence_id": "s", "source_id": "uav_0", "task_id": "a", "x": 0, "y": 0, "z": 0,
            "position_semantics": "bbox_bottom_center_on_ground_plane", "frontend": "frozen_detector"}
    path.write_text(json.dumps(base) + "\n")
    frame = prepare_uav3d({"reports_path": str(path)})
    assert frame.position_semantics.tolist() == ["bbox_bottom_center_on_ground_plane"]
    assert frame.attrs["position_semantics"] == "bbox_bottom_center_on_ground_plane"
    assert frame.frontend.tolist() == ["frozen_detector"]
    other = dict(base, source_id="uav_1", position_semantics="object_center_3d")
    path.write_text(json.dumps(base) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="Mixed position_semantics"):
        prepare_uav3d({"reports_path": str(path)})


def test_raw_uav3d_nanosecond_time(tmp_path):
    _metadata(tmp_path)
    path = tmp_path / "sample.json"
    payload = json.loads(path.read_text())
    payload[0]["timestamp"] = 1701664619345580032
    path.write_text(json.dumps(payload))
    frames = load_uav3d_metadata(tmp_path)
    assert np.allclose(frames.timestamp, 1701664619.345580032)
    assert frames.metadata_timestamp_unit.tolist() == ["ns", "ns"]


def test_remote_metadata_reader_budget_and_ignored_range(tmp_path, monkeypatch):
    import importlib.util
    script = Path(__file__).parents[1] / "scripts/fetch_uav3d_metadata.py"
    spec = importlib.util.spec_from_file_location("metadata_fetch_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.Remote("https://example.invalid/archive.zip", 7000000000, budget=1024)
    with pytest.raises(RuntimeError, match="byte cap"):
        reader.read(32)

    class IgnoredRange:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, count):
            raise AssertionError("Must refuse ignored range before reading a multi-GB response")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *args, **kwargs: IgnoredRange())
    reader = module.Remote("https://example.invalid/archive.zip", 7000000000, budget=30000000)
    with pytest.raises(ValueError, match="ignored or changed HTTP Range"):
        reader.read(32)


def test_official_double_nested_metadata_layout(tmp_path):
    directory = tmp_path / "v1.0-mini/v1.0-mini"
    directory.mkdir(parents=True)
    _metadata(directory)
    frame = load_uav3d_metadata(tmp_path)
    assert len(frame) == 2
    assert frame.raw_timestamp.tolist() == [2000000, 2000000]
