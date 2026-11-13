"""Projection and exporter tests use stub detections; no weight download."""
import json
import numpy as np
import pytest
import torch

from uav_abnormality.data import perception


def frame(convention="uav3d_carla"):
    # CARLA pitch -90 degrees points the camera body x-axis downward.
    rotation = [2 ** -.5, 0., -2 ** -.5, 0.] if convention == "uav3d_carla" else [0., 1., 0., 0.]
    return {"sample_data_token": "sd0", "sample_token": "s0", "sequence_id": "scene0",
            "source_id": "uav_0", "timestamp": 1., "sensor_timestamp": 1.,
            "channel": "CAMERA_BOTTOM_id_0", "width": 100, "height": 100,
            "camera_intrinsic": [[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],
            "sensor_rotation": rotation, "sensor_translation": [10.,20.,60.],
            "ego_rotation": [1.,0.,0.,0.], "ego_translation": [999.,999.,999.]}


def test_uav3d_carla_optical_axis_and_raw_global_camera_location():
    meta = frame()
    rotation, origin = perception.camera_world_pose(meta)
    np.testing.assert_allclose(rotation @ [0,0,1], [0,0,-1], atol=1e-12)
    np.testing.assert_allclose(origin, [10,20,60])  # Raw ego offset is not applied twice.
    np.testing.assert_allclose(perception.project_pixel_to_ground([50,50], meta, 0), [10,20,0], atol=1e-10)
    np.testing.assert_allclose(perception.project_pixel_to_ground([60,50], meta, 0), [10,26,0], atol=1e-10)
    np.testing.assert_allclose(perception.project_pixel_to_ground([50,60], meta, 0), [4,20,0], atol=1e-10)


def test_carla_rotation_matches_official_euler_matrix():
    roll, pitch, yaw = .2, -.6, .8
    # Standard XYZ quaternion, as stored by UAV3D before its CARLA conversion.
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    q = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
         cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
    meta = frame()
    meta["sensor_rotation"] = q
    actual, _ = perception.camera_world_pose(meta)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    official_body = np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                              [sy*cp, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                              [sp, -cp*sr, cp*cr]])
    body_to_optical = np.array([[0,1,0],[0,0,-1],[1,0,0]])
    np.testing.assert_allclose(actual, official_body @ body_to_optical.T, atol=1e-12)


def test_standard_nuscenes_optical_uses_ego_transform():
    meta = frame("nuscenes_optical")
    meta["sensor_translation"] = [1,0,60]
    meta["ego_translation"] = [10,20,0]
    meta["ego_rotation"] = [2 ** -.5,0,0,2 ** -.5]
    point = perception.project_pixel_to_ground([50,50], meta, 0, camera_convention="nuscenes_optical")
    np.testing.assert_allclose(point, [10,21,0], atol=1e-10)


def test_projection_rejects_upward_parallel_and_excessive_range():
    meta = frame()
    assert perception.project_pixel_to_ground([50,50], meta, 0, max_range_m=10) is None
    assert perception.project_pixel_to_ground([50,50], meta, 100) is None
    meta["sensor_rotation"] = [1,0,0,0]  # Body forward points horizontally.
    assert perception.project_pixel_to_ground([50,50], meta, 0) is None
    meta["sensor_rotation"] = [2 ** -.5,0,2 ** -.5,0]  # Upward body forward.
    assert perception.project_pixel_to_ground([50,50], meta, 0) is None
    meta["camera_intrinsic"] = [[0,0,0],[0,0,0],[0,0,0]]
    with pytest.raises(ValueError, match="intrinsic"):
        perception.project_pixel_to_ground([50,50], meta, 0)


def test_stub_predictions_filter_classes_confidence_and_use_bottom_center():
    prediction = {"boxes": np.array([[40,30,60,50]] * 4),
                  "scores": np.array([.9,.95,.2,.8]), "labels": np.array([3,1,6,8])}
    rows = perception.prediction_reports(frame(), prediction, ground_z=0)
    assert [row["class_name"] for row in rows] == ["car", "truck"]
    np.testing.assert_allclose([[r["x"],r["y"],r["z"]] for r in rows], [[10,20,0],[10,20,0]], atol=1e-10)
    assert all(row["coordinate_frame"] == "global" for row in rows)
    assert all("label" not in row and "task_id" not in row and not any(k.startswith("truth_") for k in row) for row in rows)


def write_metadata(root):
    tables = {"scene": [{"token":"scene", "name":"scene0"}],
              "sample": [{"token":"s0", "scene_token":"scene", "timestamp":1000000}],
              "ego_pose": [{"token":"ego", "rotation":[1,0,0,0], "translation":[999,999,999]}],
              "sensor": [], "calibrated_sensor": [], "sample_data": []}
    for index, (channel, source) in enumerate([("CAMERA_BOTTOM_id_0",0), ("CAMERA_BOTTOM_id_1",1), ("CAMERA_FRONT_id_0",0)]):
        tables["sensor"].append({"token":"sensor%d" % index, "channel":channel, "modality":"camera"})
        tables["calibrated_sensor"].append({"token":"cal%d" % index, "sensor_token":"sensor%d" % index,
                    "rotation":[2**-.5,0,-2**-.5,0], "translation":[10+source,20,60],
                    "camera_intrinsic":[[100,0,50],[0,100,50],[0,0,1]]})
        tables["sample_data"].append({"token":"sd%d" % index, "sample_token":"s0", "ego_pose_token":"ego",
                    "calibrated_sensor_token":"cal%d" % index, "timestamp":1000000, "is_key_frame":True,
                    "filename":"image%d.png" % index,"width":100,"height":100})
        (root / ("image%d.png" % index)).write_bytes(b"stub image loaded by patched tensor helper")
    for name, records in tables.items():
        (root / (name+".json")).write_text(json.dumps(records))
    # Deliberately invalid: exporter must never read this ground-truth file.
    (root / "sample_annotation.json").write_text("not readable as JSON")


def test_exporter_uses_independent_physical_sources_and_never_annotations(tmp_path, monkeypatch):
    write_metadata(tmp_path)
    calls = []
    class Detector:
        def __call__(self, inputs):
            calls.append(len(inputs))
            return [{"boxes":torch.tensor([[40.,30.,60.,50.]]), "scores":torch.tensor([.9]), "labels":torch.tensor([3])}]
    monkeypatch.setattr(perception, "_load_detector", lambda device: (Detector(), None, "cpu", perception.VEHICLE_LABELS, {"stub_for_unit_test":True}))
    monkeypatch.setattr(perception, "_image_tensor", lambda *args: torch.zeros(3,100,100))
    output = tmp_path / "reports.jsonl"
    reports = perception.export_perception_reports({"raw_dir":str(tmp_path), "ground_z":0, "output":str(output)})
    assert calls == [1,1]  # Default bottom channels: one independent view per drone.
    assert reports.source_id.tolist() == ["uav_0","uav_1"]
    np.testing.assert_allclose(reports[["x","y","z"]], [[10,20,0],[11,20,0]], atol=1e-10)
    assert output.exists()
    receipt = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert receipt["processed_camera_frames"] == 2
    assert receipt["ground_truth_read"] is False
    assert "sample_annotation.json" not in receipt["metadata_sha256"]
    assert "label" not in reports and "task_id" not in reports
    from uav_abnormality.data.uav3d import prepare_uav3d
    canonical = prepare_uav3d({"raw_dir": str(tmp_path), "reports_path": str(output),
                              "metadata_coordinate_convention": "uav3d_carla"})
    assert canonical.task_id.nunique() == 1
    assert canonical.source_id.nunique() == 2
    assert (canonical.label == -1).all()
    np.testing.assert_allclose(canonical[["value_0", "value_1", "value_2"]],
                               [[10,20,0],[11,20,0]], atol=1e-10)
    with pytest.raises(FileExistsError):
        perception.export_perception_reports({"raw_dir":str(tmp_path),"ground_z":0,"output":str(output)})


def test_explicit_ground_height_required_before_any_model_load(monkeypatch):
    monkeypatch.setattr(perception, "_load_detector", lambda *_: pytest.fail("Model must not load"))
    with pytest.raises(ValueError, match="ground_z"):
        perception.export_perception_reports({"raw_dir":"missing"})


def test_real_uav3d_image_prefix_resolves_after_local_extraction(tmp_path):
    root = tmp_path / "uav3d"
    image = root / "v1.0-mini" / "sweeps" / "CAMERA_BOTTOM_id_0" / "frame.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"path resolution fixture")
    filename = "./data/uav3d/v1.0-mini/sweeps/CAMERA_BOTTOM_id_0/frame.png"
    assert perception._image_path(root, filename) == image
    assert perception._image_path(root / "v1.0-mini", filename) == image
    assert perception._image_path(root / "v1.0-mini" / "v1.0-mini", filename) == image
