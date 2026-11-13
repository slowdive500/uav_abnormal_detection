"""UAV3D detector-report adapter, with explicit metadata and coordinate handling.

Images/3D annotations are not anomaly observations. A fixed perception frontend
must export one report per physical UAV and observed target. GT-derived reports
are only available through the explicitly named oracle_semisynthetic mode.
"""
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def quaternion_matrix(quaternion):
    """nuScenes quaternion order is w,x,y,z."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Quaternion must contain four finite w,x,y,z components")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _read_json_table(directory, name, required=True):
    path = Path(directory) / (name + ".json")
    if not path.exists():
        if required:
            raise FileNotFoundError("Missing official nuScenes-style metadata: %s" % path)
        return {}
    records = json.loads(path.read_text())
    return {record["token"]: record for record in records}


def _metadata_directory(raw_dir, version=None):
    root = Path(raw_dir)
    versions = [version] if version else ["v1.0-mini", "v1.0-trainval", "v1.0-test"]
    candidates = [] if version else [root]
    for name in versions:
        candidates.extend([root / name, root / name / name, root / "v1.0" / name])
    if version:
        candidates.append(root)
    for directory in candidates:
        if (directory / "sample.json").exists():
            return directory
    raise FileNotFoundError("Cannot find sample.json; raw_dir must point to the official metadata directory or its parent")


def load_uav3d_metadata(raw_dir, version=None, channel_sources=None, coordinate_convention="uav3d_carla", timestamp_unit=None):
    """Return sensor frames indexed independently of annotation/ground truth."""
    if coordinate_convention not in ("uav3d_carla", "nuscenes_optical"):
        raise ValueError("metadata_coordinate_convention must be uav3d_carla or nuscenes_optical")
    timestamp_unit = timestamp_unit or ("ns" if coordinate_convention == "uav3d_carla" else "us")
    if timestamp_unit not in ("s", "ms", "us", "ns"):
        raise ValueError("metadata_timestamp_unit must be s, ms, us, or ns")
    timestamp_divisor = {"s": 1., "ms": 1e3, "us": 1e6, "ns": 1e9}[timestamp_unit]
    directory = _metadata_directory(raw_dir, version)
    scenes = _read_json_table(directory, "scene")
    samples = _read_json_table(directory, "sample")
    data = _read_json_table(directory, "sample_data")
    poses = _read_json_table(directory, "ego_pose")
    calibration = _read_json_table(directory, "calibrated_sensor")
    sensors = _read_json_table(directory, "sensor", required=False)
    records = []
    for token, frame in data.items():
        if not frame.get("is_key_frame", True):
            continue
        sample = samples[frame["sample_token"]]
        calibrated = calibration[frame["calibrated_sensor_token"]]
        sensor = sensors.get(calibrated.get("sensor_token"), {})
        channel = sensor.get("channel", frame.get("channel", calibrated.get("channel", "")))
        source = (channel_sources or {}).get(channel)
        if source is None:
            match = re.search(r"(?:^|_)id_(\d+)(?:_|$)", channel, flags=re.I)
            if match:
                source = "uav_" + match.group(1)
        if source is None:
            raise ValueError("Cannot identify physical UAV from channel %r; supply channel_sources explicitly" % channel)
        pose = poses[frame["ego_pose_token"]]
        scene = scenes[sample["scene_token"]]
        records.append({"sample_data_token": token, "sample_token": frame["sample_token"],
                        "sequence_id": scene.get("name", sample["scene_token"]),
                        "scene_token": sample["scene_token"], "source_id": str(source),
                        "timestamp": float(sample["timestamp"]) / timestamp_divisor,
                        "sensor_timestamp": float(frame.get("timestamp", sample["timestamp"])) / timestamp_divisor,
                        "raw_timestamp": sample["timestamp"],
                        "raw_sensor_timestamp": frame.get("timestamp", sample["timestamp"]),
                        "channel": channel, "filename": frame.get("filename", ""),
                        "width": int(frame.get("width", 0)), "height": int(frame.get("height", 0)),
                        "sensor_rotation": calibrated["rotation"], "sensor_translation": calibrated["translation"],
                        "ego_rotation": pose["rotation"], "ego_translation": pose["translation"],
                        "camera_intrinsic": calibrated.get("camera_intrinsic", []),
                        "coordinate_convention": coordinate_convention, "metadata_timestamp_unit": timestamp_unit})
    if not records:
        raise ValueError("Metadata has no key frames")
    return pd.DataFrame(records)


def export_uav3d_metadata(raw_dir, output_path, version=None, channel_sources=None, coordinate_convention="uav3d_carla", timestamp_unit=None):
    """Create JSONL frame mapping for detector exporters, without opening GT."""
    frames = load_uav3d_metadata(raw_dir, version, channel_sources, coordinate_convention, timestamp_unit)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames.to_json(output, orient="records", lines=True)
    return frames


def _read_reports(path, source_id=None):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        frame = pd.read_json(path, lines=True)
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "results" in payload:
            rows = []
            for sample, boxes in payload["results"].items():
                for box in boxes:
                    row = dict(box)
                    row.setdefault("sample_token", sample)
                    if "translation" in row:
                        row.update(dict(zip(("x", "y", "z"), row["translation"])))
                    rows.append(row)
            frame = pd.DataFrame(rows)
        elif isinstance(payload, list):
            frame = pd.DataFrame(payload)
        else:
            raise ValueError("Expected CSV, JSONL rows, or nuScenes results JSON")
    if source_id is not None:
        if "source_id" in frame and not (frame["source_id"].astype(str) == str(source_id)).all():
            raise ValueError("Configured source_id disagrees with report content")
        frame["source_id"] = str(source_id)
    if "translation" in frame and not {"x", "y", "z"}.issubset(frame):
        frame[["x", "y", "z"]] = pd.DataFrame(frame.translation.tolist(), index=frame.index)
    if "detection_score" in frame and "confidence" not in frame:
        frame["confidence"] = frame.detection_score
    if "detection_name" in frame and "class_name" not in frame:
        frame["class_name"] = frame.detection_name
    return frame


def _attach_metadata(reports, frames):
    by_token = {r["sample_data_token"]: r for r in frames.to_dict("records")}
    by_sample_source = {}
    for record in frames.to_dict("records"):
        by_sample_source.setdefault((record["sample_token"], record["source_id"]), []).append(record)
    result = []
    for row in reports.to_dict("records"):
        token = row.get("sample_data_token")
        if token is not None and pd.notna(token):
            if token not in by_token:
                raise ValueError("Unknown sample_data_token %s" % token)
            meta = by_token[token]
        else:
            if not row.get("sample_token") or not row.get("source_id"):
                raise ValueError("Reports require sample_data_token or sample_token + physical source_id")
            choices = by_sample_source.get((row["sample_token"], str(row["source_id"])), [])
            if row.get("channel"):
                choices = [m for m in choices if m["channel"] == row["channel"]]
            if not choices:
                raise ValueError("No sensor frame for report sample/source/channel")
            if len(choices) > 1 and row.get("coordinate_frame", "global") in ("sensor", "sensor_optical"):
                raise ValueError("Sensor-frame reports must specify sample_data_token or channel")
            meta = choices[0]  # Global/ego values do not depend on camera calibration.
        if row.get("source_id") is not None and str(row["source_id"]) != meta["source_id"]:
            raise ValueError("Report source_id disagrees with physical UAV metadata")
        merged = dict(row)
        merged.update({k: meta[k] for k in ("sequence_id", "source_id", "timestamp", "sample_token")})
        merged["_meta"] = meta
        result.append(merged)
    return result


def _camera_global_transform(meta):
    """Return optical-camera -> world transform for the declared convention.

    UAV3D raw calibrated_sensor.translation is already world-referenced. Its
    official visualization uses CARLA body axes; optical=(body_y,-body_z,body_x).
    Standard nuScenes metadata instead composes calibrated sensor and ego pose.
    """
    convention = meta.get("coordinate_convention", "uav3d_carla")
    if convention == "uav3d_carla":
        handedness = np.diag([1., 1., -1.])
        carla_rotation = handedness @ quaternion_matrix(meta["sensor_rotation"]) @ handedness
        body_to_optical = np.array([[0., 1., 0.], [0., 0., -1.], [1., 0., 0.]])
        return carla_rotation @ body_to_optical.T, np.asarray(meta["sensor_translation"], dtype=float)
    if convention == "nuscenes_optical":
        ego_rotation = quaternion_matrix(meta["ego_rotation"])
        return (ego_rotation @ quaternion_matrix(meta["sensor_rotation"]),
                ego_rotation @ np.asarray(meta["sensor_translation"]) + np.asarray(meta["ego_translation"]))
    raise ValueError("Unknown metadata coordinate convention")


def _to_global(position, meta, coordinate_frame):
    position = np.asarray(position, dtype=float)
    if coordinate_frame == "global":
        return position
    if meta is None:
        raise ValueError("Local-coordinate reports require official calibration/pose metadata")
    if coordinate_frame in ("sensor", "sensor_optical"):
        rotation, translation = _camera_global_transform(meta)
        return rotation @ position + translation
    if coordinate_frame == "ego":
        if meta.get("coordinate_convention", "uav3d_carla") != "nuscenes_optical":
            raise ValueError("Raw UAV3D has nonstandard ego metadata; export global coordinates or explicitly normalize to nuscenes_optical")
        return quaternion_matrix(meta["ego_rotation"]) @ position + np.asarray(meta["ego_translation"])
    raise ValueError("coordinate_frame must be global, ego, or sensor (optical xyz)")


def associate_reports(reports, distance_gate_m=5.0, max_gap_seconds=5.0):
    """Deterministic geometric association without annotations.

    At each time step, Hungarian assignments associate each physical UAV's
    detections with constant-velocity tracks. One track gets at most one report
    per UAV. This is a preprocessing assumption, not a perception benchmark claim.
    """
    if distance_gate_m <= 0 or max_gap_seconds <= 0:
        raise ValueError("Association gates must be positive")
    table = reports.copy()
    table["task_id"] = ""
    for scene, group in table.groupby("sequence_id", sort=True):
        tracks, next_id = {}, 0
        for timestamp, current in group.groupby("timestamp", sort=True):
            timestamp = float(timestamp)
            tracks = {key: value for key, value in tracks.items() if timestamp - value["time"] <= max_gap_seconds}
            observations = {}
            for source, agent_rows in current.groupby("source_id", sort=True):
                indexes = list(agent_rows.index)
                candidates = list(tracks)
                points = agent_rows[["value_0", "value_1", "value_2"]].to_numpy(dtype=float)
                assigned = set()
                if candidates:
                    predicted = np.array([tracks[k]["position"] + tracks[k]["velocity"] * (timestamp - tracks[k]["time"]) for k in candidates])
                    cost = np.linalg.norm(points[:, None, :] - predicted[None, :, :], axis=-1)
                    for i, idx in enumerate(indexes):
                        cls = table.at[idx, "class_name"]
                        for j, key in enumerate(candidates):
                            if cls != tracks[key]["class_name"]:
                                cost[i, j] = 1e12
                    row_indices, track_indices = linear_sum_assignment(cost)
                    for i, j in zip(row_indices, track_indices):
                        if cost[i, j] > distance_gate_m:
                            continue
                        key = candidates[j]
                        idx = indexes[i]
                        table.at[idx, "task_id"] = key
                        observations.setdefault(key, []).append(points[i])
                        assigned.add(i)
                for i, idx in enumerate(indexes):
                    if i in assigned:
                        continue
                    key = "track_%06d" % next_id
                    next_id += 1
                    tracks[key] = {"position": points[i], "velocity": np.zeros(3), "time": timestamp,
                                   "class_name": table.at[idx, "class_name"]}
                    observations.setdefault(key, []).append(points[i])
                    table.at[idx, "task_id"] = key
            for key, points in observations.items():
                position = np.mean(points, axis=0)
                elapsed = timestamp - tracks[key]["time"]
                velocity = ((position - tracks[key]["position"]) / elapsed if elapsed > 0 else tracks[key]["velocity"])
                tracks[key].update(position=position, velocity=velocity, time=timestamp)
    return table


def _oracle_reports(config, frames):
    if not config.get("acknowledge_oracle", False):
        raise ValueError("oracle_semisynthetic requires acknowledge_oracle=true; GT is not observed detector data")
    warnings.warn("Oracle semisynthetic UAV3D observations are generated from annotations and are not detector outputs", RuntimeWarning)
    directory = _metadata_directory(config["raw_dir"], config.get("version"))
    annotations = _read_json_table(directory, "sample_annotation")
    samples = {}
    for annotation in annotations.values():
        samples.setdefault(annotation["sample_token"], []).append(annotation)
    rng = np.random.default_rng(int(config.get("seed", 0)))
    sigma = float(config.get("observation_noise_std_m", 0.5))
    if sigma <= 0:
        raise ValueError("oracle_semisynthetic observation_noise_std_m must be positive")
    rows = []
    seen = set()
    for meta in frames.to_dict("records"):
        intrinsic = np.asarray(meta["camera_intrinsic"], dtype=float)
        if intrinsic.shape != (3, 3) or meta["width"] <= 0 or meta["height"] <= 0:
            continue
        for annotation in samples.get(meta["sample_token"], []):
            key = (meta["sample_token"], meta["source_id"], annotation["instance_token"])
            if key in seen:
                continue
            truth = np.asarray(annotation["translation"], dtype=float)
            rotation, translation = _camera_global_transform(meta)
            sensor = rotation.T @ (truth - translation)
            if sensor[2] <= 0 or np.linalg.norm(sensor) > float(config.get("max_range_m", 200)):
                continue
            pixel = intrinsic @ sensor
            pixel = pixel[:2] / pixel[2]
            if not (0 <= pixel[0] < meta["width"] and 0 <= pixel[1] < meta["height"]):
                continue
            observed = truth + rng.normal(0, sigma, 3)
            row = {"sample_data_token": meta["sample_data_token"], "source_id": meta["source_id"],
                   "task_id": annotation["instance_token"], "x": observed[0], "y": observed[1], "z": observed[2],
                   "label": 0, "coordinate_frame": "global", "class_name": "vehicle"}
            row.update({"truth_%d" % i: truth[i] for i in range(3)})
            rows.append(row)
            seen.add(key)
    if not rows:
        raise ValueError("Oracle visibility projection produced no reports; inspect UAV3D camera coordinate conventions")
    return pd.DataFrame(rows)


def prepare_uav3d(config):
    """Load reports and optional official metadata into the canonical long table.

    Real mode supports detector CSV/JSONL or one nuScenes results JSON per UAV.
    Inputs with no task_id use geometric association, never GT instance IDs.
    """
    mode = config.get("mode", "detector_reports")
    if mode not in ("detector_reports", "oracle_semisynthetic"):
        raise ValueError("UAV3D mode must be detector_reports or oracle_semisynthetic")
    frames = (load_uav3d_metadata(config["raw_dir"], config.get("version"), config.get("channel_sources"),
                                     config.get("metadata_coordinate_convention", "uav3d_carla"), config.get("metadata_timestamp_unit"))
              if config.get("raw_dir") else None)
    if mode == "oracle_semisynthetic":
        if frames is None:
            raise ValueError("Oracle mode requires raw_dir metadata")
        reports = _oracle_reports(config, frames)
    else:
        entries = config.get("report_files") or ([config["reports_path"]] if config.get("reports_path") else [])
        if not entries:
            raise ValueError("UAV3D requires reports_path/report_files exported by a fixed detector; annotations alone are not observations")
        parts = []
        for entry in entries:
            if isinstance(entry, str):
                entry = {"path": entry}
            parts.append(_read_reports(entry["path"], entry.get("source_id")))
        reports = pd.concat(parts, ignore_index=True)
        if reports.empty:
            raise ValueError("Detector exports contain no observations")
    reports = reports.rename(columns=config.get("column_map", {}))
    if "coordinate_frame" not in reports:
        reports["coordinate_frame"] = config.get("coordinate_frame", "global")
    if frames is not None:
        records = _attach_metadata(reports, frames)
    else:
        required = {"timestamp", "sequence_id", "source_id"}
        if not required.issubset(reports):
            raise ValueError("Without raw metadata, reports require timestamp, sequence_id, source_id")
        records = reports.to_dict("records")
    result = []
    for record in records:
        if not all(k in record for k in ("x", "y", "z")):
            raise ValueError("Each report must contain observed center x,y,z (or translation)")
        meta = record.get("_meta")
        position = _to_global([record["x"], record["y"], record["z"]], meta,
                              record.get("coordinate_frame", config.get("coordinate_frame", "global")))
        if not np.isfinite(position).all():
            continue
        timestamp = record["timestamp"]
        if not isinstance(timestamp, (int, float, np.number)):
            timestamp = pd.Timestamp(timestamp).timestamp()
        elif frames is None:
            units = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}
            timestamp = float(timestamp) / units[config.get("timestamp_unit", "s")]
        row = {"timestamp": float(timestamp), "sequence_id": str(record["sequence_id"]),
               "source_id": str(record["source_id"]), "label": int(record.get("label", -1)),
               "class_name": record.get("class_name", "vehicle"),
               "dataset": "uav3d", "observation_kind": mode,
               "position_semantics": str(record.get("position_semantics", "object_center_3d"))}
        for provenance_key in ("frontend", "detector_name", "camera_convention", "observation_origin"):
            if provenance_key in record and pd.notna(record[provenance_key]):
                row[provenance_key] = record[provenance_key]
        for i in range(3):
            row["value_%d" % i] = float(position[i])
            if "truth_%d" % i in record:
                row["truth_%d" % i] = float(record["truth_%d" % i])
        if record.get("task_id") is not None and pd.notna(record.get("task_id")):
            row["task_id"] = str(record["task_id"])
        if "confidence" in record:
            row["condition_confidence"] = float(record["confidence"])
        if meta is not None:
            sensor_global = _to_global([0., 0., 0.], meta, "sensor")
            row["condition_range_m"] = float(np.linalg.norm(position - sensor_global))
            row["condition_uav_altitude"] = float(sensor_global[2])
            for axis, value in zip(("x", "y", "z"), sensor_global):
                row["condition_uav_" + axis] = float(value)
        result.append(row)
    table = pd.DataFrame(result)
    if table.empty:
        raise ValueError("No finite detector reports")
    if table.position_semantics.nunique() != 1:
        raise ValueError("Mixed position_semantics: 3D centers and projected ground contact points require separate experiments")
    if not set(table.label.unique()).issubset({-1, 0, 1}):
        raise ValueError("Labels must be -1 unknown, 0 normal, or 1 anomalous")
    if "task_id" not in table or table.task_id.isna().any():
        if "task_id" in table and table.task_id.notna().any():
            raise ValueError("Supply task_id for all reports or none")
        table = associate_reports(table, float(config.get("association_gate_m", 5)), float(config.get("association_max_gap_seconds", 5)))
        table["task_assignment"] = "geometric_association_without_gt"
    else:
        table["task_assignment"] = "oracle_instance" if mode == "oracle_semisynthetic" else "provided_association"
    # Several cameras on one drone are one physical contributor, not extra UAVs.
    keys = ["sequence_id", "task_id", "timestamp", "source_id"]
    if table.duplicated(keys).any():
        if config.get("duplicate_policy", "error") == "highest_confidence" and "condition_confidence" in table:
            table = table.sort_values("condition_confidence", ascending=False).drop_duplicates(keys)
        else:
            raise ValueError("Duplicate target/time/physical-UAV reports; fuse cameras upstream or choose highest_confidence")
    counts = table.groupby(["sequence_id", "task_id", "timestamp"])["source_id"].nunique()
    table.attrs["source_overlap"] = {str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()}
    table.attrs["ground_truth_used_to_generate_observations"] = mode == "oracle_semisynthetic"
    table.attrs["position_semantics"] = table.position_semantics.iloc[0]
    table.attrs["metadata_coordinate_convention"] = config.get("metadata_coordinate_convention", "uav3d_carla")
    for provenance_key in ("frontend", "detector_name", "camera_convention", "observation_origin"):
        if provenance_key in table:
            table.attrs[provenance_key] = sorted(str(value) for value in table[provenance_key].dropna().unique())
    if table.source_id.nunique() < int(config.get("min_physical_sources", 1)):
        raise ValueError("Not enough physical UAVs; fused central predictions cannot be copied into multiple sources")
    return table.sort_values(keys).reset_index(drop=True)
