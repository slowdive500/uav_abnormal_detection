"""Optional UAV3D RGB -> planar vehicle-location reports, without annotations.

This is a fixed COCO SSDLite frontend, not the official UAV3D 3D detector.
The geometric approximation and raw CARLA camera convention are explicit.
Torchvision and Pillow are imported only when actual image inference is used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .uav3d import load_uav3d_metadata, quaternion_matrix


VEHICLE_LABELS = {3: "car", 6: "bus", 8: "truck"}
REPORT_COLUMNS = [
    "sample_data_token", "sample_token", "sequence_id", "source_id",
    "timestamp", "sensor_timestamp", "channel", "x", "y", "z",
    "confidence", "class_name", "coordinate_frame", "position_semantics",
    "frontend", "camera_convention", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]
FRONTEND_NAME = "torchvision_ssdlite320_mobilenet_v3_large_COCO_V1"


def camera_world_pose(meta, camera_convention="uav3d_carla"):
    """Return camera optical-to-global rotation and camera global origin.

    UAV3D's official visualizer treats raw calibrated_sensor.translation as
    global position and q as CARLA Euler orientation. Its body axes are
    x-forward/y-right/z-up, then optical coordinates are (y,-z,x).
    Converted, standard nuScenes optical metadata uses a different chain.
    """
    sensor_translation = np.asarray(meta["sensor_translation"], dtype=float)
    if sensor_translation.shape != (3,) or not np.isfinite(sensor_translation).all():
        raise ValueError("Camera translation must contain three finite values")
    sensor_rotation = quaternion_matrix(meta["sensor_rotation"])
    if camera_convention == "uav3d_carla":
        reflection = np.diag([1., 1., -1.])
        carla_rotation = reflection @ sensor_rotation @ reflection
        body_to_optical = np.array([[0., 1., 0.], [0., 0., -1.], [1., 0., 0.]])
        return carla_rotation @ body_to_optical.T, sensor_translation
    if camera_convention == "nuscenes_optical":
        ego_rotation = quaternion_matrix(meta["ego_rotation"])
        ego_translation = np.asarray(meta["ego_translation"], dtype=float)
        if ego_translation.shape != (3,) or not np.isfinite(ego_translation).all():
            raise ValueError("Ego translation must contain three finite values")
        return ego_rotation @ sensor_rotation, ego_rotation @ sensor_translation + ego_translation
    raise ValueError("camera_convention must be uav3d_carla or nuscenes_optical")


def project_pixel_to_ground(pixel, meta, ground_z, max_range_m=250.,
                            min_downward_cosine=0.05,
                            camera_convention="uav3d_carla"):
    """Intersect an image ray with an explicitly configured horizontal plane.

    Returns xyz or None for invalid/upward/nearly-parallel/out-of-range rays.
    Invalid calibration/configuration raises instead of silently moving objects.
    Global z is upward; the camera must be above the requested ground plane.
    """
    if not np.isfinite(ground_z):
        raise ValueError("ground_z must be an explicitly supplied finite height")
    if not np.isfinite(max_range_m) or max_range_m <= 0:
        raise ValueError("max_range_m must be finite and positive")
    if not 0 < min_downward_cosine <= 1:
        raise ValueError("min_downward_cosine must be in (0,1]")
    pixel = np.asarray(pixel, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        return None
    intrinsic = np.asarray(meta["camera_intrinsic"], dtype=float)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Camera intrinsic must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or abs(np.linalg.det(intrinsic)) < 1e-12:
        raise ValueError("Camera intrinsic must be invertible with positive focal lengths")
    rotation, origin = camera_world_pose(meta, camera_convention)
    if origin[2] <= ground_z:
        return None
    camera_ray = np.linalg.solve(intrinsic, np.r_[pixel, 1.])
    ray = rotation @ camera_ray
    norm = np.linalg.norm(ray)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    ray /= norm
    if ray[2] >= -min_downward_cosine:
        return None
    distance = (float(ground_z) - origin[2]) / ray[2]
    if not np.isfinite(distance) or distance <= 0 or distance > max_range_m:
        return None
    point = origin + distance * ray
    point[2] = float(ground_z)
    return point


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def prediction_reports(meta, prediction, ground_z, score_threshold=0.5,
                       max_range_m=250., min_downward_cosine=0.05,
                       camera_convention="uav3d_carla", label_names=None):
    """Turn one camera's independent detector outputs into report dictionaries."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must lie in [0,1]")
    label_names = VEHICLE_LABELS if label_names is None else label_names
    boxes, scores, labels = (_numpy(prediction[key]) for key in ("boxes", "scores", "labels"))
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or labels.shape != (len(boxes),):
        raise ValueError("Detector output must contain boxes[N,4], scores[N], labels[N]")
    width, height = int(meta["width"]), int(meta["height"])
    if min(width, height) <= 0:
        raise ValueError("Projection requires positive image dimensions")
    rows = []
    for box, confidence, label in zip(boxes, scores, labels):
        if not np.isfinite(confidence) or confidence < score_threshold or confidence > 1:
            continue
        if not np.isfinite(label) or int(label) not in label_names or not np.isfinite(box).all():
            continue
        box = np.asarray(box, dtype=float).copy()
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        # Bounding-box bottom center approximates a vehicle ground contact.
        pixel = [(box[0] + box[2]) / 2., box[3]]
        point = project_pixel_to_ground(pixel, meta, ground_z, max_range_m,
                                        min_downward_cosine, camera_convention)
        if point is None:
            continue
        row = {key: meta[key] for key in (
            "sample_data_token", "sample_token", "sequence_id", "source_id",
            "timestamp", "sensor_timestamp", "channel",
        )}
        row.update({"x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
                    "confidence": float(confidence), "class_name": label_names[int(label)],
                    "coordinate_frame": "global", "position_semantics": "bbox_bottom_center_on_ground_plane",
                    "frontend": FRONTEND_NAME, "camera_convention": camera_convention})
        row.update(dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), box.tolist())))
        rows.append(row)
    return rows


def _load_detector(device):
    """Lazy optional dependencies; downloads fixed COCO_V1 weights if uncached."""
    try:
        import torch
        import torchvision
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large,
        )
    except (ImportError, RuntimeError) as error:
        raise RuntimeError("RGB export needs compatible torchvision and Pillow; for torch 2.4.1 install torchvision 0.19.1") from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    detector = ssdlite320_mobilenet_v3_large(weights=weights, progress=True).to(device).eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    categories = weights.meta["categories"]
    names = {index: name for index, name in enumerate(categories) if name in {"car", "bus", "truck"}}
    info = {"torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
            "weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1", "weights_url": weights.url,
            "device": str(device), "training": False, "label_names": names}
    return detector, weights.transforms(), str(device), names, info


def _image_tensor(path, transform, expected_width, expected_height):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("RGB export requires Pillow") from error
    with Image.open(path) as image:
        if image.size != (expected_width, expected_height):
            raise ValueError("Image dimensions disagree with calibration metadata: %s (%r vs %r)" % (
                path, image.size, (expected_width, expected_height)))
        return transform(image.convert("RGB"))


def _image_path(raw_dir, filename, image_root=None):
    if not filename:
        raise ValueError("Camera frame has no image filename")
    filename = Path(filename)
    if filename.is_absolute():
        candidates = [filename]
    else:
        roots = [Path(image_root)] if image_root else [Path(raw_dir), Path(raw_dir).parent]
        candidates = [root / filename for root in roots]
        # Raw UAV3D stores author-local prefixes such as
        # ./data/uav3d/v1.0-mini/sweeps/... inside its ZIP metadata.
        # Resolve the retained version/sweeps suffix against the local extract.
        parts = filename.parts
        for marker in ("v1.0-mini", "v1.0-trainval", "v1.0-test", "sweeps", "samples"):
            if marker in parts:
                suffix = Path(*parts[parts.index(marker):])
                candidates.extend(root / suffix for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Missing camera image %s; extract UAV3D images or supply image_root" % filename)


def _metadata_hashes(raw_dir, version):
    from .uav3d import _metadata_directory
    directory = _metadata_directory(raw_dir, version)
    result = {}
    for name in ("scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor"):
        path = directory / (name + ".json")
        if path.exists():
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def export_perception_reports(config):
    """Run fixed per-camera SSDLite and export a CSV/JSONL report table.

    Required: raw_dir, ground_z. Optional: output, version, image_root, device,
    channels (default: CAMERA_BOTTOM_id_*), max_frames, score_threshold, max_range_m, camera_convention,
    min_downward_cosine, channel_sources. No annotation file is opened.
    """
    if "raw_dir" not in config or "ground_z" not in config:
        raise ValueError("Perception export requires raw_dir and explicit ground_z")
    ground_z = float(config["ground_z"])
    if not np.isfinite(ground_z):
        raise ValueError("ground_z must be finite")
    score_threshold = float(config.get("score_threshold", 0.5))
    max_range = float(config.get("max_range_m", 250.))
    min_cosine = float(config.get("min_downward_cosine", 0.05))
    convention = config.get("camera_convention", "uav3d_carla")
    if not 0 <= score_threshold <= 1 or max_range <= 0 or not np.isfinite(max_range):
        raise ValueError("Invalid score threshold or maximum range")
    if not 0 < min_cosine <= 1 or convention not in ("uav3d_carla", "nuscenes_optical"):
        raise ValueError("Invalid minimum ray angle or camera convention")
    frames = load_uav3d_metadata(config["raw_dir"], config.get("version"), config.get("channel_sources"),
                                convention, config.get("metadata_timestamp_unit"))
    camera_frames = frames[frames.camera_intrinsic.map(lambda value: np.asarray(value).shape == (3, 3))].copy()
    if config.get("channels"):
        camera_frames = camera_frames[camera_frames.channel.isin(config["channels"])]
    else:
        # The five cameras of one UAV are one contributor. Keep one downward
        # camera per UAV unless the caller supplies a separate fusion policy.
        camera_frames = camera_frames[camera_frames.channel.str.match(r"(?i)^CAMERA_BOTTOM_id_\d+$")]
    camera_frames = camera_frames.sort_values(["sequence_id", "timestamp", "source_id", "channel", "sample_data_token"])
    available_frames = len(camera_frames)
    if config.get("max_frames") is not None:
        if int(config["max_frames"]) < 1:
            raise ValueError("max_frames must be positive")
        camera_frames = camera_frames.head(int(config["max_frames"]))
    if camera_frames.empty:
        raise ValueError("No selected camera frames with valid intrinsics; default channels are CAMERA_BOTTOM_id_*. Supply channels explicitly for other metadata")
    output = Path(config["output"]) if config.get("output") else None
    if output and output.suffix.lower() not in (".jsonl", ".ndjson", ".csv"):
        raise ValueError("Perception output must end in .jsonl, .ndjson or .csv")
    if output and output.exists() and not config.get("overwrite", False):
        raise FileExistsError("Output already exists: %s" % output)
    # Validate all selected image paths before downloading optional weights.
    records = camera_frames.to_dict("records")
    paths = [_image_path(config["raw_dir"], frame["filename"], config.get("image_root")) for frame in records]
    import torch
    # Project-local cache makes the delivered weights usable without a second
    # download and permits explicit relocation on other machines.
    torch.hub.set_dir(config.get("weights_cache", "data/weights/torch"))
    detector, transform, device, label_names, detector_info = _load_detector(config.get("device", "auto"))
    reports, detection_count, empty_frames = [], 0, 0
    with torch.inference_mode():
        for meta, path in zip(records, paths):
            tensor = _image_tensor(path, transform, int(meta["width"]), int(meta["height"]))
            predictions = detector([tensor.to(device)])
            if len(predictions) != 1:
                raise ValueError("Expected one prediction result per camera frame")
            detection_count += len(predictions[0]["boxes"])
            rows = prediction_reports(meta, predictions[0], ground_z, score_threshold,
                                      max_range, min_cosine, convention, label_names)
            if not rows:
                empty_frames += 1
            reports.extend(rows)
    result = pd.DataFrame(reports, columns=REPORT_COLUMNS)
    receipt = {"frontend": FRONTEND_NAME, "detector": detector_info,
               "weights_cache": config.get("weights_cache", "data/weights/torch"),
               "raw_dir": str(config["raw_dir"]), "ground_z": ground_z,
               "camera_convention": convention, "score_threshold": score_threshold,
               "selected_channels": sorted(camera_frames.channel.unique().tolist()),
               "max_range_m": max_range, "min_downward_cosine": min_cosine,
               "available_camera_frames": available_frames, "processed_camera_frames": len(records),
               "frames_without_accepted_reports": empty_frames, "raw_detections": detection_count,
               "accepted_reports": len(result), "source_ids": sorted(result.source_id.unique().tolist()),
               "physical_source_ids_in_frames": sorted(camera_frames.source_id.unique().tolist()),
               "metadata_sha256": _metadata_hashes(config["raw_dir"], config.get("version")),
               "ground_truth_read": False, "fine_tuned": False,
               "position_semantics": "bbox_bottom_center_on_ground_plane",
               "task": "2.5D planar vehicle localization, not UAV3D 3D detection benchmark"}
    result.attrs["export_receipt"] = receipt
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".csv":
            result.to_csv(output, index=False)
        else:
            result.to_json(output, orient="records", lines=True, double_precision=12)
        receipt["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix(output.suffix + ".manifest.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-z", required=True, type=float, help="Known ground plane global z in metres")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--version")
    parser.add_argument("--image-root")
    parser.add_argument("--weights-cache", default="data/weights/torch")
    parser.add_argument("--metadata-timestamp-unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-range-m", type=float, default=250.)
    parser.add_argument("--min-downward-cosine", type=float, default=0.05)
    parser.add_argument("--camera-convention", choices=("uav3d_carla", "nuscenes_optical"), default="uav3d_carla")
    parser.add_argument("--channels", nargs="+", help="Explicit channel list; default CAMERA_BOTTOM_id_* (one view per UAV)")
    parser.add_argument("--overwrite", action="store_true")
    config = vars(parser.parse_args(argv))
    result = export_perception_reports(config)
    print(json.dumps(result.attrs["export_receipt"], indent=2))


if __name__ == "__main__":
    main()
