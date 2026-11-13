"""TaxiBJ traffic fields with explicitly simulated, noisy participant reports.

TaxiBJ contains ground-vehicle aggregate flows, not UAV sensor readings. This
adapter simulates a report-level UAV crowdsensing experiment on those fields;
it does not simulate cameras, flight dynamics, or claim real aerial collection.
The underlying traffic field is retained only in evaluation ``truth_*`` columns.
"""
from pathlib import Path
import json
from http.client import IncompleteRead
import shutil
from urllib.error import URLError
import zipfile
import numpy as np
import pandas as pd

from .schema import validate_table


CHANNELS = {"inflow": 0, "outflow": 1}


def download_taxibj(manifest, destination):
    """Acquire original mirror files and verify the original per-file hashes.

    The optional archive avoids transferring hundreds of MB of mostly integer
    values encoded as uncompressed float64. Only explicitly manifested TaxiBJ
    members are extracted; other repository files are never used or executed.
    """
    from .download import _inside, download_manifest, verify_file

    def fetch_files(specification, target):
        # Long public HTTP responses can be truncated by network proxies. The
        # shared downloader keeps partial files and validates resumed ranges.
        for attempt in range(4):
            try:
                return download_manifest(specification, target)
            except (IncompleteRead, URLError, TimeoutError, ValueError):
                if attempt == 3:
                    raise

    manifest = json.loads(Path(manifest).read_text()) if not isinstance(manifest, dict) else manifest
    root = Path(destination)
    archive = manifest.get("acquisition_archive")
    if archive and (not archive.get("checksum") or not archive.get("size")):
        raise ValueError("Optional TaxiBJ acquisition_archive needs pinned size and checksum")
    if not archive or all(verify_file(_inside(root, e["path"]), e) for e in manifest["files"]):
        receipt_path = root / "download_receipt.json"
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        receipt = fetch_files(manifest, root)
        for key in ("acquisition_archive", "acquisition_note", "transport_url"):
            if previous.get(key):
                receipt[key] = previous[key]
        if any(key in previous for key in ("acquisition_archive", "acquisition_note", "transport_url")):
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    archive_receipt = download_manifest({"dataset": manifest["dataset"], "files": [archive]}, root / "archive")
    with zipfile.ZipFile(root / "archive" / archive["path"]) as handle:
        for entry in manifest["files"]:
            member = archive["member_prefix"] + entry["path"]
            info = handle.getinfo(member)
            if info.file_size != entry["size"]:
                raise ValueError("TaxiBJ archive member size differs from manifest: " + member)
            target = _inside(root, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with handle.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if not verify_file(temporary, entry):
                raise ValueError("TaxiBJ archive member checksum differs from manifest: " + member)
            temporary.replace(target)
    receipt = download_manifest(manifest, root)
    receipt["acquisition_archive"] = archive_receipt["files"][0]
    (root / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def decode_timestamps(raw_dates):
    """Decode YYYYMMDD[01..48] as ends of 30-minute Beijing-local bins.

    Aggregate flow becomes available after its bin has finished. The returned
    day and slot retain the original local task date, including slot 48.
    """
    dates = [value.decode("ascii") if isinstance(value, bytes) else str(value)
             for value in raw_dates]
    if any(len(value) != 10 or not value.isdigit() for value in dates):
        raise ValueError("TaxiBJ date must be YYYYMMDD followed by slot 01..48")
    slots = np.array([int(value[-2:]) for value in dates], dtype=int)
    if ((slots < 1) | (slots > 48)).any():
        raise ValueError("TaxiBJ slots must be in 01..48")
    days = np.array([value[:8] for value in dates])
    midnight = pd.to_datetime(days, format="%Y%m%d", errors="raise")
    local = midnight + pd.to_timedelta(slots * 30, unit="min")
    utc = local.tz_localize("Asia/Shanghai").tz_convert("UTC")
    return utc.astype("int64").to_numpy() / 1e9, days, slots


def _read_fields(config):
    import h5py

    root = Path(config.get("raw_dir", "data/raw/taxibj"))
    names = config.get("files", ["BJ15_M32x32_T30_InOut.h5"])
    if isinstance(names, str) or not names:
        raise ValueError("files must be a nonempty list of original TaxiBJ HDF5 filenames")
    chunks = []
    for name in names:
        path = root / name
        with h5py.File(path, "r") as handle:
            if "data" not in handle or "date" not in handle:
                raise ValueError("Need original TaxiBJ data/date datasets, not a prediction HDF5")
            data = handle["data"][:]
            timestamps, days, slots = decode_timestamps(handle["date"][:])
        if data.shape != (len(timestamps), 2, 32, 32):
            raise ValueError("TaxiBJ data must have shape [time,2,32,32]")
        chunks.append((data, timestamps, days, slots, np.repeat(str(name), len(timestamps))))
    data, timestamps, days, slots, filenames = [np.concatenate(parts) for parts in zip(*chunks)]
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("Duplicate TaxiBJ timestamps across input files")
    order = np.argsort(timestamps, kind="stable")
    data, timestamps, days, slots, filenames = [array[order] for array in
                                              (data, timestamps, days, slots, filenames)]
    keep = np.ones(len(timestamps), dtype=bool)
    if config.get("complete_days_only", True):
        complete = [day for day in np.unique(days)
                    if np.array_equal(np.sort(slots[days == day]), np.arange(1, 49))]
        keep &= np.isin(days, complete)
    if config.get("start_date"):
        keep &= days >= str(config["start_date"]).replace("-", "")
    if config.get("end_date"):
        keep &= days <= str(config["end_date"]).replace("-", "")
    if config.get("max_days") is not None:
        limit = int(config["max_days"])
        if limit < 1:
            raise ValueError("max_days must be positive or null")
        keep &= np.isin(days, np.unique(days[keep])[:limit])
    if not keep.any():
        raise ValueError("TaxiBJ date/complete-day selection is empty")
    return tuple(array[keep] for array in (data, timestamps, days, slots, filenames))


def prepare_taxibj(config):
    """Generate normal reports; apply adversarial corruption AFTER splitting.

    Each virtual participant has a stable relative calibration bias and noise
    scale. Participation is sampled independently for each task/time, followed
    by a configurable random quorum fill. Sources may report several cells in
    one 30-minute interval: this is a task assignment model, not a flight path.
    No predictor and no model-specific feature is computed from ground truth.
    """
    channels = config.get("channels", ["inflow"])
    if not channels or len(set(channels)) != len(channels) or any(c not in CHANNELS for c in channels):
        raise ValueError("channels must be a nonempty unique list of inflow/outflow")
    cells = config.get("cells", [[15, 15], [15, 16], [16, 15], [16, 16]])
    if cells == "all":
        cells = [[row, col] for row in range(32) for col in range(32)]
    coords = np.asarray(cells)
    if (coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0
            or not np.issubdtype(coords.dtype, np.integer)
            or (coords < 0).any() or (coords >= 32).any()
            or len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError("cells must contain distinct integer [row,column] pairs in 0..31")
    source_count = int(config.get("source_count", 8))
    min_sources = int(config.get("min_sources", 3))
    probability = float(config.get("participation_probability", 0.8))
    if not 1 <= min_sources <= source_count or not 0 <= probability <= 1:
        raise ValueError("Require 1<=min_sources<=source_count and participation_probability in [0,1]")
    noise = config.get("normal_report_noise", {})
    relative_std = float(noise.get("relative_std", 0.02))
    absolute_std = float(noise.get("absolute_std", 0.25))
    bias_std = float(noise.get("source_bias_std", 0.005))
    if not np.isfinite([relative_std, absolute_std, bias_std]).all() or min(relative_std, absolute_std, bias_std) < 0:
        raise ValueError("Normal report noise parameters must be finite and nonnegative")
    if relative_std == absolute_std == 0:
        raise ValueError("Normal reports require measurement noise; exact ground-truth copies are not supported")
    seed = int(config.get("seed", 20261113))
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    fields, timestamps, days, slots, filenames = _read_fields(config)
    sequence_mode = config.get("sequence_mode", "campaign")
    if sequence_mode not in ("campaign", "day"):
        raise ValueError("sequence_mode must be campaign or day")
    sequence_ids = (["taxibj_" + day for day in days] if sequence_mode == "day" else
                    ["taxibj_" + Path(name).stem for name in filenames])
    sequence_ids = np.asarray(sequence_ids)
    source_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    source_bias = source_rng.normal(0, bias_std, (source_count, len(channels)))
    source_scale = source_rng.uniform(0.75, 1.25, (source_count, len(channels)))
    ids = np.array(["virtual_uav_%03d" % (i + 1) for i in range(source_count)])
    pieces = []
    for row, column in coords:
        truth = fields[:, [CHANNELS[channel] for channel in channels], row, column].astype(float)
        valid = np.isfinite(truth).all(axis=1) & (truth >= 0).all(axis=1)
        # RNG for each task is independent of selected cells and their ordering.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, int(row), int(column)]))
        quorum_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, int(row), int(column)]))
        measurement_rng = np.random.default_rng(np.random.SeedSequence([seed, 3, int(row), int(column)]))
        active = rng.random((len(timestamps), source_count)) < probability
        for time_index in np.flatnonzero(active.sum(axis=1) < min_sources):
            needed = min_sources - int(active[time_index].sum())
            absent = np.flatnonzero(~active[time_index])
            active[time_index, quorum_rng.choice(absent, needed, replace=False)] = True
        active[~valid] = False
        scale = (absolute_std + relative_std * truth[:, None, :]) * source_scale[None, :, :]
        values = np.maximum(0, truth[:, None, :] * (1 + source_bias[None, :, :])
                            + measurement_rng.normal(size=scale.shape) * scale)
        if noise.get("round_counts", False):
            values = np.rint(values)
        times, sources = np.nonzero(active)
        piece = pd.DataFrame({
            "sequence_id": sequence_ids[times],
            "task_id": "taxibj_r%02d_c%02d" % (row, column),
            "timestamp": timestamps[times], "source_id": ids[sources],
            "condition_bin_start": timestamps[times] - 1800,
            "condition_bin_end": timestamps[times],
            "label": 0, "grid_row": int(row), "grid_column": int(column),
            "condition_x": float(column), "condition_y": float(row),
            "condition_time_sin": np.sin(2 * np.pi * (slots[times] - 1) / 48),
            "condition_time_cos": np.cos(2 * np.pi * (slots[times] - 1) / 48),
            "timeslot": slots[times], "raw_file": filenames[times],
            "source_kind": "simulated_uav_participant",
            "reference_semantics": "traffic_field_ground_truth",
        })
        for feature in range(len(channels)):
            piece["value_%d" % feature] = values[times, sources, feature]
            piece["truth_%d" % feature] = truth[times, feature]
        pieces.append(piece)
    result = validate_table(pd.concat(pieces, ignore_index=True))
    result.attrs["provenance"] = {
        "dataset": "TaxiBJ", "original_data": "ground_taxi_aggregate_traffic",
        "experiment": "simulated_uav_report_level_crowdsensing", "actual_uav_flights": False,
        "channels": list(channels), "grid_shape": [32, 32], "interval_seconds": 1800,
        "timestamp_semantics": "Beijing local bin end (report availability) converted to UTC seconds",
        "complete_days_only": bool(config.get("complete_days_only", True)),
        "files": list(config.get("files", ["BJ15_M32x32_T30_InOut.h5"])),
        "sequence_mode": sequence_mode,
        "seed": seed, "normal_report_noise": {"relative_std": relative_std,
            "absolute_std": absolute_std, "source_bias_std": bias_std},
        "anomalies": "none; injected by the common protocol after splitting",
        "reference_semantics": "traffic_field_ground_truth",
        "truth_features_used_for_model_input": False,
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download original TaxiBJ matrices with hash verification")
    parser.add_argument("--manifest", default="data/manifests/taxibj.json")
    parser.add_argument("--destination", default="data/raw/taxibj")
    arguments = parser.parse_args()
    print(json.dumps(download_taxibj(arguments.manifest, arguments.destination), indent=2))
