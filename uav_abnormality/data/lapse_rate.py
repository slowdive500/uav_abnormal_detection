"""University of Kentucky LAPSE-RATE NetCDF -> comparable UAV contributions.

UTC and absolute geolocation are retained. Flight filenames identify physical
platforms; sensors and repeated flights are never promoted to extra UAVs.
"""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


PLATFORMS = {"BCT5B", "BCT5C", "BCT5D", "BCT5E", "SOLOW", "S1000", "M600P"}
DEFAULT_CONDITIONS = {
    "latitude": "Lat", "longitude": "Long", "altitude_asl": "Alt",
    "altitude_agl": "z", "velocity_east": "gvel_E", "velocity_north": "gvel_N",
    "velocity_up": "gvel_up", "yaw": "yaw", "pitch": "pitch", "roll": "roll",
}


def _clean(values, missing_values):
    result = np.asarray(values, dtype=float).reshape(-1).copy()
    for sentinel in missing_values:
        result[np.isclose(result, float(sentinel), rtol=0, atol=1e-3)] = np.nan
    result[~np.isfinite(result)] = np.nan
    return result


def _timestamp(dataset, config):
    name = config.get("time_variable")
    if name:
        raw = dataset[name].values
        if np.issubdtype(raw.dtype, np.number):
            unit = config.get("timestamp_unit")
            if not unit:
                raise ValueError("Numeric time_variable requires explicit timestamp_unit")
            return pd.to_datetime(raw, unit=unit, utc=True)
        return pd.to_datetime(raw, utc=True)
    parts = config.get("time_components", {
        "year": "Year", "month": "Month", "day": "Day", "hour": "Hour",
        "minute": "Minute", "second": "Sec"})
    # Sec can be fractional or exactly 60; add a timedelta to the minute.
    base = pd.to_datetime({k: dataset[v].values for k, v in parts.items() if k != "second"}, utc=True)
    return base + pd.to_timedelta(dataset[parts["second"]].values, unit="s")


def _feature_spec(config):
    features = config.get("features", ["u", "v"])
    if isinstance(features, dict):
        return [(name, variable if isinstance(variable, dict) else {"variable": variable})
                for name, variable in features.items()]
    return [(str(i), item if isinstance(item, dict) else {"variable": item}) for i, item in enumerate(features)]


def prepare_lapse_rate(config):
    """Read real NetCDFs and aggregate only colocated, contemporaneous reports.

    config: raw_dir; files or glob; features (NetCDF variable names or specs with
    variable/scale/offset); conditions mapping; time_bin_seconds; horizontal_bin_m;
    vertical_bin_m; origin_lat/lon; min_altitude_agl/max_altitude_agl; min_sources.
    Default label=-1 means the public QC flag is not an anomaly ground truth.
    """
    import xarray as xr

    root = Path(config["raw_dir"])
    paths = ([root / name for name in config["files"]] if config.get("files")
             else sorted(root.glob(config.get("glob", "**/*.cdf"))))
    if not paths:
        raise FileNotFoundError("No LAPSE-RATE NetCDF files found in %s" % root)
    features = _feature_spec(config)
    conditions = config.get("conditions", DEFAULT_CONDITIONS)
    missing = config.get("missing_values", [-9999.9])
    time_bin = float(config.get("time_bin_seconds", 5.0))
    h_bin = float(config.get("horizontal_bin_m", 250.0))
    v_bin = float(config.get("vertical_bin_m", 50.0))
    if min(time_bin, h_bin, v_bin) <= 0:
        raise ValueError("Time and spatial bin sizes must be positive")
    lat0, lon0 = float(config.get("origin_lat", 38.0)), float(config.get("origin_lon", -106.0))
    rows, audits = [], []
    allowed = set(config.get("platforms", sorted(PLATFORMS)))
    for path in paths:
        match = re.match(r"UKY\.([^.]+)\.b1\.(\d{8})\.", path.name)
        source = config.get("source_ids", {}).get(path.name, match.group(1) if match else None)
        if source not in allowed:
            continue  # Ground towers are references, not physical UAV sources.
        with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as dataset:
            times = _timestamp(dataset, config)
            frame = pd.DataFrame({"timestamp": times})
            for index, (_, spec) in enumerate(features):
                name = spec["variable"]
                if name not in dataset:
                    raise ValueError("%s has no feature %s" % (path.name, name))
                values = _clean(dataset[name].values, missing)
                frame["value_%d" % index] = values * float(spec.get("scale", 1)) + float(spec.get("offset", 0))
            for condition, variable in conditions.items():
                frame["condition_" + condition] = (_clean(dataset[variable].values, missing)
                    if variable in dataset else np.nan)
            # Never use file-local x/y for cross-flight voxel membership.
            lat = _clean(dataset[config.get("latitude_variable", "Lat")].values, missing)
            lon = _clean(dataset[config.get("longitude_variable", "Long")].values, missing)
            altitude = _clean(dataset[config.get("altitude_asl_variable", "Alt")].values, missing)
            agl = _clean(dataset[config.get("altitude_agl_variable", "z")].values, missing)
            east = np.deg2rad(lon - lon0) * 6371000.0 * np.cos(np.deg2rad(lat0))
            north = np.deg2rad(lat - lat0) * 6371000.0
            frame["_east"], frame["_north"], frame["_alt"] = east, north, altitude
            keep = np.isfinite(east) & np.isfinite(north) & np.isfinite(altitude)
            if config.get("min_altitude_agl") is not None:
                keep &= agl >= float(config["min_altitude_agl"])
            if config.get("max_altitude_agl") is not None:
                keep &= agl <= float(config["max_altitude_agl"])
            value_cols = ["value_%d" % i for i in range(len(features))]
            keep &= frame[value_cols].notna().any(axis=1).to_numpy()
            frame = frame.loc[keep].copy()
            if frame.empty:
                audits.append({"file": path.name, "source": source, "usable_rows": 0})
                continue
            frame["source_id"] = source
            frame["flight_id"] = path.stem
            frame["sequence_id"] = "lapse_rate_" + frame["timestamp"].dt.strftime("%Y%m%d")
            frame["timestamp"] = pd.to_datetime(np.floor(frame["timestamp"].astype("int64") / 1e9 / time_bin) * time_bin, unit="s", utc=True)
            grid = np.floor(frame[["_east", "_north", "_alt"]].to_numpy() / [h_bin, h_bin, v_bin]).astype(np.int64)
            frame["task_id"] = ["voxel_%d_%d_%d" % tuple(g) for g in grid]
            frame.drop(columns=["_east", "_north", "_alt"], inplace=True)
            audits.append({"file": path.name, "source": source, "usable_rows": len(frame),
                           "start": str(frame.timestamp.min()), "end": str(frame.timestamp.max())})
            rows.append(frame)
    if not rows:
        raise ValueError("No valid UAV measurements after platform/feature/altitude filtering")
    table = pd.concat(rows, ignore_index=True)
    keys = ["sequence_id", "task_id", "source_id", "timestamp"]
    numeric = [c for c in table if c.startswith(("value_", "condition_"))]
    # Aggregation collapses oversampling and duplicate on-board sensor releases.
    aggregate = {c: "mean" for c in numeric}
    aggregate["flight_id"] = lambda values: "|".join(sorted(set(values)))
    table = table.groupby(keys, as_index=False, sort=True).agg(aggregate)
    table["label"] = -1
    counts = table.groupby(["sequence_id", "task_id", "timestamp"])["source_id"].transform("nunique")
    minimum = int(config.get("min_sources", 1))
    table = table.loc[counts >= minimum].reset_index(drop=True)
    table["dataset"] = "lapse_rate"
    table["observation_kind"] = "measured"
    table.attrs["feature_names"] = [spec["variable"] for _, spec in features]
    table.attrs["file_audit"] = audits
    table.attrs["grouping"] = {"time_bin_seconds": time_bin, "horizontal_bin_m": h_bin, "vertical_bin_m": v_bin,
                               "vertical_reference": "ASL", "origin_lat": lat0, "origin_lon": lon0}
    if table.empty:
        raise ValueError("No concurrent groups meet min_sources; inspect flight times/locations, do not invent UAV sources")
    groups = table.groupby(["sequence_id", "task_id", "timestamp"])["source_id"].nunique()
    table.attrs["source_overlap"] = {str(int(k)): int(v) for k, v in groups.value_counts().sort_index().items()}
    if groups.max() < 2:
        warnings.warn("Selected LAPSE-RATE windows contain no colocated simultaneous multi-UAV reports", RuntimeWarning)
    return table
