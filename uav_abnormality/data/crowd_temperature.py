"""Import the published queensu/crowd_temperature reports without regenerating them.

The source dataset is taxi mobility with author-simulated temperatures, not UAV
measurements. A virtual-UAV interpretation changes metadata only. The original
CSV is account-gated at IEEE DataPort; this module also supports explicit column
mapping for legitimately downloaded exports and never substitutes generated data.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .download import download_manifest, file_digest, verify_file
from .schema import KEYS, validate_table


OFFICIAL_MD5 = "8edf0ea0427d9e961ce77ca0c9a0d2f0"
DATASET_URL = "https://ieee-dataport.org/open-access/crawdad-queensucrowdtemperature"
ALIASES = {
    "source": ("source_id", "taxi_id", "taxicab_id", "cab_id", "taxi", "participant_id"),
    "task": ("task_id", "grid_id", "region_id", "area_id", "grid", "region", "area"),
    "value": ("temperature", "sensed_temperature", "reported_temperature", "value_0"),
    "timestamp": ("timestamp", "datetime", "date_time"),
    "date": ("date", "day"),
    "period": ("period", "time_period", "time_slot"),
    "truth": ("ground_truth", "ground_truth_temperature", "true_temperature", "truth_0"),
    "source_class": ("source_class", "taxi_class", "participant_class", "class", "type"),
    "noise": ("sigma", "error_range", "noise_std", "standard_deviation"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "long", "lon"),
}


def _norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _column(frame, name, mapping, required=False):
    explicit = mapping.get(name)
    if explicit is False:
        return None
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError("Configured %s column %r not found; available: %s" %
                             (name, explicit, list(frame.columns)))
        return explicit
    aliases = set(_norm(x) for x in ALIASES.get(name, ()))
    matches = [c for c in frame.columns if _norm(c) in aliases]
    if len(matches) > 1:
        raise ValueError("Ambiguous %s columns %s; set columns.%s explicitly" % (name, matches, name))
    if not matches and required:
        raise ValueError("Cannot identify %s column; set columns.%s. Available: %s" %
                         (name, name, list(frame.columns)))
    return matches[0] if matches else None


def _identity(series, name):
    values = series.astype("string").str.strip()
    if values.isna().any() or (values == "").any():
        raise ValueError("Missing %s identity" % name)
    return values.astype(str)


def _finite(series, name, allow_missing=False):
    values = pd.to_numeric(series, errors="raise").astype(float)
    if np.isinf(values).any() or (not allow_missing and values.isna().any()):
        raise ValueError("Non-finite %s values" % name)
    return values


def _timestamps(raw, columns, config):
    stamp_col = _column(raw, "timestamp", columns)
    if stamp_col:
        values = raw[stamp_col]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all() and not config.get("timestamp_format"):
            unit = config.get("timestamp_unit")
            if unit not in ("s", "ms", "us", "ns"):
                raise ValueError("Numeric timestamps require explicit timestamp_unit: s/ms/us/ns")
            stamp = pd.to_datetime(numeric, unit=unit, utc=True, errors="raise")
        else:
            stamp = pd.to_datetime(values, format=config.get("timestamp_format"), errors="raise")
    else:
        day_col = _column(raw, "date", columns, required=True)
        period_col = _column(raw, "period", columns, required=True)
        dates = raw[day_col]
        if pd.to_numeric(dates, errors="coerce").notna().all() and not config.get("date_format"):
            if config.get("date_origin") is None or config.get("day_index_base") is None:
                raise ValueError("Numeric days require date_origin and day_index_base; dates are never guessed")
            indices = _finite(dates, "day") - float(config["day_index_base"])
            if not np.equal(indices, np.floor(indices)).all() or (indices < 0).any():
                raise ValueError("Day indices must be nonnegative integers after day_index_base")
            dates = pd.Timestamp(config["date_origin"]) + pd.to_timedelta(indices, unit="D")
        else:
            dates = pd.to_datetime(dates, format=config.get("date_format"), errors="raise")
        hours = config.get("period_to_hours")
        if not isinstance(hours, dict) or not hours:
            raise ValueError("Date/period data require explicit period_to_hours mapping; period meanings are not guessed")
        hours = {str(k).strip(): float(v) for k, v in hours.items()}
        if any(not np.isfinite(x) or x < 0 or x >= 24 for x in hours.values()):
            raise ValueError("period_to_hours values must lie in [0,24)")
        if len(set(hours.values())) != len(hours):
            raise ValueError("Different periods must map to different hours")
        offsets = _identity(raw[period_col], "period").map(hours)
        if offsets.isna().any():
            raise ValueError("Unmapped periods: %s" % sorted(set(raw.loc[offsets.isna(), period_col])))
        stamp = dates + pd.to_timedelta(offsets, unit="h")
    if stamp.isna().any():
        raise ValueError("Missing timestamps")
    if stamp.dt.tz is None:
        timezone = config.get("timezone")
        if not timezone:
            raise ValueError("Naive timestamps require an explicit timezone, e.g. Europe/Rome")
        stamp = stamp.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return stamp.dt.tz_convert("UTC")


def _input_path(config):
    path = Path(config.get("input", Path(config.get("raw_dir", "data/raw/crowd_temperature")) /
                           "crowd_temperature.csv"))
    if not path.is_file():
        raise FileNotFoundError(
            "Missing published crowd_temperature.csv at %s. Download the original file after IEEE login at %s "
            "and place it here; expected original MD5 %s. No replacement data are generated. "
            "See docs/crowd_temperature.md." % (path, DATASET_URL, OFFICIAL_MD5))
    checksum = config.get("expected_checksum", "md5:" + OFFICIAL_MD5)
    if not checksum or not verify_file(path, {"checksum": checksum}):
        raise ValueError("crowd_temperature checksum mismatch (expected %r). Pin a verified export checksum "
                         "explicitly; never accept an HTML login page as the dataset." % checksum)
    return path


def prepare_crowd_temperature(config):
    """Return a canonical contribution table from the actual published CSV.

    Unknown labels stay -1. Source honesty classes and noise parameters are
    preserved as audit fields, not model features or per-report anomaly labels.
    Date/period mappings must be explicit because the gated source header could
    not be inspected during implementation. ``profiles.smoke`` may filter a
    fixed source/task subset; it never repeats or synthesizes observations.
    """
    config = dict(config)
    profile = config.get("profile", "full")
    profiles = config.get("profiles", {})
    if profiles:
        if profile not in profiles:
            raise ValueError("Unknown crowd_temperature profile %r" % profile)
        config.update(profiles[profile] or {})
    path = _input_path(config)
    raw = pd.read_csv(path, sep=config.get("delimiter", ","), dtype=str,
                      encoding=config.get("encoding", "utf-8-sig"), keep_default_na=True)
    if not len(raw):
        raise ValueError("crowd_temperature CSV has no rows")
    if config.get("layout") == "published_csv":
        return _prepare_published_csv(raw, config, path)
    columns = config.get("columns", {}) or {}
    source = _column(raw, "source", columns, required=True)
    task = _column(raw, "task", columns, required=True)
    value = _column(raw, "value", columns, required=True)
    truth = _column(raw, "truth", columns)
    if truth == value:
        raise ValueError("Observed temperature and ground truth cannot use the same column")
    frame = pd.DataFrame({
        "timestamp": _timestamps(raw, columns, config),
        "sequence_id": str(config.get("sequence_id", "rome_2014_02_01_to_04")),
        "source_id": _identity(raw[source], "source"),
        "task_id": _identity(raw[task], "task"),
        "value_0": _finite(raw[value], "temperature", allow_missing=True),
        "label": np.full(len(raw), -1, dtype=np.int8),
        "original_source_id": _identity(raw[source], "source"),
        "original_task_id": _identity(raw[task], "task"),
        "observation_origin": "published_taxi_mobility_with_author_simulated_temperature",
        "source_role": "virtual_uav_from_taxi" if config.get("virtual_uav", False) else "taxicab",
        "virtual_uav": bool(config.get("virtual_uav", False)),
        "label_provenance": "unknown_no_verified_report_labels",
        "raw_row_number": np.arange(2, len(raw) + 2),
    })
    for original in raw.columns:
        name = "audit_raw_" + re.sub(r"\W+", "_", original.strip()).strip("_").lower()
        if name in frame:
            raise ValueError("Raw column names collide after normalization: %r" % original)
        frame[name] = raw[original]
    for field in ("source_class", "noise"):
        column = _column(raw, field, columns)
        if column:
            frame["audit_" + field] = raw[column]
    for field in ("latitude", "longitude"):
        column = _column(raw, field, columns)
        if column:
            frame["audit_" + field] = _finite(raw[column], field, allow_missing=True)
    if truth:
        frame["truth_0"] = _finite(raw[truth], "reference truth", allow_missing=True)
        inconsistent = frame.groupby(["task_id", "timestamp"]).truth_0.nunique(dropna=True) > 1
        if inconsistent.any():
            raise ValueError("Conflicting reference truth for the same region and period")
    label_column = columns.get("label")
    if label_column:
        if config.get("label_semantics") != "per_report_anomaly":
            raise ValueError("Importing labels requires label_semantics: per_report_anomaly; source classes are not report labels")
        if label_column not in raw or label_column in (source, task, value, truth,
                                                      _column(raw, "source_class", columns),
                                                      _column(raw, "noise", columns)):
            raise ValueError("Invalid report label column; source honesty/noise cannot label individual reports")
        mapping = config.get("label_map")
        if not isinstance(mapping, dict) or not mapping or any(v not in (-1, 0, 1) for v in mapping.values()):
            raise ValueError("Explicit label_map to -1/0/1 is required")
        labels = raw[label_column].map({str(k): v for k, v in mapping.items()})
        unknown_values = raw.loc[labels.isna() & raw[label_column].notna(), label_column].unique()
        if len(unknown_values):
            raise ValueError("Unmapped report labels: %s" % unknown_values.tolist())
        frame["label"] = labels.fillna(-1).astype(np.int8)
        frame["label_provenance"] = "explicit_imported_per_report_annotation"
    duplicate_count = int(frame.duplicated(KEYS).sum())
    if duplicate_count:
        raise ValueError("Duplicate source/region/period reports (%d); inspect original timestamps and period mapping" % duplicate_count)
    total_rows = len(frame)
    for field, key in (("source_id", "source_ids"), ("task_id", "task_ids")):
        if config.get(key) is not None:
            requested = set(str(x) for x in config[key])
            absent = requested - set(frame[field])
            if absent:
                raise ValueError("Requested %s absent from source file: %s" % (key, sorted(absent)))
            frame = frame.loc[frame[field].isin(requested)]
    if config.get("max_sources") is not None:
        limit = int(config["max_sources"])
        if limit < 1:
            raise ValueError("max_sources must be positive")
        selected = sorted(frame.source_id.unique())[:limit]
        frame = frame.loc[frame.source_id.isin(selected)]
    frame = validate_table(frame)
    counts = frame.loc[frame.value_0.notna()].groupby(["task_id", "timestamp"]).source_id.nunique()
    stamp = np.sort(frame.timestamp.unique())
    frame.attrs.update({
        "dataset": "queensu/crowd_temperature", "doi": "10.15783/C7CG65", "input": str(path),
        "sha256": file_digest(path), "md5": file_digest(path, "md5"),
        "official_original_checksum_verified": file_digest(path, "md5") == OFFICIAL_MD5,
        "source_dataset_is_uav": False, "virtual_uav": bool(config.get("virtual_uav", False)),
        "author_simulated_temperature": True, "adapter_generated_observations": False,
        "ground_truth_used_to_generate_adapter_observations": False,
        "profile": profile, "original_rows": total_rows, "selected_rows": len(frame),
        "source_count": int(frame.source_id.nunique()), "task_count": int(frame.task_id.nunique()),
        "timestamp_count": len(stamp), "median_step_seconds": float(np.median(np.diff(stamp))) if len(stamp) > 1 else None,
        "rounds_with_three_sources": int((counts >= 3).sum()),
        "truth_available": "truth_0" in frame, "known_report_labels": int((frame.label >= 0).sum()),
        "native_timing_preserved": True, "resampled": False,
        "timestamp_interpretation": {k: config[k] for k in ("timezone", "timestamp_unit", "timestamp_format",
                                      "date_format", "date_origin", "day_index_base", "period_to_hours") if k in config},
    })
    return frame


def _prepare_published_csv(raw, config, path):
    """Verified original six-column CSV: explicit time buckets and Rome grid."""
    required = ["Taxi ID", "Date", "Time", "Latitude", "Longitude", "Temperature"]
    if list(raw.columns) != required:
        raise ValueError("Published CSV requires exact original columns: %s" % required)
    date_text = raw.Date.str.strip()
    hyphen = date_text.str.fullmatch(r"\d{2}-\d{2}-\d{2}")
    slash = date_text.str.fullmatch(r"\d{2}/\d{2}/\d{4}")
    if not (hyphen | slash).all():
        raise ValueError("Unrecognized original date encoding")
    dates = pd.Series(index=raw.index, dtype="datetime64[ns]")
    dates.loc[hyphen] = pd.to_datetime(date_text[hyphen], format="%m-%d-%y", errors="raise")
    dates.loc[slash] = pd.to_datetime(date_text[slash], format="%d/%m/%Y", errors="raise")
    times = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + raw.Time.str.strip(),
                           format="mixed", utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Invalid original Date/Time")
    latitude = _finite(raw.Latitude, "latitude")
    longitude = _finite(raw.Longitude, "longitude")
    bounds = np.asarray(config["grid_bounds"], float)
    shape = np.asarray(config.get("grid_shape", [3, 3]), int)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("grid_bounds must be [lat_min, lat_max, lon_min, lon_max]")
    if shape.shape != (2,) or (shape < 1).any():
        raise ValueError("grid_shape must be two positive integers")
    inside = latitude.between(bounds[0], bounds[1], inclusive="left") & longitude.between(bounds[2], bounds[3], inclusive="left")
    if not inside.all():
        raise ValueError("Original positions outside configured Rome task grid")
    row = np.floor((latitude - bounds[0]) / (bounds[1] - bounds[0]) * shape[0]).astype(int)
    column = np.floor((longitude - bounds[2]) / (bounds[3] - bounds[2]) * shape[1]).astype(int)
    if "time_bin_minutes" in config and "time_bin_hours" in config:
        raise ValueError("Set only one of time_bin_minutes and time_bin_hours")
    if "time_bin_minutes" in config:
        minutes = config["time_bin_minutes"]
        if (isinstance(minutes, (bool, np.bool_)) or not isinstance(minutes, (int, np.integer))
                or minutes < 1 or 1440 % minutes):
            raise ValueError("time_bin_minutes must be a positive integer divisor of 1440")
    else:
        hours = config.get("time_bin_hours", 6)
        if (isinstance(hours, (bool, np.bool_)) or not isinstance(hours, (int, np.integer))
                or hours < 1 or 24 % hours):
            raise ValueError("time_bin_hours must be a positive integer divisor of 24")
        minutes = 60 * hours
    bucket = (times.dt.tz_convert("Europe/Rome").dt.floor("%dmin" % minutes)
              + pd.Timedelta(minutes=minutes))
    frame = pd.DataFrame({"sequence_id": str(config.get("sequence_id", "rome_20140201_04")),
                          "task_id": "rome_grid_" + row.astype(str) + "_" + column.astype(str),
                          "timestamp": bucket.dt.tz_convert("UTC").astype("int64") / 1e9,
                          "source_id": _identity(raw["Taxi ID"], "source"),
                          "value_0": _finite(raw.Temperature, "temperature", allow_missing=True),
                          "sample_timestamp": times.astype("int64") / 1e9,
                          "condition_latitude": latitude, "condition_longitude": longitude,
                          "condition_grid_row": row, "condition_grid_column": column})
    original_count = len(frame)
    all_sources = sorted(frame.source_id.unique(), key=int)
    # Fixed-vocabulary neural baselines may only use identities seen in the
    # training prefix. This selection never reads temperature or future values.
    prefix = config.get("source_selection_first_slots")
    candidates = all_sources
    if prefix is not None:
        if int(prefix) < 1:
            raise ValueError("source_selection_first_slots must be positive")
        initial = np.sort(frame.timestamp.unique())[:int(prefix)]
        candidates = sorted(frame.loc[frame.timestamp.isin(initial), "source_id"].unique(), key=int)
    if config.get("source_ids") is not None:
        candidates = [str(x) for x in config["source_ids"]]
        if not set(candidates).issubset(all_sources):
            raise ValueError("Requested original taxi IDs not in the dataset")
    if config.get("max_sources") is not None:
        if int(config["max_sources"]) < 1:
            raise ValueError("max_sources must be positive")
        candidates = candidates[:int(config["max_sources"])]
    frame = frame[frame.source_id.isin(candidates)].copy()
    selected_raw_count = len(frame)
    grouped = frame.groupby(KEYS, as_index=False).agg(
        value_0=("value_0", "mean"), sample_count=("value_0", "count"),
        sample_start=("sample_timestamp", "min"), sample_end=("sample_timestamp", "max"),
        condition_latitude=("condition_latitude", "mean"), condition_longitude=("condition_longitude", "mean"),
        condition_grid_row=("condition_grid_row", "first"), condition_grid_column=("condition_grid_column", "first"))
    grouped["label"] = -1
    grouped["original_source_id"] = grouped.source_id
    grouped["source_role"] = "virtual_uav_from_taxi"
    grouped["observation_origin"] = "author_simulated_temperature_real_taxi_participation"
    grouped["condition_sample_count"] = grouped.sample_count
    grouped["condition_last_sample_age_seconds"] = grouped.timestamp - grouped.sample_end
    grouped = validate_table(grouped)
    grouped.attrs.update({"dataset": "queensu/crowd_temperature", "doi": "10.15783/C7CG65",
        "sha256": file_digest(path), "official_original_checksum_verified": file_digest(path, "md5") == OFFICIAL_MD5,
        "original_rows": original_count, "original_sources": len(all_sources), "selected_original_rows": selected_raw_count,
        "reports_after_explicit_aggregation": len(grouped), "selected_source_ids": candidates,
        "source_selection": {"first_slots": prefix, "max_sources": config.get("max_sources"),
                             "rule": "numeric ID order in training prefix, independent of measurements"},
        "timestamp_count": int(grouped.timestamp.nunique()), "time_bin_minutes": int(minutes),
        "time_bin_hours": minutes / 60,
        "timestamp_convention": "Rome-local bucket right endpoint converted to UTC",
        "date_formats": {"hyphen": "%m-%d-%y", "slash": "%d/%m/%Y", "slash_rows": int(slash.sum())},
        "grid_bounds": bounds.tolist(), "grid_shape": shape.tolist(),
        "grid_semantics": "explicit experimental Rome grid; original CSV has no region IDs",
        "aggregation": "observed same-source same-grid same-period mean; no interpolation",
        "reference_semantics": "original CSV has no reference temperature, source classes or fault labels",
        "truth_available": False, "author_simulated_temperature": True, "source_dataset_is_uav": False,
        "source_mapping": "stable original taxi ID interpreted as virtual UAV; no simulated flight trajectory"})
    return grouped


def acquire_crowd_temperature(destination, input_path=None, download_url=None):
    """Import an authenticated browser download or a caller-supplied public URL.

    Always verifies the original published checksum and uses download_manifest
    to generate a receipt. No credentials are read, requested, or persisted.
    """
    if (input_path is None) == (download_url is None):
        raise ValueError("Provide exactly one local input_path or public download_url")
    if input_path is not None and not verify_file(input_path, {"checksum": "md5:" + OFFICIAL_MD5}):
        raise ValueError("Original CSV checksum mismatch; verify the local IEEE download before importing")
    url = Path(input_path).resolve().as_uri() if input_path is not None else str(download_url)
    manifest = {"dataset": "queensu/crowd_temperature", "files": [{
        "path": "crowd_temperature.csv", "url": url,
        "checksum": "md5:" + OFFICIAL_MD5,
    }]}
    return download_manifest(manifest, destination)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Acquire the original crowd_temperature CSV with its published checksum")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Original CSV downloaded using an IEEE account")
    group.add_argument("--url", help="Public or user-authorized direct file URL")
    parser.add_argument("--destination", default="data/raw/crowd_temperature")
    arguments = parser.parse_args()
    print(json.dumps(acquire_crowd_temperature(arguments.destination, arguments.input, arguments.url), indent=2))


if __name__ == "__main__":
    main()
