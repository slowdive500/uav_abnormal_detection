"""Intel Berkeley fixed sensor observations, adapted to virtual UAV reports.

The measurements and horizontal coordinates are real; aircraft identities and
an optional fixed operating height are experimental assumptions. No flight or
sensor-height observations are invented. Held-out nodes never become reports.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = ["date", "time", "epoch", "node_id", "temperature", "humidity", "light", "voltage"]
REFERENCE_SEMANTICS = "held_out_measured_temperature_proxy"


def _node_ids(values, name):
    if values is None:
        return None
    result = []
    for value in values:
        number = float(value)
        if not np.isfinite(number) or number != int(number) or not 1 <= number <= 54:
            raise ValueError("%s must contain integer node IDs from 1 to 54" % name)
        result.append(int(number))
    if len(result) != len(set(result)):
        raise ValueError("%s contains duplicate node IDs" % name)
    return result


def _locations(path):
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["node_id", "x", "y"])
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Node coordinates must be finite")
    _node_ids(frame.node_id.tolist(), "coordinate node IDs")
    frame["node_id"] = frame.node_id.astype(int)
    return frame.set_index("node_id")


def _regions(config):
    regions = config.get("spatial_regions")
    if regions is None:
        return [{"id": "lab_temperature", "center_xy": None, "radius_m": None}]
    if not regions:
        raise ValueError("spatial_regions cannot be empty")
    result = []
    for region in regions:
        center = np.asarray(region["center_xy"], dtype=float)
        radius = float(region["radius_m"])
        if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Each spatial region needs finite center_xy and positive radius_m")
        result.append({"id": str(region["id"]), "center_xy": center.tolist(), "radius_m": radius})
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Spatial region IDs must be unique")
    return result


def prepare_intel_berkeley(config):
    """Return canonical reports plus JSON-serializable evaluation query attrs.

    ``node_ids`` selects reporting nodes (default all). ``query_node_ids`` is
    removed from reporting nodes, even if included in node_ids. ``start`` is
    inclusive, ``end`` exclusive. ``temperature_bounds_c`` is an optional,
    explicit validity policy, not an anomaly label. Missing times are never
    interpolated. ``time_bin_seconds`` averages observed samples in left-closed
    intervals and timestamps them at the right endpoint, after they are known.
    """
    root = Path(config["raw_dir"])
    data_path = root / config.get("data_file", "data.txt.gz")
    locations_path = root / config.get("locations_file", "mote_locs.txt")
    locations = _locations(locations_path)
    requested = _node_ids(config.get("node_ids"), "node_ids")
    query_nodes = set(_node_ids(config.get("query_node_ids", []), "query_node_ids"))
    report_nodes = (set(locations.index) if requested is None else set(requested)) - query_nodes
    if not report_nodes:
        raise ValueError("No reporting nodes remain after excluding query nodes")
    selected_nodes = report_nodes | query_nodes
    unknown = selected_nodes - set(locations.index)
    if unknown:
        raise ValueError("Selected nodes have no coordinates: %s" % sorted(unknown))
    regions = _regions(config)
    bin_seconds = float(config.get("time_bin_seconds", 300))
    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("time_bin_seconds must be finite and positive")
    bin_ns = int(round(bin_seconds * 1e9))
    if bin_ns <= 0:
        raise ValueError("time_bin_seconds must be at least one nanosecond")
    min_sources = int(config.get("min_sources", 1))
    if min_sources < 1:
        raise ValueError("min_sources must be positive")
    bounds = config.get("temperature_bounds_c")
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("temperature_bounds_c must be [lower, upper]")
    height = config.get("simulated_operation_height_m")
    if height is not None and (not np.isfinite(float(height)) or float(height) < 0):
        raise ValueError("simulated_operation_height_m must be finite and nonnegative")
    start = pd.to_datetime(config["start"], utc=True) if config.get("start") else None
    end = pd.to_datetime(config["end"], utc=True) if config.get("end") else None
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be before end")

    audit = {"raw_rows": 0, "invalid_timestamp_rows": 0, "invalid_node_rows": 0,
             "unselected_node_rows": 0, "outside_time_range_rows": 0,
             "selected_rows": 0, "nonfinite_temperature_rows": 0,
             "temperature_bounds_rejected_rows": 0}
    pieces = []
    # Chunking bounds memory on the full 2.3-million-row official file.
    # The official log uses literal spaces, with empty fields for missing
    # measurements. Collapsing whitespace would shift voltage into temperature.
    for chunk in pd.read_csv(data_path, sep=" ", header=None, names=RAW_COLUMNS,
                             dtype=str, chunksize=int(config.get("chunksize", 200000))):
        audit["raw_rows"] += len(chunk)
        times = pd.to_datetime(chunk.date + " " + chunk.time, errors="coerce", utc=True)
        nodes = pd.to_numeric(chunk.node_id, errors="coerce")
        valid_time = times.notna()
        valid_node = np.isfinite(nodes) & (nodes == np.floor(nodes)) & nodes.isin(locations.index)
        audit["invalid_timestamp_rows"] += int((~valid_time).sum())
        audit["invalid_node_rows"] += int((valid_time & ~valid_node).sum())
        valid = valid_time & valid_node
        selected = nodes.isin(selected_nodes)
        audit["unselected_node_rows"] += int((valid & ~selected).sum())
        valid &= selected
        in_time = pd.Series(True, index=chunk.index)
        if start is not None:
            in_time &= times >= start
        if end is not None:
            in_time &= times < end
        audit["outside_time_range_rows"] += int((valid & ~in_time).sum())
        valid &= in_time
        if not valid.any():
            continue
        piece = pd.DataFrame({"sample_timestamp": times[valid], "node_id": nodes[valid].astype(int)})
        piece["temperature"] = pd.to_numeric(chunk.loc[valid, "temperature"], errors="coerce")
        piece["epoch"] = pd.to_numeric(chunk.loc[valid, "epoch"], errors="coerce")
        audit["selected_rows"] += len(piece)
        finite = np.isfinite(piece.temperature)
        audit["nonfinite_temperature_rows"] += int((~finite).sum())
        if bounds is not None:
            within = piece.temperature.between(*bounds)
            audit["temperature_bounds_rejected_rows"] += int((finite & ~within).sum())
            finite &= within
        # Missing readings remain absent; they are never imputed or marked good.
        piece = piece.loc[finite].copy()
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        raise ValueError("No finite selected Intel Berkeley temperature readings")
    samples = pd.concat(pieces, ignore_index=True)
    audit["valid_observation_rows_before_deduplication"] = len(samples)
    # Exactly repeated packets cannot receive extra weight in an interval mean.
    samples = samples.drop_duplicates(["node_id", "sample_timestamp", "epoch", "temperature"])
    audit["duplicate_packets_removed"] = audit["valid_observation_rows_before_deduplication"] - len(samples)
    bin_start = (samples.sample_timestamp.astype("int64") // bin_ns) * bin_ns
    samples["timestamp"] = pd.to_datetime(bin_start + bin_ns, utc=True)
    samples["sequence_id"] = "intel_berkeley_" + pd.to_datetime(bin_start, utc=True).dt.strftime("%Y%m%d")
    grouped = samples.groupby(["sequence_id", "timestamp", "node_id"], as_index=False, sort=True).agg(
        value_0=("temperature", "mean"), sample_count=("temperature", "size"),
        sample_start=("sample_timestamp", "min"), sample_end=("sample_timestamp", "max"),
        epoch_min=("epoch", "min"), epoch_max=("epoch", "max"))
    grouped["condition_x"] = grouped.node_id.map(locations.x)
    grouped["condition_y"] = grouped.node_id.map(locations.y)
    if height is not None:
        grouped["condition_simulated_operation_height_m"] = float(height)
    grouped["source_id"] = grouped.node_id.map(lambda x: "virtual_uav_node_%02d" % x)
    grouped["raw_node_id"] = grouped.pop("node_id")
    grouped["label"] = -1
    grouped["dataset"] = "intel_berkeley"
    grouped["observation_kind"] = "measured_fixed_sensor_virtual_uav_mapping"
    reports, queries = [], []
    for region in regions:
        selected = grouped
        if region["center_xy"] is not None:
            delta = grouped[["condition_x", "condition_y"]].to_numpy() - region["center_xy"]
            selected = grouped.loc[np.linalg.norm(delta, axis=1) <= region["radius_m"]]
        selected = selected.copy()
        selected["task_id"] = region["id"]
        query = selected.loc[selected.raw_node_id.isin(query_nodes)].copy()
        report = selected.loc[selected.raw_node_id.isin(report_nodes)].copy()
        group_keys = ["sequence_id", "task_id", "timestamp"]
        quorum = report.groupby(group_keys).source_id.transform("nunique")
        report = report.loc[quorum >= min_sources]
        if report.empty:
            continue
        # A held-out measurement evaluates only a region/time with real reports.
        query = query.merge(report[group_keys].drop_duplicates(), on=group_keys, how="inner")
        reports.append(report)
        if not query.empty:
            query.rename(columns={"value_0": "reference_0", "source_id": "query_id"}, inplace=True)
            query["reference_semantics"] = REFERENCE_SEMANTICS
            queries.append(query)
    if not reports:
        raise ValueError("No reporting groups satisfy spatial regions and min_sources")
    table = pd.concat(reports, ignore_index=True).sort_values(
        ["sequence_id", "task_id", "timestamp", "source_id"]).reset_index(drop=True)
    query_table = pd.concat(queries, ignore_index=True) if queries else pd.DataFrame()
    overlap = table.groupby(["sequence_id", "task_id", "timestamp"]).source_id.nunique()
    table.attrs.update({
        "feature_names": ["temperature_c"], "raw_audit": audit,
        "report_node_ids": sorted(report_nodes), "query_node_ids": sorted(query_nodes),
        "source_mapping": "one fixed sensor node to one stable virtual UAV identity; no real flight",
        "coordinate_frame": "official laboratory xy metres, origin upper right corner",
        "height_semantics": "simulated fixed operation condition only; original sensor height unknown",
        "natural_fault_labels": "unknown; all original report labels remain -1",
        "reference_semantics": "no independent ground truth; raw readings may contain natural faults",
        "time_semantics": "source timezone unspecified; wall-clock encoded nominal UTC, not a verified timezone conversion",
        "grouping": {"time_bin_seconds": bin_seconds, "timestamp_convention": "right endpoint",
                     "aggregation": "observed-sample arithmetic mean", "spatial_regions": regions,
                     "interpolation": False},
        "validity_policy": {"temperature_bounds_c": None if bounds is None else bounds.tolist()},
        "source_overlap": {str(int(k)): int(v) for k, v in overlap.value_counts().sort_index().items()},
        "evaluation_query_semantics": REFERENCE_SEMANTICS,
        "evaluation_query_count": len(query_table),
        "evaluation_queries": json.loads(query_table.to_json(orient="records", date_format="iso")) if len(query_table) else [],
    })
    return table
