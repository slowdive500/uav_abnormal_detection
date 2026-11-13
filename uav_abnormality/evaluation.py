"""Offline reference evaluation. None of these functions feeds model inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data.schema import feature_columns

ROUND = ["sequence_id", "task_id", "timestamp"]


def retained_mask(frame, threshold):
    if "prediction" in frame:
        return frame.prediction.to_numpy() == 0
    return frame.score.to_numpy() <= threshold if threshold is not None else np.ones(len(frame), dtype=bool)


def _errors(prediction, reference):
    prediction, reference = np.asarray(prediction, float), np.asarray(reference, float)
    valid = np.isfinite(prediction) & np.isfinite(reference)
    error = prediction[valid] - reference[valid]
    return {"n": int(valid.sum()), "mae": float(np.abs(error).mean()) if len(error) else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else None}


def _rank(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or len(np.unique(a[valid])) < 2 or len(np.unique(b[valid])) < 2:
        return None
    return float(spearmanr(a[valid], b[valid]).statistic)


def evaluate_references(predictions, threshold, output):
    """Report error and identical median postprocessing for both detectors.

    ``truth_*`` is supplied only by adapters with a task reference. Original
    Intel measurements are not promoted to true environmental values here.
    """
    if not len(predictions):
        return {"status": "no_scored_reports", "retained_fraction": None}
    cols = feature_columns(predictions)
    truth = ["truth_" + c[len("value_"):] for c in cols]
    if not all(c in predictions for c in truth):
        return {"status": "no_task_reference"}
    measured, reference = predictions[cols].to_numpy(float), predictions[truth].to_numpy(float)
    error = np.abs(measured - reference)
    count = np.isfinite(error).sum(axis=1)
    row_error = np.nansum(error, axis=1) / np.maximum(count, 1)
    row_error[count == 0] = np.nan
    retained = retained_mask(predictions, threshold)
    result = {"status": "ok", "all_reports": _errors(measured, reference),
              "retained_reports": _errors(measured[retained], reference[retained]),
              "retained_fraction": float(retained.mean()),
              "score_error_spearman": _rank(predictions.score, row_error)}
    if "reference_semantics" in predictions:
        result["reference_semantics"] = sorted(predictions.reference_semantics.dropna().unique().tolist())
    rows = []
    for key, group in predictions.groupby(ROUND, sort=True):
        row = dict(zip(ROUND, key))
        selected = group[retained_mask(group, threshold)]
        for value_col, truth_col in zip(cols, truth):
            observed_truth = group[truth_col].dropna().to_numpy(float)
            if not len(observed_truth) or not np.allclose(observed_truth, observed_truth[0]):
                continue  # spatially different references must not be averaged
            suffix = value_col[len("value_"):]
            row["reference_" + suffix] = float(observed_truth[0])
            row["all_median_" + suffix] = group[value_col].median()
            row["filtered_median_" + suffix] = selected[value_col].median()
            row["retained_sources"] = int(selected.source_id.nunique())
        if len(row) > len(ROUND):
            rows.append(row)
    estimates = pd.DataFrame(rows)
    if len(estimates):
        estimates.to_csv(Path(output) / "environment_estimates.csv", index=False)
        result["environment"] = {"method": "common_report_median_postprocessing", "task_rounds": len(estimates)}
        for col in cols:
            suffix = col[len("value_"):]
            reference_col = "reference_" + suffix
            if reference_col in estimates:
                result["environment"][suffix] = {
                    "all": _errors(estimates["all_median_" + suffix], estimates[reference_col]),
                    "filtered": _errors(estimates["filtered_median_" + suffix], estimates[reference_col])}
    return result


def evaluate_queries(predictions, threshold, query_path, output, config=None):
    """Compare a fixed inverse-distance estimator against held-out measurements.

    The query value is read only after scoring. The same estimator and spatial
    rule are used with all reports and with detector-retained reports. This is
    a downstream utility test, not a fitted condition-alignment algorithm.
    """
    config = config or {}
    queries = pd.read_csv(query_path, dtype={"sequence_id": str, "task_id": str, "query_id": str})
    required = ROUND + ["condition_x", "condition_y", "reference_0"]
    if not set(required).issubset(queries):
        raise ValueError("Query sidecar must contain " + str(required))
    if not pd.api.types.is_numeric_dtype(queries.timestamp):
        queries["timestamp"] = pd.to_datetime(queries.timestamp, utc=True, errors="raise").astype("int64") / 1e9
    if not np.isfinite(queries.timestamp).all():
        raise ValueError("Query timestamps must be finite")
    if not {"condition_x", "condition_y", "value_0"}.issubset(predictions):
        raise ValueError("Spatial query evaluation requires report x/y and value_0")
    if "raw_node_id" in queries and "raw_node_id" in predictions:
        if set(queries.raw_node_id) & set(predictions.raw_node_id):
            raise ValueError("Query sensors must be held out from submitted reports")
    minimum = int(config.get("query_min_sources", 3))
    neighbors = int(config.get("query_neighbors", 8))
    radius = config.get("query_radius")
    if minimum < 1 or neighbors < minimum or (radius is not None and float(radius) <= 0):
        raise ValueError("Invalid query neighborhood")
    groups = {key: group for key, group in predictions.groupby(ROUND, sort=False)}
    rows = []
    for _, query in queries.iterrows():
        key = tuple(query[c] for c in ROUND)
        if key not in groups:
            continue  # never evaluate other splits using an absent report round
        group = groups[key]
        distance = np.hypot(group.condition_x - query.condition_x, group.condition_y - query.condition_y)
        valid = np.isfinite(group.value_0) & np.isfinite(distance)
        if radius is not None:
            valid &= distance <= float(radius)
        eligible = group.loc[valid].copy()
        eligible["distance"] = distance.loc[valid]
        eligible = eligible.sort_values(["distance", "source_id"], kind="stable").head(neighbors)
        row = query.to_dict()
        for name, chosen in (("all", eligible), ("filtered", eligible[retained_mask(eligible, threshold)])):
            row[name + "_sources"] = int(chosen.source_id.nunique())
            row[name + "_estimate"] = np.nan
            if row[name + "_sources"] >= minimum:
                weights = 1 / np.maximum(chosen.distance.to_numpy(float), 1e-6)
                row[name + "_estimate"] = float(np.average(chosen.value_0, weights=weights))
        rows.append(row)
    estimates = pd.DataFrame(rows)
    estimates.to_csv(Path(output) / "query_estimates.csv", index=False)
    result = {"method": "fixed_inverse_distance_common_postprocessing", "query_rounds": len(estimates),
              "reference_semantics": "held_out_measured_temperature_proxy", "query_sources_excluded": True}
    if len(estimates):
        for name in ("all", "filtered"):
            result[name] = _errors(estimates[name + "_estimate"], estimates.reference_0)
            result[name]["coverage"] = result[name]["n"] / len(estimates)
    return result
