"""Task-sampled Label-Trim calibration and its exact upper-tail p-values.

Sampling uses report identities before inspecting score availability. The limited
review budget is then spent on the largest sampled scores; only a verified
anomaly is removed. The score pool is stored directly in the calibration artifact.

Reference: Bashari, Sesia and Romano (ICML 2025), Algorithms 1–2:
https://proceedings.mlr.press/v267/bashari25a.html
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


_REPORT_KEYS = ["sequence_id", "task_id", "timestamp", "source_id"]
_UNIT_KEYS = ["sequence_id", "task_id", "timestamp"]


def _seconds(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(value, utc=True, errors="raise")
        result = float(parsed.value) / 1e9
    if not np.isfinite(result):
        raise ValueError("Calibration timestamps must be finite")
    return result


def _identity(row, columns):
    values = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            raise ValueError("Calibration report identities must not contain missing values")
        if column == "timestamp":
            value = format(_seconds(value), ".17g")
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be an integer".format(name))
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{} must be an integer".format(name))
    if converted != value:
        raise ValueError("{} must be an integer".format(name))
    return converted


def _alpha(config):
    alpha = float(config.get("alpha", 0.05))
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Label-Trim alpha must be in (0, 1)")
    return alpha


def _budget_limit(alpha, n):
    # Use the same floating-point comparison as p <= alpha at an integer boundary.
    maximum = int(math.floor(alpha * (n + 1)))
    while maximum > 0 and maximum / (n + 1) > alpha:
        maximum -= 1
    while maximum < n and (maximum + 1) / (n + 1) <= alpha:
        maximum += 1
    return maximum


def _review_label(value):
    if pd.isna(value):
        return -1
    try:
        label = float(value)
    except (TypeError, ValueError):
        raise ValueError("Reviewed labels must be 0 (normal), 1 (anomaly), or -1 (unknown)")
    if label not in (-1, 0, 1):
        raise ValueError("Reviewed labels must be 0 (normal), 1 (anomaly), or -1 (unknown)")
    return int(label)


def _external_reviews(config, report_columns):
    path = config.get("external_verification_csv", config.get("verification_csv"))
    if path is None:
        raise ValueError("External verification requires external_verification_csv")
    frame = pd.read_csv(Path(path), dtype={key: str for key in report_columns if key != "timestamp"})
    missing = set(report_columns + ["verified_label"]) - set(frame.columns)
    if missing:
        raise ValueError("External verification is missing columns: {}".format(sorted(missing)))
    reviews = {}
    for _, row in frame.iterrows():
        key = _identity(row, report_columns)
        if key in reviews:
            raise ValueError("External verification contains duplicate report identities")
        reviews[key] = row["verified_label"]
    return reviews


def fit_label_trim(scored_frame, config=None):
    """Return ``(calibration_dict, audit_frame)`` from all calibration reports.

    ``scored_frame`` contains the report registry, including unscored reports
    with NaN scores. A SHA256 draw chooses one report in each ``unit_columns``
    group. Optional ``separation_seconds`` thins those units chronologically
    within ``separation_group_columns`` (by default sequence and task).

    ``verification`` is ``none``, ``injected_fault_oracle``, or ``external``.
    External review CSV rows use the report keys and ``verified_label``. Missing
    reviews remain unknown. In oracle mode only the selected review candidates'
    ``injection_label`` values are read; natural ``label`` is never used.
    """
    config = dict(config or {})
    alpha = _alpha(config)
    mode = config.get("verification", config.get("verification_mode", "none"))
    if mode not in ("none", "injected_fault_oracle", "external"):
        raise ValueError("Unknown Label-Trim verification mode: {}".format(mode))
    units = list(config.get("unit_columns", _UNIT_KEYS))
    report_columns = list(config.get("report_columns", _REPORT_KEYS))
    if not units or len(set(units)) != len(units):
        raise ValueError("unit_columns must contain distinct identity columns")
    if not report_columns or len(set(report_columns)) != len(report_columns):
        raise ValueError("report_columns must contain distinct identity columns")
    if not set(units).issubset(report_columns):
        raise ValueError("unit_columns must be included in report_columns")
    forbidden = {"score", "label", "injection_label", "verified_label", "original_label"}
    if forbidden.intersection(report_columns):
        raise ValueError("Sampling keys must be report identities, not scores or labels")
    seed = _integer(config.get("selection_seed", 42), "selection_seed")
    separation = float(config.get("separation_seconds", 0.0))
    if not np.isfinite(separation) or separation < 0:
        raise ValueError("separation_seconds must be finite and nonnegative")
    groups = list(config.get("separation_group_columns", [key for key in units if key != "timestamp"]))
    if not set(groups).issubset(units):
        raise ValueError("separation_group_columns must be a subset of unit_columns")
    if separation and "timestamp" not in report_columns:
        raise ValueError("Time-separated sampling requires timestamp in report_columns")
    missing = set(report_columns + ["score"]) - set(scored_frame.columns)
    if missing:
        raise ValueError("Calibration reports are missing columns: {}".format(sorted(missing)))

    # Identity selection intentionally precedes score conversion, validation,
    # availability filtering, and all access to verification labels.
    chosen = {}
    seen = set()
    for position, (_, row) in enumerate(scored_frame[report_columns].iterrows()):
        key = _identity(row, report_columns)
        if key in seen:
            raise ValueError("Calibration contains duplicate report identities")
        seen.add(key)
        unit = _identity(row, units)
        payload = json.dumps([seed, list(key)], ensure_ascii=False, separators=(",", ":"))
        draw = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        order = (draw, key)
        if unit not in chosen or order < chosen[unit][0]:
            chosen[unit] = (order, position)

    records = []
    for unit, (_, position) in sorted(chosen.items()):
        row = scored_frame.iloc[position]
        record = {key: row[key] for key in report_columns}
        record.update({"selection_hash": chosen[unit][0][0], "sampled": True,
                       "score": np.nan, "finite_score": False, "reviewed": False,
                       "review_rank": -1, "verified_label": -1, "removed": False,
                       "retained": False, "status": "pending", "_position": position,
                       "_identity": _identity(row, report_columns),
                       "_group": _identity(row, groups),
                       "_time": _seconds(row["timestamp"]) if "timestamp" in report_columns else 0.0})
        records.append(record)
    records.sort(key=lambda row: (row["_group"], row["_time"], row["_identity"]))
    last_time = {}
    for record in records:
        group, timestamp = record["_group"], record["_time"]
        if separation and group in last_time and timestamp - last_time[group] < separation:
            record.update(sampled=False, status="time_separation")
            continue
        last_time[group] = timestamp
        value = scored_frame.iloc[record["_position"]]["score"]
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = np.nan
        finite = bool(np.isfinite(score))
        record.update(score=score if finite else np.nan, finite_score=finite,
                      retained=finite, status="retained" if finite else "score_unavailable")

    finite_records = [record for record in records if record["sampled"] and record["finite_score"]]
    n = len(finite_records)
    limit = _budget_limit(alpha, n)
    requested = config.get("budget", "auto")
    if requested is None or requested == "auto":
        budget = limit
    else:
        budget = _integer(requested, "Label-Trim budget")
        if budget < 0:
            raise ValueError("Label-Trim budget must be nonnegative")
    if mode == "none":
        budget = 0
    elif budget > limit:
        raise ValueError("Label-Trim budget {} exceeds floor(alpha * (n + 1)) = {}".format(budget, limit))
    candidates = sorted(finite_records, key=lambda row: (-row["score"], row["_identity"]))[:budget]
    reviews = _external_reviews(config, report_columns) if mode == "external" and budget else {}
    if mode == "injected_fault_oracle" and budget and "injection_label" not in scored_frame:
        raise ValueError("injected_fault_oracle requires injection_label for review candidates")
    for rank, record in enumerate(candidates, start=1):
        if mode == "injected_fault_oracle":
            value = scored_frame.iloc[record["_position"]]["injection_label"]
        else:
            value = reviews.get(record["_identity"], -1)
        label = _review_label(value)
        removed = label == 1
        record.update(reviewed=True, review_rank=rank, verified_label=label,
                      removed=removed, retained=not removed,
                      status="verified_anomaly" if removed else "reviewed_normal" if label == 0 else "review_unknown")

    pool = sorted(float(record["score"]) for record in finite_records if record["retained"])
    removed = sum(record["removed"] for record in records)
    result = {"method": "label_trim", "version": 1, "alpha": alpha,
              "verification": mode, "selection_seed": seed,
              "selection_rule": "minimum_sha256_of_seed_and_report_identity_before_score_filtering",
              "unit_columns": units, "report_columns": report_columns,
              "separation_seconds": separation, "separation_group_columns": groups,
              "candidate_tasks": len(records), "sampled_tasks": sum(row["sampled"] for row in records),
              "unavailable_selected_tasks": sum(row["status"] == "score_unavailable" for row in records),
              "n": n, "m": budget, "budget_limit": limit, "removed": int(removed),
              "retained": len(pool), "unknown_reviews": sum(row["reviewed"] and row["verified_label"] == -1 for row in records),
              "min_p": 1.0 / (len(pool) + 1), "score_pool": pool,
              "pvalue_comparison": "calibration_score >= test_score",
              "decision_rule": "p_value <= alpha", "status": "ready" if pool else "empty_score_pool"}
    result["threshold"] = label_trim_threshold(result)
    result["alarms_attainable"] = result["threshold"] is not None
    columns = report_columns + ["selection_hash", "sampled", "score", "finite_score", "reviewed",
                                "review_rank", "verified_label", "removed", "retained", "status"]
    return result, pd.DataFrame.from_records(records, columns=columns)


def _pool(calibration):
    pool = np.asarray(calibration["score_pool"], dtype=float)
    if pool.ndim != 1 or not np.isfinite(pool).all():
        raise ValueError("Saved Label-Trim score_pool must be a finite one-dimensional array")
    return np.sort(pool)


def label_trim_pvalues(scores, calibration):
    """Compute (1 + number of retained scores >= s) / (N + 1)."""
    pool = _pool(calibration)
    values = np.asarray(scores, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    result[valid] = (1 + len(pool) - np.searchsorted(pool, values[valid], side="left")) / (len(pool) + 1)
    return result


def label_trim_predictions(scores, calibration):
    """Boolean alarm decisions; unavailable scores have no alarm decision."""
    pvalues = label_trim_pvalues(scores, calibration)
    return np.isfinite(pvalues) & (pvalues <= _alpha(calibration))


def label_trim_threshold(calibration):
    """Return a strict ``score > threshold`` boundary, or None if unattainable.

    None represents an infinite threshold in JSON. Ties remain in the upper
    tail, matching the p-value definition exactly.
    """
    pool = _pool(calibration)
    maximum = _budget_limit(_alpha(calibration), len(pool))
    if maximum < 1 or not len(pool):
        return None
    return float(pool[len(pool) - maximum])
