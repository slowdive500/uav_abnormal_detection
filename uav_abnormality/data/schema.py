"""Canonical contribution table, with strict provenance-preserving validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

KEYS = ["sequence_id", "task_id", "timestamp", "source_id"]

def feature_columns(frame, prefix="value_"):
    cols = [str(c) for c in frame.columns if str(c).startswith(prefix)]
    return sorted(cols, key=lambda c: (0, int(c[len(prefix):])) if c[len(prefix):].isdigit() else (1, c))

def validate_table(frame):
    frame = frame.copy()
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError("Missing contribution keys: %s" % sorted(missing))
    values = feature_columns(frame)
    if not values or not len(frame):
        raise ValueError("Need nonempty reports with value_0 ... value_F columns")
    for col in ("sequence_id", "task_id", "source_id"):
        if frame[col].isna().any():
            raise ValueError("Null identity in " + col)
        frame[col] = frame[col].astype(str)
    if frame.timestamp.isna().any():
        raise ValueError("Timestamps cannot be null")
    if pd.api.types.is_numeric_dtype(frame.timestamp):
        frame["timestamp"] = pd.to_numeric(frame.timestamp).astype(float)
    else:
        stamp = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        if stamp.isna().any():
            raise ValueError("Timestamps cannot be NaT")
        frame["timestamp"] = stamp.astype("int64") / 1e9
    if not np.isfinite(frame.timestamp).all():
        raise ValueError("Timestamps must be finite UTC seconds")
    for col in values:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col]).any():
            raise ValueError("Infinite observed values: " + col)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate source reports for a task/time. Aggregate explicitly in the adapter.")
    if "label" not in frame:
        frame["label"] = -1
    frame["label"] = pd.to_numeric(frame.label, errors="raise").fillna(-1)
    if not frame.label.isin([-1, 0, 1]).all():
        raise ValueError("label must be -1 unknown, 0 normal, or 1 anomalous")
    frame["label"] = frame.label.astype(np.int8)
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)

def read_table(path):
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        table = pd.read_json(path, lines=True, dtype={"source_id": str, "task_id": str, "sequence_id": str})
    else:
        table = pd.read_csv(path, dtype={"source_id": str, "task_id": str, "sequence_id": str})
    return validate_table(table)

def coverage_report(frame):
    values = feature_columns(frame)
    counts = frame[frame[values].notna().any(axis=1)].groupby(["sequence_id", "task_id", "timestamp"]).source_id.nunique()
    return {"reports": len(frame), "sequences": int(frame.sequence_id.nunique()),
            "tasks": int(frame.task_id.nunique()), "physical_sources": sorted(frame.source_id.unique().tolist()),
            "features": values, "unknown_labels": int((frame.label == -1).sum()),
            "observed_fraction": float(frame[values].notna().to_numpy().mean()),
            "simultaneous_sources_histogram": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
            "rounds_with_at_least_3_sources": int((counts >= 3).sum())}

def write_table(frame, path, provenance=None):
    frame = validate_table(frame)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    meta = coverage_report(frame)
    meta["provenance"] = provenance or {}
    path.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
