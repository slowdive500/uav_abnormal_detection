"""Disjoint splits, training-only normalization and causal windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .data.schema import feature_columns, KEYS

SPLITS = ("train", "validation", "calibration", "test")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def split_table(frame, config):
    ratios = np.asarray(config.get("ratios", [0.6, 0.15, 0.1, 0.15]), float)
    if ratios.shape != (4,) or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("split.ratios must be four positive values summing to one")
    mode = config.get("mode", "sequence")
    if mode == "sequence":
        # Scene/day membership is immutable across all four splits.
        units = frame.groupby("sequence_id").timestamp.min().sort_values(kind="stable").index.to_numpy()
        membership = frame.sequence_id
    elif mode == "chronological":
        units = np.sort(frame.timestamp.unique())
        membership = frame.timestamp
    else:
        raise ValueError("split.mode must be sequence or chronological")
    if len(units) < 4:
        raise ValueError("Need at least four split units; use more scenes/days, or explicit chronological split")
    counts = np.maximum(1, np.floor(ratios * len(units)).astype(int))
    while counts.sum() > len(units):
        eligible = np.where(counts > 1, counts, -1)
        counts[int(np.argmax(eligible))] -= 1
    while counts.sum() < len(units):
        counts[int(np.argmax(ratios * len(units) - counts))] += 1
    result, manifest, offset = {}, {}, 0
    for name, count in zip(SPLITS, counts):
        chosen = units[offset:offset + count]
        subset = frame[membership.isin(chosen)].copy().reset_index(drop=True)
        result[name] = subset
        manifest[name] = {"reports": len(subset), "sequence_ids": sorted(subset.sequence_id.unique().tolist()),
                          "time_min": float(subset.timestamp.min()), "time_max": float(subset.timestamp.max())}
        offset += count
    # Windows are built AFTER splitting; there is no shared context across boundaries.
    return result, {"mode": mode, "splits": manifest, "window_overlap_between_splits": False}

class Standardizer:
    def fit(self, frame):
        self.columns = feature_columns(frame)
        values = frame[self.columns].to_numpy(float)
        if np.any(np.isfinite(values).sum(axis=0) == 0):
            raise ValueError("Each feature requires observed training data")
        self.mean = np.nanmean(values, axis=0)
        self.scale = np.nanstd(values, axis=0)
        self.scale = np.where(self.scale > 1e-6, self.scale, 1.)
        return self

    def transform(self, values):
        return (values - self.mean) / self.scale

    def to_dict(self):
        return {"columns": self.columns, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.columns = list(data["columns"])
        obj.mean, obj.scale = np.asarray(data["mean"]), np.asarray(data["scale"])
        return obj

class Windows(Dataset):
    def __init__(self, frame, sources, scaler, window_size, min_sources=2, max_gap_seconds=None, stride=1,
                 allow_empty=False):
        if window_size < 2 or min_sources < 1 or stride < 1:
            raise ValueError("window_size>=2, min_sources>=1, stride>=1 required")
        unknown = set(frame.source_id) - set(sources)
        if unknown:
            raise ValueError("Unseen physical sources require a new trained source vocabulary: %s" % sorted(unknown))
        self.sources, self.scaler, self.window_size = list(sources), scaler, window_size
        self.max_gap_seconds = max_gap_seconds
        self.frame = frame.reset_index(drop=True)
        self.groups, self.index = [], []
        source_index = {s: i for i, s in enumerate(sources)}
        for (sequence, task), group in self.frame.groupby(["sequence_id", "task_id"], sort=True):
            times = np.sort(group.timestamp.unique())
            ti = {t: i for i, t in enumerate(times)}
            shape = (len(times), len(sources), len(scaler.columns))
            values = np.full(shape, np.nan, dtype=np.float32)
            row_ids = np.full(shape[:2], -1, dtype=np.int64)
            for rid, row in group.iterrows():
                a, b = ti[row.timestamp], source_index[row.source_id]
                values[a, b] = row[scaler.columns].to_numpy(float)
                row_ids[a, b] = rid
            mask = np.isfinite(values)
            data = np.where(mask, scaler.transform(values), 0).astype(np.float32)
            # Only the current round needs a quorum; missing history remains masked.
            quorum = mask.any(-1).sum(-1) >= min_sources
            gid = len(self.groups)
            self.groups.append({"key": (str(sequence), str(task)), "times": times,
                                "x": data, "mask": mask, "rows": row_ids})
            for end in range(window_size - 1, len(times), stride):
                start = end - window_size + 1
                if not quorum[end]:
                    continue
                if max_gap_seconds is not None and np.any(np.diff(times[start:end + 1]) > max_gap_seconds):
                    continue
                self.index.append((gid, end))
        if not self.index and not allow_empty:
            raise ValueError("No eligible windows: inspect simultaneous-source coverage, window length, task bins and gap limits")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        gid, end = self.index[index]
        group = self.groups[gid]
        sl = slice(end - self.window_size + 1, end + 1)
        return {"x": torch.from_numpy(group["x"][sl]), "mask": torch.from_numpy(group["mask"][sl]),
                "index": index}

    def metadata(self, index):
        gid, end = self.index[index]
        group = self.groups[gid]
        return group["key"], group["times"][end], group["rows"][end]
