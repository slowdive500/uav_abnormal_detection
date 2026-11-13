"""Training-fitted acquisition conditions attached to causal report windows."""
import numpy as np
import torch
from .protocol import Windows


class ConditionScaler:
    def fit(self, frame, columns):
        self.columns = list(columns)
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError('Specify distinct method.condition_columns')
        if any(not name.startswith('condition_') for name in self.columns):
            raise ValueError('Condition inputs must be explicit condition_* columns')
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError('Missing acquisition conditions: ' + str(sorted(missing)))
        values = frame[self.columns].to_numpy(float)
        values = np.where(np.isfinite(values), values, np.nan)
        if np.any(np.isfinite(values).sum(0) == 0):
            raise ValueError('Each condition needs finite training observations')
        self.mean = np.nanmean(values, axis=0)
        self.scale = np.nanstd(values, axis=0)
        self.scale = np.where(self.scale > 1e-6, self.scale, 1.)
        return self

    def transform(self, values):
        return (np.asarray(values, float) - self.mean) / self.scale

    def to_dict(self):
        return {'columns': self.columns, 'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.columns = list(data['columns'])
        obj.mean = np.asarray(data['mean'], float)
        obj.scale = np.asarray(data['scale'], float)
        return obj


class ConditionWindows(Windows):
    """Source identities index reports; the condition model has no ID embedding."""
    def __init__(self, frame, sources, scaler, condition_scaler, **window):
        sources = sorted(set(sources) | set(frame.source_id))
        super().__init__(frame, sources, scaler, **window)
        self.condition_scaler = condition_scaler
        missing = set(condition_scaler.columns) - set(frame.columns)
        if missing:
            raise ValueError('Inference conditions missing: ' + str(sorted(missing)))
        values = frame.reset_index(drop=True)[condition_scaler.columns].to_numpy(float)
        for group in self.groups:
            rows = group['rows']
            valid_rows = rows >= 0
            conditions = np.zeros(rows.shape + (len(condition_scaler.columns),), dtype=np.float32)
            observed = np.zeros(rows.shape, dtype=bool)
            raw = values[rows[valid_rows]]
            finite = np.isfinite(raw).all(-1)
            normalized = condition_scaler.transform(raw)
            conditions[valid_rows] = np.where(np.isfinite(normalized), normalized, 0)
            observed[valid_rows] = finite
            group['conditions'], group['condition_mask'] = conditions, observed

    def __getitem__(self, index):
        item = super().__getitem__(index)
        gid, end = self.index[index]
        segment = slice(end - self.window_size + 1, end + 1)
        item['conditions'] = torch.from_numpy(self.groups[gid]['conditions'][segment])
        item['condition_mask'] = torch.from_numpy(self.groups[gid]['condition_mask'][segment])
        return item
