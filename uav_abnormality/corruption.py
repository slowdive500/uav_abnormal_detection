"""Reproducible report faults, applied after splitting; references stay private."""
import numpy as np
from .data.schema import feature_columns

KINDS = ("offset", "drift", "spike", "stuck", "replay", "replacement")


def inject_faults(frame, config, scales, seed, attack_profile=None):
    if config.get("protocol") == "targeted_report_attack":
        from .attacks import inject_targeted_attacks
        if attack_profile is None:
            raise ValueError("Targeted attacks require a shared training-fitted attack_profile")
        return inject_targeted_attacks(frame, config, attack_profile, seed)
    out = frame.copy()
    cols = feature_columns(frame)
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (len(cols),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Fault scales must be positive finite training-only feature scales")
    fraction = float(config.get("source_fraction", 0.2))
    rate = float(config.get("report_fraction", 0.15))
    magnitude = float(config.get("magnitude", 4.0))
    duration = int(config.get("duration", 8))
    kinds = config.get("kinds", list(KINDS))
    max_gap = config.get("max_gap_seconds")
    if not 0 <= fraction <= 1 or not 0 <= rate <= 1 or duration < 1 or not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("Invalid corruption fractions, duration or magnitude")
    if not kinds or not set(kinds).issubset(KINDS):
        raise ValueError("Unknown fault kinds")
    if max_gap is not None and (not np.isfinite(float(max_gap)) or float(max_gap) <= 0):
        raise ValueError("Fault max_gap_seconds must be positive")
    out["original_label"] = out.label
    for col in cols:
        out["baseline_" + col[len("value_"):]] = out[col]
    if (out.label == -1).any():
        if not config.get("assume_unlabelled_normal", False):
            raise ValueError("Semi-synthetic injection requires explicit assume_unlabelled_normal: true")
        out.loc[out.label == -1, "label"] = 0
    out["injection_label"] = 0
    out["fault_type"] = "none"
    out["fault_event_id"] = ""
    rng = np.random.RandomState(seed)
    selection_rng = np.random.RandomState(config.get("source_selection_seed", seed))
    sources = sorted(out.source_id.unique())
    number = 0 if fraction == 0 else max(1, int(round(len(sources) * fraction)))
    affected = set(selection_rng.choice(sources, size=number, replace=False))
    original = frame[cols].copy()
    events = []
    for (sequence, task, source), group in out.groupby(["sequence_id", "task_id", "source_id"], sort=True):
        if source not in affected or rate == 0:
            continue
        ordered = group.sort_values("timestamp")
        all_indices = ordered.index.to_numpy()
        breaks = np.where(np.diff(ordered.timestamp) > float(max_gap))[0] + 1 if max_gap is not None else []
        for indices in np.split(all_indices, breaks):
            target = max(1, int(round(len(indices) * rate)))
            selected = set()
            for start in rng.permutation(len(indices)):
                if len(selected) >= target:
                    break
                # Never bridge an already selected position to create a fake interval.
                positions = []
                for pos in range(start, min(start + duration, len(indices))):
                    if pos in selected or len(positions) >= target - len(selected):
                        break
                    positions.append(pos)
                if not positions:
                    continue
                kind = kinds[int(rng.randint(len(kinds)))]
                if kind in ("stuck", "replay", "replacement") and start == 0:
                    continue  # a causal historical fault needs a past observation
                base = original.loc[indices[positions]].to_numpy(float)
                shift = rng.choice([-1., 1.], size=len(cols)) * scales * magnitude
                if kind == "offset":
                    modified = base + shift
                elif kind == "drift":
                    modified = base + np.linspace(0.25, 1., len(positions))[:, None] * shift
                elif kind == "spike":
                    modified = base + rng.normal(size=base.shape) * shift
                elif kind == "stuck":
                    reference = original.loc[indices[start - 1]].to_numpy(float)
                    modified = np.broadcast_to(reference, base.shape).copy()
                elif kind == "replay":
                    # Replay a past block, rather than repeating a single stale value.
                    history = original.loc[indices[max(0, start - duration):start]].to_numpy(float)
                    modified = history[np.arange(len(positions)) % len(history)].copy()
                else:
                    # Replace a segment with another source's *past* reports on the
                    # same task. Never use the test future or cross a split boundary.
                    donor = frame[(frame.sequence_id == sequence) & (frame.task_id == task) &
                                  (frame.source_id != source) &
                                  (frame.timestamp < float(out.loc[indices[start], "timestamp"]))]
                    if donor.empty:
                        continue
                    donor_source = sorted(donor.source_id.unique())[int(rng.randint(donor.source_id.nunique()))]
                    history = donor[donor.source_id == donor_source].sort_values("timestamp").tail(duration)[cols].to_numpy(float)
                    modified = history[np.arange(len(positions)) % len(history)].copy()
                modified[~np.isfinite(base)] = np.nan
                modified = np.where(np.isfinite(base) & ~np.isfinite(modified), base, modified)
                changed = np.any(np.isfinite(base) & np.isfinite(modified) & (np.abs(modified - base) > 1e-12), axis=1)
                out.loc[indices[positions], cols] = modified
                chosen = indices[np.asarray(positions)[changed]]
                if len(chosen):
                    event_id = "%d:%d" % (seed, len(events))
                    out.loc[chosen, "label"] = 1
                    out.loc[chosen, "injection_label"] = 1
                    out.loc[chosen, "fault_type"] = kind
                    out.loc[chosen, "fault_event_id"] = event_id
                    events.append({"event_id": event_id, "sequence_id": str(sequence), "task_id": str(task),
                                   "source_id": str(source), "kind": kind, "changed_reports": len(chosen),
                                   "time_start": float(out.loc[chosen, "timestamp"].min()),
                                   "time_end": float(out.loc[chosen, "timestamp"].max()),
                                   "magnitude_in_training_std": magnitude if kind in ("offset", "drift", "spike") else None})
                selected.update(positions)
    return out, {"protocol": "semi_synthetic", "seed": int(seed), "affected_sources": sorted(affected),
                 "injected_reports": int(out.injection_label.sum()), "events": events,
                 "baseline_assumption": "Unlabelled baseline assumed normal only for injected-fault evaluation; natural faults remain unverified", "config": config}
