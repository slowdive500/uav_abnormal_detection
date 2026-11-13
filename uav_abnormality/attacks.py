"""Causal, bounded report attacks with independently scheduled attempts.

The generator reads identities, times and original ``value_*`` observations.
Labels/references are copied for compatibility but never select attacks or values.
An attack label describes a modified report, not successful evasion of a detector.
"""
import copy
from collections import Counter

import numpy as np

from .data.schema import feature_columns


PROTOCOL = "targeted_report_attack"
ATTACK_KINDS = ("target_bias", "change_suppression", "false_trend")


def _persistence_settings(config):
    persistence = config.get("persistence", "bounded")
    if persistence not in ("bounded", "sustained"):
        raise ValueError("persistence must be bounded or sustained")
    offset = config.get("start_offset_rounds", 0)
    if isinstance(offset, bool) or int(offset) != offset or offset < 0:
        raise ValueError("start_offset_rounds must be a nonnegative integer")
    selection = config.get("source_selection_mode", "fixed_coalition")
    if selection not in ("fixed_coalition", "present_compromised_pool"):
        raise ValueError("source_selection_mode must be fixed_coalition or present_compromised_pool")
    if selection == "present_compromised_pool" and persistence != "sustained":
        raise ValueError("present_compromised_pool requires persistence=sustained")
    return persistence, int(offset), selection


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def _settings(config):
    _persistence_settings(config)
    if config.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("Expected protocol=" + PROTOCOL)
    kinds = list(config.get("kinds", ATTACK_KINDS))
    if not kinds or len(set(kinds)) != len(kinds) or not set(kinds).issubset(ATTACK_KINDS):
        raise ValueError("Specify distinct targeted attack kinds")
    sizes = list(config.get("coalition_sizes", [1, 2]))
    if not sizes or len(set(sizes)) != len(sizes) or any(
            isinstance(n, bool) or int(n) != n or n < 1 for n in sizes):
        raise ValueError("coalition_sizes must contain distinct positive integers")
    fraction = float(config.get("source_fraction", .25))
    rate = float(config.get("report_fraction", .3))
    local = float(config.get("max_local_fraction", .49))
    if not 0 <= fraction <= 1 or not 0 <= rate <= 1 or not 0 < local < .5:
        raise ValueError("Fractions must be in [0, 1]; max_local_fraction must be in (0, .5)")
    duration = config.get("duration", 8)
    if isinstance(duration, bool) or int(duration) != duration or duration < 1:
        raise ValueError("duration must be a positive integer")
    gap = config.get("max_gap_seconds")
    if gap is not None:
        gap = _positive(gap, "max_gap_seconds")
    return kinds, [int(n) for n in sizes], fraction, rate, local, int(duration), gap


def fit_attack_profile(train, config):
    """Fit immutable scales and source selection using only training reports.

    ``constraints`` accepts target_delta_sigma (.75), max_delta_sigma (1),
    max_delta_step_sigma (.25), min_effect_sigma (.1), optional
    max_report_step_sigma, and hard_bounds={value_column: [lower, upper]}.
    A None endpoint means unbounded. All sigma limits use the original training
    population standard deviation, floored at 1e-6 for constant features.
    """
    _, _, fraction, _, _, _, _ = _settings(config)
    columns = feature_columns(train)
    if not columns or not len(train):
        raise ValueError("An attack profile needs nonempty training observations")
    values = train[columns].to_numpy(dtype=float)
    finite = np.isfinite(values)
    counts = finite.sum(0)
    if (counts == 0).any():
        raise ValueError("Each attack feature needs finite training observations")
    scales = np.asarray([max(float(np.std(values[finite[:, i], i])), 1e-6)
                         for i in range(len(columns))])
    constraints = copy.deepcopy(config.get("constraints", {}))
    defaults = {"target_delta_sigma": .75, "max_delta_sigma": 1.,
                "max_delta_step_sigma": .25, "min_effect_sigma": .1}
    for key, default in defaults.items():
        constraints[key] = _positive(constraints.get(key, default), key)
    if not constraints["min_effect_sigma"] <= constraints["target_delta_sigma"] <= constraints["max_delta_sigma"]:
        raise ValueError("Require min_effect_sigma <= target_delta_sigma <= max_delta_sigma")
    report_step = constraints.get("max_report_step_sigma")
    constraints["max_report_step_sigma"] = None if report_step is None else _positive(report_step, "max_report_step_sigma")
    bounds = constraints.get("hard_bounds", {})
    if not isinstance(bounds, dict) or set(bounds) - set(columns):
        raise ValueError("hard_bounds must map existing value columns to [lower, upper]")
    normalized_bounds = {}
    for column in columns:
        bound = bounds.get(column, [None, None])
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise ValueError("Each hard bound must contain lower and upper endpoints")
        bound = [None if value is None else float(value) for value in bound]
        if any(value is not None and not np.isfinite(value) for value in bound):
            raise ValueError("Use None for an unbounded endpoint")
        if bound[0] is not None and bound[1] is not None and bound[0] >= bound[1]:
            raise ValueError("A lower hard bound must be below its upper bound")
        normalized_bounds[column] = bound
    constraints["hard_bounds"] = normalized_bounds
    sources = sorted(str(source) for source in train.source_id.unique())
    selection_seed = int(config.get("source_selection_seed", 42))
    count = 0 if fraction == 0 else max(1, int(round(len(sources) * fraction)))
    rng = np.random.RandomState(selection_seed)
    affected = sorted(rng.choice(sources, size=count, replace=False).tolist())
    return {"protocol": PROTOCOL, "version": 1, "feature_columns": columns,
            "scales": scales.tolist(), "training_count": int(len(train)),
            "finite_training_counts": counts.astype(int).tolist(),
            "training_sources": sources, "affected_sources": affected,
            "source_selection_seed": selection_seed, "constraints": constraints,
            "fit_scope": "original_training_reports_only"}


def _candidate_blocks(length, duration):
    """Calendar partition; never shorten a complete event to fit a budget."""
    blocks = [list(range(start, min(start + duration, length)))
              for start in range(0, length, duration)]
    if len(blocks) > 1 and len(blocks[-1]) == 1 and len(blocks[-2]) > 2:
        blocks[-1].insert(0, blocks[-2].pop())
    # A lone trailing round cannot support recovery. A one-round segment still
    # gets an explicit infeasible attempt, rather than disappearing from audit.
    return [block for block in blocks if len(block) >= 2 or length == 1 or duration == 1]


def _blocks(length, rate, duration, rng):
    """Round the target budget upward to complete calendar-only events."""
    target = 0 if rate == 0 else max(1, int(round(length * rate)))
    candidates = _candidate_blocks(length, duration)
    chosen, rounds = [], 0
    for index in rng.permutation(len(candidates)):
        if rounds >= target:
            break
        block = candidates[int(index)]
        chosen.append(block)
        rounds += len(block)
    return sorted(chosen, key=lambda block: block[0])


def _scheduled_blocks(length, rate, duration, rng, persistence, offset):
    """Schedule from the calendar without requiring future values/attendance.

    Sustained attacks occupy one continuous suffix, rather than repeatedly
    restarting their state at each ``duration`` boundary. The round budget is
    computed after the caller's causal warmup. At rate=1 this means every round
    after warmup, including the last observed round, can be attacked.
    """
    available = max(0, length - offset)
    if not available:
        return [], []
    if persistence == "sustained":
        target = 0 if rate == 0 else max(1, int(round(available * rate)))
        blocks = [list(range(length - target, length))] if target else []
        return blocks, []
    blocks = [[position + offset for position in block]
              for block in _blocks(available, rate, duration, rng)]
    candidates = [[position + offset for position in block]
                  for block in _candidate_blocks(available, duration)]
    return blocks, candidates


def inject_targeted_attacks(frame, config, profile, seed):
    """Return a copied report table and a JSON-serializable attempt audit.

    Duration includes recovery: the final scheduled round has zero perturbation.
    Core rounds occupy the first half (at least one round). Perturbation bounds
    shrink with remaining rounds, so complete events return smoothly to zero.
    Missing/illegal originals and a lost local minority can force an abort; these
    interruptions are explicit failed events, not silently repaired observations.

    Opt-in ``persistence=sustained`` uses a continuous budgeted suffix after
    ``start_offset_rounds`` of causal warmup, with no forced recovery endpoint.
    An absent selected source retains its state and can resume when it next
    submits a real report; no source is substituted and no report is created.
    With ``source_selection_mode=present_compromised_pool``, a fixed identity
    priority selects up to the requested coalition size from currently present
    members of the training-frozen compromised pool, subject to the actual
    local minority. State belongs to each source. Actual normal submissions
    reset that source's perturbation; skipped rounds can be retried. Event
    ``source_ids`` then records all pool members with attempted reports, while
    ``max_active_coalition_size`` records simultaneous participation.
    """
    kinds, sizes, _, rate, local_fraction, duration, max_gap = _settings(config)
    persistence, start_offset, selection_mode = _persistence_settings(config)
    sustained = persistence == "sustained"
    dynamic_pool = selection_mode == "present_compromised_pool"
    columns = feature_columns(frame)
    if profile.get("protocol") != PROTOCOL or columns != profile.get("feature_columns"):
        raise ValueError("Attack profile and report features do not match")
    constraints = profile["constraints"]
    scales = np.asarray(profile["scales"], dtype=float)
    if scales.shape != (len(columns),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Attack profile scales must be positive finite feature scales")
    max_delta = constraints["max_delta_sigma"] * scales
    delta_step = constraints["max_delta_step_sigma"] * scales
    target_delta = constraints["target_delta_sigma"] * scales
    minimum = float(constraints["min_effect_sigma"])
    report_step = constraints.get("max_report_step_sigma")
    report_step = None if report_step is None else float(report_step) * scales
    lower = np.asarray([constraints["hard_bounds"][c][0] if constraints["hard_bounds"][c][0] is not None else -np.inf for c in columns])
    upper = np.asarray([constraints["hard_bounds"][c][1] if constraints["hard_bounds"][c][1] is not None else np.inf for c in columns])
    out = frame.copy(deep=True)
    original = frame[columns].to_numpy(dtype=float, copy=True)
    values = original.copy()
    n = len(frame)
    audit = {key: np.full(n, default, dtype=dtype) for key, default, dtype in [
        ("attack_id", "", object), ("attack_type", "none", object),
        ("attack_attempted", False, bool), ("attack_feasible", False, bool),
        ("attack_direction", 0, int), ("attack_infeasible_reason", "", object),
        ("attack_phase", "none", object), ("attack_material_effect", False, bool),
        ("attack_forced_abort", False, bool), ("attack_effect_sigma", 0., float)]}
    out["original_label"] = frame["label"] if "label" in frame else -1
    # Original labels are compatibility/audit fields, never generator inputs.
    out["label"] = out.original_label
    if config.get("assume_unlabelled_normal", False):
        out.loc[out.original_label == -1, "label"] = 0
    for column in columns:
        out["baseline_" + column[len("value_"):]] = frame[column]
    out["injection_label"] = 0
    out["fault_type"] = "none"
    out["fault_event_id"] = ""
    affected = set(profile["affected_sources"])
    training_sources = set(profile["training_sources"])
    out["attack_controlled_source"] = frame.source_id.astype(str).isin(affected)
    rng = np.random.RandomState(seed)
    # Objective coverage comes first; do not spend the first two attempts on the
    # same objective just because single/coordinated scenarios were shuffled.
    ordered_kinds = [kinds[int(i)] for i in rng.permutation(len(kinds))]
    combinations = [(kind, size) for size in sorted(sizes) for kind in ordered_kinds]
    events = []
    planned_rounds = 0
    calendar = frame[["sequence_id", "task_id", "timestamp", "source_id"]].copy()
    calendar["_position"] = np.arange(n)
    minimum_attempts = config.get("min_attempts", len(kinds))
    if isinstance(minimum_attempts, bool) or int(minimum_attempts) != minimum_attempts or minimum_attempts < 0:
        raise ValueError("min_attempts must be a nonnegative integer")
    minimum_attempts = int(minimum_attempts) if rate > 0 else 0
    plans, extra_candidates = {}, []
    total_rounds, requested_rounds = 0, 0
    for (sequence, task), group in calendar.groupby(["sequence_id", "task_id"], sort=True):
        times = np.sort(group.timestamp.unique())
        breaks = np.where(np.diff(times) > max_gap)[0] + 1 if max_gap is not None else []
        for segment in np.split(times, breaks):
            if not len(segment):
                continue
            key = (str(sequence), str(task), float(segment[0]))
            plans[key], candidates = _scheduled_blocks(
                len(segment), rate, duration, rng, persistence, start_offset)
            total_rounds += len(segment)
            available = max(0, len(segment) - start_offset)
            requested_rounds += 0 if rate == 0 or not available else max(1, int(round(available * rate)))
            selected = {block[0] for block in plans[key]}
            for block in candidates:
                if block[0] not in selected:
                    extra_candidates.append((key, block))
    planned_count = sum(len(blocks) for blocks in plans.values())
    minimum_extras = 0
    for index in rng.permutation(len(extra_candidates)):
        if planned_count >= minimum_attempts:
            break
        key, block = extra_candidates[int(index)]
        plans[key].append(block)
        plans[key].sort(key=lambda item: item[0])
        planned_count += 1
        minimum_extras += 1
    for (sequence, task), group in calendar.groupby(["sequence_id", "task_id"], sort=True):
        times = np.sort(group.timestamp.unique())
        rounds = {}
        for timestamp, rows in group.groupby("timestamp", sort=True):
            rounds[float(timestamp)] = dict(zip(rows.source_id.astype(str), rows["_position"].astype(int)))
        breaks = np.where(np.diff(times) > max_gap)[0] + 1 if max_gap is not None else []
        histories = {}
        for segment in np.split(times, breaks):
            if not len(segment):
                continue
            histories.clear()  # A gap terminates causal history and attack state.
            blocks = plans[(str(sequence), str(task), float(segment[0]))]
            planned_rounds += sum(len(block) for block in blocks)
            starts = {block[0]: block for block in blocks}
            current = None
            states = {}
            for position, timestamp in enumerate(segment):
                timestamp = float(timestamp)
                rows = rounds[timestamp]
                if max_gap is not None:
                    histories = {source: record for source, record in histories.items()
                                 if timestamp - record[0] <= max_gap}
                if position in starts:
                    block = starts[position]
                    kind, requested = combinations[len(events) % len(combinations)]
                    present_controlled = sorted(set(rows) & affected)
                    chosen = [] if dynamic_pool else sorted(rng.choice(present_controlled,
                                    size=min(requested, len(present_controlled)), replace=False).tolist())
                    source_priority = (rng.permutation(sorted(affected)).tolist()
                                       if dynamic_pool else chosen)
                    direction = int(rng.choice([-1, 1]))
                    if kind == "change_suppression":
                        direction = 0  # Suppress either sign of genuine local change.
                    current = {"event_id": "%d:attack:%d" % (seed, len(events)),
                               "kind": kind, "sequence_id": str(sequence), "task_id": str(task),
                               "source_ids": chosen, "requested_coalition_size": requested,
                               "scenario": "single" if requested == 1 else "coordinated",
                               "direction": direction, "time_start": timestamp,
                               "time_end": float(segment[block[-1]]), "scheduled_rounds": len(block),
                               "core_rounds": len(block) if sustained else max(1, len(block) // 2),
                               "persistence": persistence, "source_selection_mode": selection_mode,
                               "max_active_coalition_size": 0, "attempted_reports": 0,
                               "changed_reports": 0, "feasible_reports": 0, "material_core_reports": 0,
                               "forced_abort": False, "reason_counts": {}, "feasible": False,
                               "_start": position, "_end": block[-1]}
                    events.append(current)
                    states = {source: {"delta": np.zeros(len(columns)), "aborted": False,
                              "anchor": histories[source][1].copy() if source in histories else None,
                              "output": histories[source][1].copy() if source in histories else None,
                              "ever_changed": False} for source in source_priority}
                if current is not None and position <= current["_end"]:
                    core = position - current["_start"] < current["core_rounds"]
                    phase = "core" if core else "recovery"
                    complete = sum(np.isfinite(original[row]).all() for row in rows.values())
                    capacity = int(np.floor(local_fraction * complete + 1e-12))
                    if dynamic_pool:
                        # The compromised pool was frozen on training data. A
                        # fixed random identity order resolves current-round
                        # participation without inspecting future attendance.
                        attempt_sources = [source for source in source_priority if source in rows]
                        selected_sources = attempt_sources[:min(current["requested_coalition_size"], capacity)]
                        current["source_ids"] = sorted(set(current["source_ids"]) | set(attempt_sources))
                    else:
                        attempt_sources = current["source_ids"]
                        selected_sources = attempt_sources
                    coalition_complete = dynamic_pool or (len(current["source_ids"]) == current["requested_coalition_size"]
                                          and (sustained or all(
                                              source in rows and np.isfinite(original[rows[source]]).all()
                                              for source in current["source_ids"])))
                    active_coalition = sum(source in rows and np.isfinite(original[rows[source]]).all()
                                           for source in selected_sources)
                    current["max_active_coalition_size"] = int(max(current["max_active_coalition_size"], active_coalition))
                    local_count = active_coalition if sustained else current["requested_coalition_size"]
                    local_ok = local_count <= capacity
                    shared_reason = ""
                    if not coalition_complete:
                        shared_reason = "coalition_unavailable"
                    elif not local_ok:
                        shared_reason = "local_minority_unavailable"
                    elif not sustained and current["scheduled_rounds"] < 2:
                        shared_reason = "insufficient_recovery_rounds"
                    elif current["kind"] == "false_trend" and current["core_rounds"] < 2:
                        shared_reason = "insufficient_trend_rounds"
                    reasons = Counter(current["reason_counts"])
                    if not attempt_sources:
                        reasons["no_controlled_source_present"] += 1
                    for source in attempt_sources:
                        state = states[source]
                        if source not in rows:
                            reasons["missing_report"] += 1
                            if sustained:
                                # Absence is not a submitted zero-delta report.
                                # Keep both perturbation and output history.
                                continue
                            if np.any(state["delta"]):
                                current["forced_abort"] = True
                            state["aborted"] = True
                            state["delta"] = np.zeros(len(columns))
                            continue
                        row = rows[source]
                        base = original[row]
                        current["attempted_reports"] += 1
                        for name, value in [("attack_id", current["event_id"]), ("attack_type", current["kind"]),
                                            ("attack_attempted", True), ("attack_direction", current["direction"]),
                                            ("attack_phase", phase)]:
                            audit[name][row] = value
                        reason = shared_reason
                        if dynamic_pool and source not in selected_sources:
                            reason = ("local_minority_unavailable" if capacity == 0
                                      else "coalition_budget_not_selected")
                        if sustained and state["anchor"] is None and source in histories:
                            state["anchor"] = histories[source][1].copy()
                            state["output"] = histories[source][1].copy()
                        if not np.isfinite(base).all():
                            reason = "missing_original"
                        elif np.any(base < lower) or np.any(base > upper):
                            reason = "original_outside_hard_bounds"
                        elif report_step is not None and source in histories and np.any(
                                np.abs(base - histories[source][1]) > report_step + 1e-12):
                            reason = "original_step_exceeds_limit"
                        elif state["aborted"] and not sustained:
                            reason = "event_previously_aborted"
                        elif current["kind"] != "target_bias" and state["anchor"] is None:
                            reason = "no_causal_history"
                        delta = np.zeros(len(columns))
                        if not reason:
                            remaining = current["_end"] - position
                            lo = np.maximum.reduce([-max_delta, state["delta"] - delta_step,
                                                    lower - base])
                            hi = np.minimum.reduce([max_delta, state["delta"] + delta_step,
                                                    upper - base])
                            if not sustained:
                                lo = np.maximum(lo, -remaining * delta_step)
                                hi = np.minimum(hi, remaining * delta_step)
                            if report_step is not None and state["output"] is not None:
                                lo = np.maximum(lo, state["output"] - report_step - base)
                                hi = np.minimum(hi, state["output"] + report_step - base)
                            if np.any(lo > hi + 1e-12):
                                reason = "constraints_incompatible"
                            else:
                                direction = current["direction"]
                                if not core:
                                    desired = np.zeros(len(columns))
                                elif current["kind"] == "target_bias":
                                    desired = direction * target_delta
                                elif current["kind"] == "change_suppression":
                                    desired = state["anchor"] - base
                                else:
                                    progress = (position - current["_start"] + 1.) / current["core_rounds"]
                                    desired = state["anchor"] + direction * target_delta * progress - base
                                delta = np.minimum(np.maximum(desired, lo), hi)
                                effect = float(np.max(np.abs(delta) / scales))
                                material = effect + 1e-12 >= minimum
                                if core and not material and not state["ever_changed"]:
                                    delta[:] = 0.
                                    reason = "below_min_effect"
                                elif not core and not state["ever_changed"]:
                                    reason = "no_material_core"
                                else:
                                    audit["attack_feasible"][row] = True
                                    audit["attack_material_effect"][row] = material
                                    current["feasible_reports"] += 1
                                    if core and material:
                                        current["material_core_reports"] += 1
                        if reason:
                            reasons[reason] += 1
                            audit["attack_infeasible_reason"][row] = reason
                            if reason not in ("below_min_effect", "no_material_core"):
                                if np.any(np.abs(state["delta"]) > (delta_step + 1e-12 if sustained else 0)):
                                    audit["attack_forced_abort"][row] = True
                                    current["forced_abort"] = True
                                state["aborted"] = not sustained
                            delta[:] = 0.
                        changed = bool(np.any(np.abs(delta) > 1e-12))
                        if changed:
                            values[row] = base + delta
                            current["changed_reports"] += 1
                            state["ever_changed"] = True
                        audit["attack_effect_sigma"][row] = float(np.max(np.abs(delta) / scales))
                        state["delta"] = delta.copy()
                        state["output"] = values[row].copy() if np.isfinite(values[row]).all() else None
                        if sustained and state["anchor"] is None and np.isfinite(base).all():
                            # A first appearance supplies history only for a
                            # later report, never for the current attack.
                            state["anchor"] = base.copy()
                    current["reason_counts"] = dict(reasons)
                    current["feasible"] = current["material_core_reports"] > 0 and not current["forced_abort"]
                    if position == current["_end"]:
                        current = None
                for source, row in rows.items():
                    if np.isfinite(original[row]).all():
                        histories[source] = (timestamp, original[row].copy())
                    else:
                        histories.pop(source, None)
    changed = np.any(np.isfinite(original) & np.isfinite(values) & (np.abs(values - original) > 1e-12), axis=1)
    out[columns] = values
    out.loc[changed, "label"] = 1
    out.loc[changed, "injection_label"] = 1
    out.loc[changed, "fault_type"] = audit["attack_type"][changed]
    out.loc[changed, "fault_event_id"] = audit["attack_id"][changed]
    for name, array in audit.items():
        out[name] = array
    for index, column in enumerate(columns):
        suffix = column[len("value_"):]
        out["attack_delta_" + suffix] = np.where(np.isfinite(original[:, index]), values[:, index] - original[:, index], 0.)
        out["attack_delta_limit_" + suffix] = max_delta[index]
        out["attack_step_limit_" + suffix] = delta_step[index]
    for event in events:
        event.pop("_start")
        event.pop("_end")
    event_feasibility = {event["event_id"]: event["feasible"] for event in events}
    out["attack_event_feasible"] = [event_feasibility.get(identity, False) for identity in audit["attack_id"]]
    changed_rounds = len(calendar.loc[changed, ["sequence_id", "task_id", "timestamp"]].drop_duplicates())
    return out, {"protocol": PROTOCOL, "seed": int(seed), "affected_sources": sorted(affected),
                 "unseen_sources": sorted(set(frame.source_id.astype(str)) - training_sources),
                 "total_reports": n, "injected_reports": int(changed.sum()),
                 "injected_report_fraction": float(changed.sum()) / n if n else 0.,
                 "attempted_reports": int(audit["attack_attempted"].sum()),
                 "attempted_report_fraction": float(audit["attack_attempted"].sum()) / n if n else 0.,
                 "feasible_reports": int(audit["attack_feasible"].sum()), "scheduled_attack_rounds": planned_rounds,
                 "total_task_rounds": total_rounds, "requested_attack_rounds": requested_rounds,
                 "requested_round_fraction": rate,
                 "scheduled_round_fraction": planned_rounds / total_rounds if total_rounds else 0.,
                 "changed_round_fraction": changed_rounds / total_rounds if total_rounds else 0.,
                 "round_budget_excess": max(0, planned_rounds - requested_rounds),
                 "min_attempts": minimum_attempts, "minimum_attempt_extra_events": minimum_extras,
                 "unfilled_minimum_attempts": max(0, minimum_attempts - len(events)),
                 "budget_policy": ("continuous_suffix_after_warmup" if sustained else
                                   "whole_event_round_up_then_calendar_only_minimum_attempts"),
                 "persistence": persistence, "start_offset_rounds": start_offset,
                 "source_selection_mode": selection_mode,
                 "events": events, "profile": copy.deepcopy(profile), "config": copy.deepcopy(config),
                 "baseline_assumption": "Original labels are unverified; injection labels identify changed reports only",
                 "scheduling": ("Calendar-only balanced objective/coalition cycle; continuous suffix without forced recovery"
                                if sustained else
                                "Calendar-only balanced objective/coalition cycle; duration includes zero-delta recovery endpoint")}
