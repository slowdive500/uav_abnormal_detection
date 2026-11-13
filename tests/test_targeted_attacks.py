"""Targeted attack generation is causal, reproducible and independently bounded."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.attacks import ATTACK_KINDS, fit_attack_profile, inject_targeted_attacks


def reports(times=120, sources=8):
    return pd.DataFrame([
        {"sequence_id": "flight", "task_id": "cell", "timestamp": float(time),
         "source_id": "uav_%d" % source, "value_0": 20 + np.sin(time / 7.) + .1 * source,
         "value_1": 10 + np.cos(time / 9.) + .05 * source,
         "label": -1, "truth_0": -1000., "reference_0": 1000., "condition_x": float(source)}
        for time in range(times) for source in range(sources)])


def config(**changes):
    result = {"protocol": "targeted_report_attack", "kinds": list(ATTACK_KINDS),
              "source_fraction": .25, "source_selection_seed": 6113, "report_fraction": .8,
              "duration": 8, "coalition_sizes": [1, 2], "max_local_fraction": .49,
              "max_gap_seconds": 1.1, "assume_unlabelled_normal": True,
              "constraints": {"target_delta_sigma": .75, "max_delta_sigma": 1.,
                              "max_delta_step_sigma": .25, "min_effect_sigma": .1,
                              "hard_bounds": {"value_0": [0, 40], "value_1": [0, 30]}}}
    result.update(changes)
    return result


@pytest.mark.parametrize("mode", [{}, {"persistence": "sustained",
                                      "source_selection_mode": "present_compromised_pool"}])
def test_profile_freezes_original_training_scales_and_controlled_sources(mode):
    train = reports()
    snapshot = train.copy(deep=True)
    cfg = config(**mode)
    profile = fit_attack_profile(train, cfg)
    np.testing.assert_allclose(profile["scales"], train[["value_0", "value_1"]].std(ddof=0))
    pd.testing.assert_frame_equal(train, snapshot)
    json.dumps(profile, allow_nan=False)
    test = train.copy()
    test.loc[test.source_id == "uav_0", "source_id"] = "previously_unseen"
    test.loc[:, ["value_0", "value_1"]] += 1000
    profile_snapshot = copy.deepcopy(profile)
    attacked, metadata = inject_targeted_attacks(test, cfg, profile, seed=42)
    assert metadata["affected_sources"] == profile["affected_sources"]
    assert metadata["unseen_sources"] == ["previously_unseen"]
    assert not attacked.loc[attacked.source_id == "previously_unseen", "attack_attempted"].any()
    assert profile == profile_snapshot


def test_truth_labels_and_reference_do_not_select_or_construct_attacks():
    frame = reports()
    cfg = config()
    profile = fit_attack_profile(frame, cfg)
    altered = frame.copy()
    altered["truth_0"] = "not numeric and never read"
    altered["reference_0"] = np.arange(len(frame)) * 1e9
    altered["label"] = np.arange(len(frame)) % 2
    assert fit_attack_profile(altered, cfg) == profile
    first, first_metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    second, second_metadata = inject_targeted_attacks(altered, cfg, profile, 42)
    fields = [column for column in first if column.startswith("attack_")] + ["value_0", "value_1"]
    pd.testing.assert_frame_equal(first[fields], second[fields])
    assert first_metadata == second_metadata
    pd.testing.assert_series_equal(second.truth_0, altered.truth_0)
    pd.testing.assert_series_equal(second.reference_0, altered.reference_0)


def test_reproducible_and_does_not_mutate_input_or_config():
    frame = reports()
    snapshot = frame.copy(deep=True)
    cfg = config()
    config_snapshot = copy.deepcopy(cfg)
    profile = fit_attack_profile(frame, cfg)
    first, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    second, repeated = inject_targeted_attacks(frame, cfg, profile, 42)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, snapshot)
    assert cfg == config_snapshot
    assert metadata == repeated
    json.dumps(metadata, allow_nan=False)
    assert first.injection_label.sum() > 0
    pd.testing.assert_frame_equal(first[list(snapshot.columns.difference(["value_0", "value_1", "label"]))],
                                  snapshot[list(snapshot.columns.difference(["value_0", "value_1", "label"]))])


def test_future_values_do_not_change_past_reports_or_attack_schedule():
    frame = reports()
    cfg = config()
    profile = fit_attack_profile(frame, cfg)
    altered = frame.copy()
    altered.loc[altered.timestamp >= 60, ["value_0", "value_1"]] += 500
    first, meta = inject_targeted_attacks(frame, cfg, profile, 42)
    second, other_meta = inject_targeted_attacks(altered, cfg, profile, 42)
    # Final event feasibility is an explicitly retrospective audit, not an input.
    causal_fields = [column for column in first if column != "attack_event_feasible"]
    pd.testing.assert_frame_equal(first.loc[first.timestamp < 60, causal_fields],
                                  second.loc[second.timestamp < 60, causal_fields])
    schedule = lambda m: [(e["kind"], e["time_start"], e["time_end"], e["source_ids"], e["direction"]) for e in m["events"]]
    assert schedule(meta) == schedule(other_meta)


def test_other_sources_values_are_not_attacker_inputs():
    frame = reports()
    cfg = config()
    profile = fit_attack_profile(frame, cfg)
    altered = frame.copy()
    honest = ~altered.source_id.isin(profile["affected_sources"])
    altered.loc[honest, ["value_0", "value_1"]] += 1000
    first, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    second, other_metadata = inject_targeted_attacks(altered, cfg, profile, 42)
    pd.testing.assert_frame_equal(first.loc[~honest], second.loc[~honest])
    assert metadata == other_metadata


def test_completed_attacks_obey_bounds_and_return_to_zero_smoothly():
    frame = reports()
    cfg = config(kinds=["target_bias"])
    profile = fit_attack_profile(frame, cfg)
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert not attacked.attack_forced_abort.any()
    for feature, scale in enumerate(profile["scales"]):
        delta = attacked["attack_delta_%d" % feature]
        assert delta.abs().max() <= scale + 1e-12
        assert attacked["value_%d" % feature].min() >= 0
        for _, group in attacked.groupby(["sequence_id", "task_id", "source_id"]):
            steps = np.diff(np.r_[0., group.sort_values("timestamp")["attack_delta_%d" % feature]])
            assert np.abs(steps).max() <= .25 * scale + 1e-12
    for event in metadata["events"]:
        endpoints = attacked[(attacked.attack_id == event["event_id"]) & (attacked.timestamp == event["time_end"])]
        assert (endpoints[["attack_delta_0", "attack_delta_1"]] == 0).all().all()
    changed = (attacked[["value_0", "value_1"]].to_numpy() != frame[["value_0", "value_1"]].to_numpy()).any(1)
    np.testing.assert_array_equal(changed, attacked.injection_label.astype(bool))


def test_missingness_and_illegal_originals_are_not_repaired_and_abort_is_visible():
    frame = reports()
    cfg = config(kinds=["target_bias"], coalition_sizes=[1])
    profile = fit_attack_profile(frame, cfg)
    initial, _ = inject_targeted_attacks(frame, cfg, profile, 42)
    changed = initial[initial.injection_label == 1].iloc[0]
    illegal = (frame.source_id == changed.source_id) & (frame.timestamp == changed.timestamp + 1)
    altered = frame.copy()
    altered.loc[illegal, "value_0"] = -4.
    altered.loc[(altered.source_id == changed.source_id) & (altered.timestamp == changed.timestamp + 2), "value_1"] = np.nan
    attacked, metadata = inject_targeted_attacks(altered, cfg, profile, 42)
    assert (attacked.loc[illegal, "value_0"] == -4).all()
    assert (attacked.loc[illegal, "attack_infeasible_reason"] == "original_outside_hard_bounds").all()
    assert attacked.loc[illegal, "attack_forced_abort"].all()
    assert any(e["forced_abort"] and not e["feasible"] for e in metadata["events"])
    np.testing.assert_array_equal(attacked[["value_0", "value_1"]].isna(), altered[["value_0", "value_1"]].isna())


def test_local_majority_is_checked_from_actual_participants():
    train = reports()
    cfg = config(kinds=["target_bias"], coalition_sizes=[2])
    profile = fit_attack_profile(train, cfg)
    controlled = set(profile["affected_sources"])
    honest = sorted(set(train.source_id) - controlled)
    sparse = train[train.source_id.isin(controlled | set(honest[:2]))].copy()
    attacked, metadata = inject_targeted_attacks(sparse, cfg, profile, 42)
    assert attacked.attack_attempted.any()
    assert not attacked.injection_label.any()
    assert all(not event["feasible"] for event in metadata["events"])
    assert any("local_minority_unavailable" in event["reason_counts"] for event in metadata["events"])
    enough = train[train.source_id.isin(controlled | set(honest[:3]))].copy()
    attacked, _ = inject_targeted_attacks(enough, cfg, profile, 42)
    assert attacked.injection_label.any()
    for _, round_reports in attacked.groupby("timestamp"):
        assert round_reports.injection_label.sum() <= .49 * len(round_reports)


def test_balanced_objective_scenarios_and_failed_short_attempts_are_retained():
    frame = reports(times=200)
    cfg = config(duration=2)
    profile = fit_attack_profile(frame, cfg)
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    for size in [1, 2]:
        counts = [sum(e["kind"] == kind and e["requested_coalition_size"] == size for e in metadata["events"])
                  for kind in ATTACK_KINDS]
        assert max(counts) - min(counts) <= 1
    trend = attacked[attacked.attack_type == "false_trend"]
    assert len(trend) > 0 and trend.attack_attempted.all()
    assert not trend.injection_label.any()
    assert not trend.attack_event_feasible.any()
    bias = attacked[attacked.attack_type == "target_bias"]
    assert bias.injection_label.any()


def test_gaps_end_event_and_causal_history():
    frame = reports(times=40)
    frame.loc[frame.timestamp >= 20, "timestamp"] += 1000
    cfg = config()
    profile = fit_attack_profile(frame, cfg)
    _, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert all(event["time_end"] - event["time_start"] < 20 for event in metadata["events"])


def test_zero_budget_and_constant_training_features_are_well_defined():
    frame = reports()
    frame["value_0"] = 20.
    cfg = config(report_fraction=0)
    profile = fit_attack_profile(frame, cfg)
    assert profile["scales"][0] == 1e-6
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    pd.testing.assert_frame_equal(attacked[["value_0", "value_1"]], frame[["value_0", "value_1"]])
    assert not metadata["events"]


def test_small_round_budget_never_truncates_two_round_bias_recovery():
    train = reports()
    cfg = config(kinds=["target_bias"], coalition_sizes=[1], duration=2, report_fraction=.05)
    profile = fit_attack_profile(train, cfg)
    short = reports(times=2)
    attacked, metadata = inject_targeted_attacks(short, cfg, profile, 42)
    assert metadata["requested_attack_rounds"] == 1
    assert metadata["scheduled_attack_rounds"] == 2
    assert metadata["scheduled_round_fraction"] == 1.
    assert metadata["round_budget_excess"] == 1
    assert attacked.loc[attacked.timestamp == 0, "injection_label"].sum() == 1
    assert attacked.loc[attacked.timestamp == 1, "injection_label"].sum() == 0


def test_calendar_only_minimum_attempts_covers_objectives_before_coalitions():
    frame = reports(times=100)
    cfg = config(report_fraction=.01, min_attempts=3)
    profile = fit_attack_profile(frame, cfg)
    _, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert len(metadata["events"]) == 3
    assert {event["kind"] for event in metadata["events"]} == set(ATTACK_KINDS)
    assert all(event["scenario"] == "single" for event in metadata["events"])
    assert all(event["scheduled_rounds"] >= 2 for event in metadata["events"])
    assert metadata["minimum_attempt_extra_events"] == 2
    assert metadata["unfilled_minimum_attempts"] == 0


def test_optional_report_step_bound_and_no_repair_of_original_jumps():
    frame = reports()
    cfg = config(kinds=["target_bias"], coalition_sizes=[1])
    cfg["constraints"]["max_report_step_sigma"] = .5
    profile = fit_attack_profile(frame, cfg)
    attacked, _ = inject_targeted_attacks(frame, cfg, profile, 42)
    for feature, scale in enumerate(profile["scales"]):
        for _, group in attacked.groupby("source_id"):
            assert group.sort_values("timestamp")["value_%d" % feature].diff().abs().max() <= .5 * scale + 1e-12
    changed = attacked[(attacked.injection_label == 1) & (attacked.timestamp > 0)].iloc[0]
    altered = frame.copy()
    jump = (altered.source_id == changed.source_id) & (altered.timestamp == changed.timestamp)
    altered.loc[jump, "value_0"] += 2
    result, _ = inject_targeted_attacks(altered, cfg, profile, 42)
    pd.testing.assert_series_equal(result.loc[jump, "value_0"], altered.loc[jump, "value_0"])
    assert (result.loc[jump, "attack_infeasible_reason"] == "original_step_exceeds_limit").all()


@pytest.mark.parametrize("slope_sign", [-1, 1])
def test_change_suppression_flattens_both_rises_and_falls(slope_sign):
    cfg = config(kinds=["change_suppression"], coalition_sizes=[1], report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    frame = reports(times=64)
    frame["value_0"] = 20 + slope_sign * .1 * frame.timestamp
    frame["value_1"] = 10 + slope_sign * .05 * frame.timestamp
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    core = attacked[(attacked.attack_phase == "core") & (attacked.injection_label == 1)]
    assert len(core) > 0
    assert (core.attack_direction == 0).all()
    assert (slope_sign * core.attack_delta_0 < 0).all()
    for event in metadata["events"]:
        assert event["direction"] == 0
        if not event["feasible"]:
            continue
        reports_in_core = core[core.attack_id == event["event_id"]]
        anchor_time = event["time_start"] - 1
        np.testing.assert_allclose(reports_in_core.value_0, 20 + slope_sign * .1 * anchor_time)
        np.testing.assert_allclose(reports_in_core.value_1, 10 + slope_sign * .05 * anchor_time)


def test_sustained_bias_starts_after_warmup_and_never_restarts_or_forces_recovery():
    frame = reports(times=20)
    cfg = config(kinds=["target_bias"], coalition_sizes=[1], persistence="sustained",
                 start_offset_rounds=2, report_fraction=1., duration=2)
    profile = fit_attack_profile(reports(), cfg)
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert not attacked.loc[attacked.timestamp < 2, "attack_attempted"].any()
    assert len(metadata["events"]) == 1
    event = metadata["events"][0]
    assert (event["time_start"], event["time_end"], event["core_rounds"]) == (2., 19., 18)
    assert metadata["scheduled_attack_rounds"] == 18
    assert not attacked.attack_forced_abort.any()
    controlled = attacked[attacked.attack_attempted].sort_values("timestamp")
    for feature, scale in enumerate(profile["scales"]):
        delta = controlled["attack_delta_%d" % feature].to_numpy()
        np.testing.assert_allclose(np.abs(delta[:3]), np.array([.25, .5, .75]) * scale)
        np.testing.assert_allclose(delta[2:], np.full(len(delta) - 2, delta[2]))
        assert np.max(np.abs(np.diff(np.r_[0., delta]))) <= .25 * scale + 1e-12
    assert set(controlled.attack_phase) == {"core"}
    # ``duration`` is a legacy bounded-event setting, not a reset interval.
    other, _ = inject_targeted_attacks(frame, dict(cfg, duration=8), profile, 42)
    pd.testing.assert_frame_equal(attacked, other)


def test_sustained_fixed_coalition_resumes_after_absence_without_fabricating_reports():
    frame = reports(times=12)
    cfg = config(kinds=["target_bias"], coalition_sizes=[2], persistence="sustained",
                 start_offset_rounds=1, report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    missing_source, remaining_source = profile["affected_sources"]
    honest = sorted(set(frame.source_id) - set(profile["affected_sources"]))
    # At round 5 only one attacker and two honest reporters are present: the
    # actual one-member coalition is still a strict minority.
    frame = frame[~((frame.source_id == missing_source) & frame.timestamp.isin([4., 5.]))]
    frame = frame[(frame.timestamp != 5.) | frame.source_id.isin([remaining_source] + honest[:2])].copy()
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert len(attacked) == len(frame)
    pd.testing.assert_frame_equal(attacked[["timestamp", "source_id"]], frame[["timestamp", "source_id"]])
    assert not attacked.attack_forced_abort.any()
    assert metadata["events"][0]["feasible"]
    absent = attacked[attacked.source_id == missing_source].set_index("timestamp")
    np.testing.assert_allclose(absent.loc[3., ["attack_delta_0", "attack_delta_1"]].astype(float),
                               absent.loc[6., ["attack_delta_0", "attack_delta_1"]].astype(float))
    assert attacked.loc[attacked.timestamp == 5., "injection_label"].sum() == 1


def test_sustained_dynamic_pool_attacks_later_arrivals_and_keeps_real_participation():
    frame = reports(times=14)
    cfg = config(kinds=["target_bias"], coalition_sizes=[1], persistence="sustained",
                 source_selection_mode="present_compromised_pool", start_offset_rounds=1,
                 report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    first_source, later_source = profile["affected_sources"]
    frame = frame[~(((frame.source_id == first_source) & (frame.timestamp >= 7)) |
                    ((frame.source_id == later_source) & (frame.timestamp < 7)))].copy()
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert len(attacked) == len(frame)
    assert attacked.loc[attacked.timestamp >= 7, "injection_label"].sum() == 7
    assert set(attacked.loc[attacked.injection_label == 1, "source_id"]) == set(profile["affected_sources"])
    assert metadata["events"][0]["source_ids"] == profile["affected_sources"]
    assert metadata["events"][0]["max_active_coalition_size"] == 1
    assert not attacked.attack_forced_abort.any()
    later = attacked[(attacked.source_id == later_source) & (attacked.timestamp == 7)].iloc[0]
    assert abs(later.attack_delta_0) == pytest.approx(.25 * profile["scales"][0])
    for _, group in attacked.groupby("timestamp"):
        assert group.injection_label.sum() <= .49 * len(group)


def test_sustained_dynamic_pool_retries_after_quorum_loss_and_resets_actual_normal_reports():
    frame = reports(times=14)
    cfg = config(kinds=["target_bias"], coalition_sizes=[1], persistence="sustained",
                 source_selection_mode="present_compromised_pool", start_offset_rounds=1,
                 report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    first, _ = inject_targeted_attacks(frame, cfg, profile, 42)
    selected = first.loc[first.injection_label == 1, "source_id"].iloc[0]
    frame = frame[(frame.timestamp != 6.) | (frame.source_id == selected)].copy()
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    lonely = attacked[attacked.timestamp == 6.].iloc[0]
    assert lonely.injection_label == 0
    assert lonely.attack_infeasible_reason == "local_minority_unavailable"
    assert lonely.attack_forced_abort  # A forced return to original is audited.
    resumed = attacked[(attacked.timestamp == 7.) & (attacked.source_id == selected)].iloc[0]
    assert resumed.injection_label == 1
    assert abs(resumed.attack_delta_0) == pytest.approx(.25 * profile["scales"][0])
    assert metadata["events"][0]["forced_abort"]
    assert not metadata["events"][0]["feasible"]  # Do not hide the interruption.


@pytest.mark.parametrize("kind", ["target_bias", "change_suppression"])
def test_sustained_dynamic_pool_remains_causal_and_never_reads_truth(kind):
    frame = reports(times=30)
    cfg = config(kinds=[kind], coalition_sizes=[2], persistence="sustained",
                 source_selection_mode="present_compromised_pool", start_offset_rounds=2,
                 report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    altered = frame.copy()
    altered.loc[altered.timestamp >= 15, ["value_0", "value_1"]] += 1000
    altered[["truth_0", "reference_0", "label"]] = -9999
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    other, other_metadata = inject_targeted_attacks(altered, cfg, profile, 42)
    fields = [column for column in attacked if column.startswith("attack_")
              and column != "attack_event_feasible"] + ["value_0", "value_1", "injection_label"]
    pd.testing.assert_frame_equal(attacked.loc[attacked.timestamp < 15, fields],
                                  other.loc[other.timestamp < 15, fields])
    schedule = lambda meta: [(event["time_start"], event["time_end"], event["source_ids"],
                              event["direction"]) for event in meta["events"]]
    assert schedule(metadata) == schedule(other_metadata)
    json.dumps(metadata, allow_nan=False)


def test_sustained_suppression_uses_earlier_anchor_and_can_hold_last_report():
    cfg = config(kinds=["change_suppression"], coalition_sizes=[1], persistence="sustained",
                 source_selection_mode="present_compromised_pool", start_offset_rounds=2,
                 report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    frame = reports(times=8)
    frame["value_0"] = 20 + .1 * frame.timestamp
    frame["value_1"] = 10 + .05 * frame.timestamp
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    changed = attacked[attacked.injection_label == 1]
    assert len(changed) == 6
    np.testing.assert_allclose(changed.value_0, 20.1)
    np.testing.assert_allclose(changed.value_1, 10.05)
    assert changed.timestamp.max() == 7.
    assert metadata["events"][0]["feasible"]


def test_sustained_suppression_late_arrival_supplies_history_only_for_next_report():
    cfg = config(kinds=["change_suppression"], coalition_sizes=[1], persistence="sustained",
                 source_selection_mode="present_compromised_pool", start_offset_rounds=1,
                 report_fraction=1.)
    profile = fit_attack_profile(reports(), cfg)
    frame = reports(times=10)
    frame["value_0"] = 20 + .1 * frame.timestamp
    late, other = profile["affected_sources"]
    frame = frame[(frame.source_id != other) &
                  ((frame.source_id != late) | (frame.timestamp >= 5.))].copy()
    attacked, _ = inject_targeted_attacks(frame, cfg, profile, 42)
    late_reports = attacked[attacked.source_id == late].set_index("timestamp")
    assert late_reports.loc[5., "attack_infeasible_reason"] == "no_causal_history"
    assert late_reports.loc[5., "injection_label"] == 0
    assert late_reports.loc[6., "injection_label"] == 1
    assert late_reports.loc[6., "value_0"] == pytest.approx(20.5)


def test_sustained_budget_and_unfillable_minimum_attempts_are_explicit():
    frame = reports(times=10)
    cfg = config(kinds=["target_bias", "change_suppression"], persistence="sustained",
                 start_offset_rounds=2, report_fraction=.5, min_attempts=2)
    profile = fit_attack_profile(reports(), cfg)
    attacked, metadata = inject_targeted_attacks(frame, cfg, profile, 42)
    assert metadata["requested_attack_rounds"] == metadata["scheduled_attack_rounds"] == 4
    assert metadata["unfilled_minimum_attempts"] == 1
    assert not attacked.loc[attacked.timestamp < 6, "attack_attempted"].any()
    assert len(metadata["events"]) == 1
    _, no_room = inject_targeted_attacks(frame, dict(cfg, start_offset_rounds=10), profile, 42)
    assert not no_room["events"]
    assert no_room["scheduled_attack_rounds"] == 0


@pytest.mark.parametrize("changes", [
    {"max_local_fraction": .5}, {"coalition_sizes": [0]},
    {"kinds": ["spike"]}, {"duration": 1.5},
    {"constraints": {"min_effect_sigma": 2., "max_delta_sigma": 1.}},
    {"constraints": {"hard_bounds": {"value_0": [10, 0]}}},
    {"persistence": "forever"}, {"start_offset_rounds": -1},
    {"start_offset_rounds": 1.5}, {"source_selection_mode": "invent_sources"},
    {"source_selection_mode": "present_compromised_pool"},
])
def test_invalid_constraints_fail_before_injection(changes):
    with pytest.raises(ValueError):
        fit_attack_profile(reports(), config(**changes))
