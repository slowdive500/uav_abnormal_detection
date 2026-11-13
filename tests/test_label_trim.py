"""Label-Trim's review budget, task sampling, and exact p-value decisions."""
import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.calibration import (
    fit_label_trim,
    label_trim_predictions,
    label_trim_pvalues,
    label_trim_threshold,
)


def reports(n=39, sources=1):
    return pd.DataFrame([{"sequence_id": "day1", "task_id": "area1", "timestamp": float(t),
                          "source_id": "source{}".format(source), "score": float(t),
                          "injection_label": 0, "label": 1}
                         for t in range(n) for source in range(sources)])


def test_only_verified_anomalous_top_m_are_removed():
    frame = reports()
    frame.loc[38, "injection_label"] = 1
    frame.loc[1, "injection_label"] = 1  # An unreviewed anomaly remains in the pool.
    calibration, audit = fit_label_trim(frame, {"verification": "injected_fault_oracle"})
    assert (calibration["n"], calibration["m"], calibration["removed"], calibration["retained"]) == (39, 2, 1, 38)
    assert audit.loc[audit.reviewed, "timestamp"].tolist() == [37.0, 38.0]
    assert 37 in calibration["score_pool"] and 1 in calibration["score_pool"]
    assert 38 not in calibration["score_pool"]
    assert (audit.loc[~audit.reviewed, "verified_label"] == -1).all()
    assert "label" not in audit and "injection_label" not in audit


def test_small_pool_zero_budget_and_unattainable_alarm():
    calibration, audit = fit_label_trim(reports(8), {"verification": "injected_fault_oracle"})
    assert calibration["m"] == 0
    assert not audit.reviewed.any()
    assert calibration["threshold"] is None
    assert calibration["min_p"] == 1 / 9
    assert not label_trim_predictions([1e9], calibration)[0]
    json.dumps(calibration, allow_nan=False)


def test_budget_limit_and_none_mode():
    with pytest.raises(ValueError, match="exceeds"):
        fit_label_trim(reports(39), {"verification": "injected_fault_oracle", "budget": 3})
    calibration, _ = fit_label_trim(reports(39), {"verification": "none", "budget": 100})
    assert calibration["m"] == 0 and calibration["removed"] == 0
    for budget in (-1, 1.5, True):
        with pytest.raises(ValueError):
            fit_label_trim(reports(), {"budget": budget})


def test_unknown_review_retained_and_unselected_labels_never_used():
    frame = reports()
    frame["injection_label"] = "not inspected"
    frame.loc[38, "injection_label"] = -1
    frame.loc[37, "injection_label"] = 0
    calibration, audit = fit_label_trim(frame, {"verification": "injected_fault_oracle"})
    assert calibration["removed"] == 0 and calibration["unknown_reviews"] == 1
    assert audit.loc[audit.timestamp == 38, "status"].item() == "review_unknown"


def test_sampling_independent_of_scores_labels_order_and_availability():
    frame = reports(30, 5)
    config = {"verification": "none", "selection_seed": 711}
    original, audit = fit_label_trim(frame, config)
    chosen = audit.loc[audit.timestamp == 10, "source_id"].item()
    changed = frame.sample(frac=1.0, random_state=72).reset_index(drop=True)
    changed["score"] = np.arange(len(changed), dtype=float) * 17
    changed["label"] = 0
    changed["injection_label"] = 1
    changed.loc[(changed.timestamp == 10) & (changed.source_id == chosen), "score"] = np.nan
    updated, new_audit = fit_label_trim(changed, config)
    keys = ["sequence_id", "task_id", "timestamp", "source_id", "selection_hash"]
    pd.testing.assert_frame_equal(audit[keys], new_audit[keys])
    assert original["n"] == 30 and updated["n"] == 29
    assert updated["unavailable_selected_tasks"] == 1
    selected = new_audit[new_audit.timestamp == 10].iloc[0]
    assert selected.source_id == chosen and selected.status == "score_unavailable"


def test_time_thinning_precedes_availability_and_keeps_each_task_separate():
    first = reports(10, 2)
    second = first.assign(task_id="area2")
    frame = pd.concat([first, second], ignore_index=True)
    frame.loc[frame.timestamp == 0, "score"] = np.nan
    calibration, audit = fit_label_trim(frame, {"separation_seconds": 3})
    assert calibration["candidate_tasks"] == 20 and calibration["sampled_tasks"] == 8
    assert calibration["n"] == 6 and calibration["unavailable_selected_tasks"] == 2
    for _, task in audit.groupby("task_id"):
        assert task.loc[task.sampled, "timestamp"].tolist() == [0, 3, 6, 9]
    assert (audit.loc[~audit.sampled, "status"] == "time_separation").all()


def test_external_reviews_keep_missing_and_normal_labels(tmp_path):
    frame = reports(59)
    review = frame.loc[[1, 57, 58], ["sequence_id", "task_id", "timestamp", "source_id"]].copy()
    # The invalid unrequested label is deliberately never interpreted.
    review["verified_label"] = ["not reviewed", 0, 1]
    path = tmp_path / "verified.csv"
    review.to_csv(path, index=False)
    calibration, audit = fit_label_trim(frame, {"verification": "external", "external_verification_csv": str(path)})
    assert calibration["m"] == 3 and calibration["removed"] == 1
    assert calibration["unknown_reviews"] == 1
    assert 56 in calibration["score_pool"] and 57 in calibration["score_pool"]
    assert 58 not in calibration["score_pool"]
    assert audit.loc[audit.timestamp == 56, "verified_label"].item() == -1


def test_tied_review_order_is_deterministic():
    frame = reports(39).assign(score=5.0)
    config = {"verification": "injected_fault_oracle"}
    _, first = fit_label_trim(frame, config)
    _, second = fit_label_trim(frame.sample(frac=1, random_state=9), config)
    columns = ["timestamp", "reviewed", "review_rank"]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert first.reviewed.sum() == 2


@pytest.mark.parametrize("alpha", [.05, .1, .2, .58, .75])
def test_threshold_and_pvalues_agree_including_ties(alpha):
    pool = sorted([0, 1, 1, 1, 3, 3, 9] * 7)
    calibration = {"alpha": alpha, "score_pool": pool}
    scores = np.array([-1, 0, .5, 1, 1.00001, 3, 3.00001, 9, 10], dtype=float)
    expected = np.array([(1 + np.sum(np.asarray(pool) >= score)) / (len(pool) + 1) for score in scores])
    np.testing.assert_array_equal(label_trim_pvalues(scores, calibration), expected)
    threshold = label_trim_threshold(calibration)
    np.testing.assert_array_equal(label_trim_predictions(scores, calibration), scores > threshold)


def test_json_reload_is_complete_and_missing_scores_remain_unevaluated():
    calibration, _ = fit_label_trim(reports(49), {"verification": "none"})
    saved = json.loads(json.dumps(calibration, allow_nan=False))
    scores = np.array([0, 47, 48, 49, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(label_trim_pvalues(scores, saved), label_trim_pvalues(scores, calibration))
    assert np.isnan(label_trim_pvalues(scores, saved)[-3:]).all()
    assert not label_trim_predictions(scores, saved)[-3:].any()
    assert label_trim_pvalues(49.0, saved).item() == 1 / 50


def test_empty_selected_pool_is_serializable_and_never_alarms():
    calibration, audit = fit_label_trim(reports(4).assign(score=np.nan))
    assert calibration["n"] == 0 and calibration["score_pool"] == []
    assert calibration["min_p"] == 1 and calibration["threshold"] is None
    assert calibration["status"] == "empty_score_pool"
    assert len(audit) == 4
    assert label_trim_pvalues([1000], calibration).tolist() == [1]
    json.dumps(calibration, allow_nan=False)


def test_bad_keys_duplicates_and_sampling_options_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        fit_label_trim(pd.concat([reports(2), reports(2)]))
    with pytest.raises(ValueError, match="identities"):
        fit_label_trim(reports(2), {"report_columns": ["score"], "unit_columns": ["score"]})
    for alpha in (0, 1, np.nan):
        with pytest.raises(ValueError):
            fit_label_trim(reports(), {"alpha": alpha})
    for separation in (-1, np.inf, np.nan):
        with pytest.raises(ValueError):
            fit_label_trim(reports(), {"separation_seconds": separation})
    broken = reports(2)
    broken.loc[0, "source_id"] = None
    with pytest.raises(ValueError, match="identities"):
        fit_label_trim(broken)


def test_window_plan_skips_warmup_before_spacing_without_replacing_unavailable_source():
    from uav_abnormality.condition_experiment import plan_calibration_registry
    frame = reports(2, 3)
    frame.loc[frame.timestamp == 0, "score"] = np.nan
    planned = plan_calibration_registry(frame, {"window_size": 2, "stride": 1, "max_gap_seconds": 1.1})
    assert planned.loc[planned.calibration_planned, "timestamp"].tolist() == [1, 1, 1]
    config = {"separation_seconds": 2, "verification": "none"}
    calibration, audit = fit_label_trim(planned[planned.calibration_planned], config)
    assert calibration["n"] == 1 and calibration["sampled_tasks"] == 1
    chosen = audit.source_id.item()
    frame.loc[(frame.timestamp == 1) & (frame.source_id == chosen), "score"] = np.nan
    changed = plan_calibration_registry(frame, {"window_size": 2, "stride": 1, "max_gap_seconds": 1.1})
    pd.testing.assert_series_equal(planned.calibration_planned, changed.calibration_planned)
    unavailable, audit2 = fit_label_trim(changed[changed.calibration_planned], config)
    assert unavailable["n"] == 0 and unavailable["unavailable_selected_tasks"] == 1
    assert audit2.source_id.item() == chosen
