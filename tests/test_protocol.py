"""Scientific protocol invariants independent of model architecture."""
import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from uav_abnormality.corruption import inject_faults, KINDS
from uav_abnormality.data.schema import validate_table
from uav_abnormality.experiment import score_windows, validation_loss
from uav_abnormality.metrics import binary_metrics, calibrate_threshold, evaluate_reports
from uav_abnormality.protocol import Standardizer, Windows, split_table


def reports(sequences=1, times=20, tasks=1, sources=3):
    records = []
    for seq, task, time, source in itertools.product(range(sequences), range(tasks),
                                                    range(times), range(sources)):
        records.append({"sequence_id": "flight_%d" % seq, "task_id": "cell_%d" % task,
                        "timestamp": float(seq * 1000 + time), "source_id": "uav_%d" % source,
                        "value_0": float(time + source), "value_1": float(time - source),
                        "label": -1, "truth_0": -1e9, "fault_type": "unknown"})
    return validate_table(pd.DataFrame(records))


def used_keys(dataset):
    used = set()
    for gid, end in dataset.index:
        group = dataset.groups[gid]
        for time in group["times"][end - dataset.window_size + 1:end + 1]:
            used.add(group["key"] + (time,))
    return used


@pytest.mark.parametrize("mode,sequences,times", [("chronological", 1, 80), ("sequence", 8, 12)])
def test_split_windows_share_no_context(mode, sequences, times):
    frame = reports(sequences=sequences, times=times)
    parts, manifest = split_table(frame, {"mode": mode, "ratios": [.4, .2, .2, .2]})
    scaler = Standardizer().fit(parts["train"])
    datasets = {name: Windows(part, ["uav_0", "uav_1", "uav_2"], scaler, 4)
                for name, part in parts.items()}
    keys = {name: used_keys(data) for name, data in datasets.items()}
    for one, two in itertools.combinations(keys, 2):
        assert not keys[one].intersection(keys[two])
    if mode == "sequence":
        for one, two in itertools.combinations(parts, 2):
            assert not set(parts[one].sequence_id).intersection(parts[two].sequence_id)
    assert manifest["window_overlap_between_splits"] is False


def test_standardizer_is_fitted_only_to_supplied_training_values():
    frame = reports(times=80)
    parts, _ = split_table(frame, {"mode": "chronological", "ratios": [.4, .2, .2, .2]})
    scaler = Standardizer().fit(parts["train"])
    expected = parts["train"][["value_0", "value_1"]].to_numpy()
    np.testing.assert_allclose(scaler.mean, expected.mean(0))
    np.testing.assert_allclose(scaler.scale, expected.std(0))
    snapshot = scaler.to_dict()
    parts["test"].loc[:, ["value_0", "value_1"]] += 1e6
    test = Windows(parts["test"], ["uav_0", "uav_1", "uav_2"], scaler, 3)
    assert test[0]["x"].abs().min() > 1000
    assert snapshot == scaler.to_dict()
    restored = Standardizer.from_dict(snapshot)
    np.testing.assert_allclose(restored.transform(expected), scaler.transform(expected))


def test_labels_and_reference_truth_never_enter_model_tensors():
    frame = reports(times=8)
    changed = frame.copy()
    changed["label"] = np.arange(len(frame)) % 2
    changed["truth_0"] = np.arange(len(frame)) * 1e12
    changed["truth_1"] = "deliberately nonnumeric audit-only field"
    changed["fault_type"] = "offset"
    first_scaler, second_scaler = Standardizer().fit(frame), Standardizer().fit(changed)
    assert first_scaler.to_dict() == second_scaler.to_dict()
    first = Windows(frame, ["uav_0", "uav_1", "uav_2"], first_scaler, 4)
    second = Windows(changed, ["uav_0", "uav_1", "uav_2"], second_scaler, 4)
    assert len(first) == len(second)
    for index in range(len(first)):
        assert set(first[index]) == {"x", "mask", "index"}
        torch.testing.assert_close(first[index]["x"], second[index]["x"])
        torch.testing.assert_close(first[index]["mask"], second[index]["mask"])
        assert first[index]["x"].shape[-1] == 2


def test_unseen_source_is_rejected_instead_of_reindexed():
    frame = reports()
    with pytest.raises(ValueError, match="Unseen physical sources"):
        Windows(frame, ["uav_0", "uav_1"], Standardizer().fit(frame), 4)


@pytest.mark.parametrize("kind", KINDS)
def test_fault_injection_preserves_missingness_truth_and_row_identity(kind):
    frame = reports(times=12)
    frame.loc[frame.index[::4], "value_0"] = np.nan
    frame.loc[frame.index[::7], ["value_0", "value_1"]] = np.nan
    config = {"source_fraction": 1.0, "report_fraction": 1.0,
              "magnitude": 4.0, "duration": 3, "kinds": [kind],
              "assume_unlabelled_normal": True}
    original = frame.copy(deep=True)
    corrupted, metadata = inject_faults(frame, config, [1.0, 1.0], seed=7)
    columns = ["value_0", "value_1"]
    np.testing.assert_array_equal(frame[columns].isna(), corrupted[columns].isna())
    pd.testing.assert_frame_equal(frame, original)
    audit = ["sequence_id", "task_id", "timestamp", "source_id", "truth_0"]
    pd.testing.assert_frame_equal(frame[audit], corrupted[audit])
    changed = ((frame[columns] - corrupted[columns]).abs() > 1e-12).any(axis=1)
    np.testing.assert_array_equal(corrupted.label.to_numpy() == 1, changed.to_numpy())
    assert metadata["injected_reports"] == int(changed.sum())
    repeated, repeated_metadata = inject_faults(frame, config, [1.0, 1.0], seed=7)
    pd.testing.assert_frame_equal(corrupted, repeated)
    assert metadata == repeated_metadata


def test_unknown_normality_assumption_must_be_explicit():
    with pytest.raises(ValueError, match="assume_unlabelled_normal"):
        inject_faults(reports(), {"source_fraction": .2}, [1, 1], seed=0)


def test_unknown_labels_are_not_fabricated_as_normal_metrics():
    result = binary_metrics([-1, -1, -1], [0.1, 0.9, 0.5], 0.4)
    assert result["status"] == "no_ground_truth_labels"
    assert result["labelled_reports"] == 0
    assert result["unknown_or_invalid_reports"] == 3
    assert "f1" not in result and "auroc" not in result
    mixed = binary_metrics([-1, 0, 1, 0], [100, .1, .9, float("nan")], .5)
    assert mixed["labelled_reports"] == 2
    assert mixed["f1"] == 1.0
    assert mixed["unknown_or_invalid_reports"] == 2


def test_fixed_threshold_raw_metrics_and_single_class_are_explicit():
    result = binary_metrics([0, 0, 1, 1], [.1, .6, .4, .9], .5)
    assert [result[k] for k in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert result["f1"] == .5
    assert result["point_adjustment"] is False
    single = binary_metrics([0, 0], [.1, .9], .5)
    assert single["status"] == "single_label_class"
    assert single["auroc"] is None
    assert single["average_precision"] is None
    assert single["recall"] is None and single["f1"] is None
    assert single["false_positive_rate"] == .5
    no_attack = evaluate_reports(pd.DataFrame({"source_id": ["a", "b"],
        "label": [0, 0], "score": [.1, .9]}), .5)
    assert no_attack["macro_source_f1"] is None
    assert no_attack["false_positive_rate"] == .5
    assert calibrate_threshold([0, 1, 2, 3, 4], .2) == 4


def test_per_fault_uses_same_clean_negatives_not_other_fault_positives():
    frame = pd.DataFrame({"source_id": ["a", "a", "b", "b"],
                          "label": [0, 1, 1, 1], "score": [.1, .9, .9, .1],
                          "fault_type": ["none", "offset", "spike", "none"]})
    result = evaluate_reports(frame, .5)
    assert result["per_fault"]["offset"]["labelled_reports"] == 2
    assert result["per_fault"]["offset"]["f1"] == 1.0


class CountingStream(torch.nn.Module):
    """Expose reset boundaries through scores, independent of PRBTD internals."""
    def __init__(self):
        super().__init__()
        self.resets = 0
        self.count = 0

    def reset_state(self):
        self.resets += 1
        self.count = 0

    def score(self, x, mask):
        raise AssertionError("Stateful evaluation must use stream_score")

    def stream_score(self, x, mask):
        assert not self.training
        counts = torch.arange(self.count + 1, self.count + len(x) + 1,
                              device=x.device, dtype=x.dtype)
        self.count += len(x)
        return counts[:, None].repeat(1, x.shape[2])


def test_stream_resets_between_sequences_tasks_and_repeated_evaluations():
    frame = reports(sequences=2, times=8, tasks=2)
    data = Windows(frame, ["uav_0", "uav_1", "uav_2"], Standardizer().fit(frame), 3)
    first = CountingStream()
    result = score_windows(first, data, torch.device("cpu"), batch_size=4)
    assert first.resets == 4
    for _, group in result.groupby(["sequence_id", "task_id"]):
        sequence_scores = group.groupby("timestamp").score.first().to_numpy()
        np.testing.assert_array_equal(sequence_scores, np.arange(1, 7))
    repeated = score_windows(first, data, torch.device("cpu"), batch_size=1)
    assert first.resets == 8
    pd.testing.assert_frame_equal(result, repeated)


def test_stream_resets_across_long_gap_within_same_task():
    frame = reports(times=12)
    frame.loc[frame.timestamp >= 6, "timestamp"] += 94
    data = Windows(frame, ["uav_0", "uav_1", "uav_2"], Standardizer().fit(frame),
                   window_size=3, max_gap_seconds=2)
    for gid, end in data.index:
        assert np.diff(data.groups[gid]["times"][end - 2:end + 1]).max() <= 2
    model = CountingStream()
    scored = score_windows(model, data, torch.device("cpu"), batch_size=3)
    endpoint_scores = scored.groupby("timestamp").score.first()
    np.testing.assert_array_equal(endpoint_scores.to_numpy(), [1, 2, 3, 4, 1, 2, 3, 4])
    assert model.resets == 2


def test_scores_exclude_missing_current_reports():
    frame = reports(times=8)
    frame.loc[(frame.timestamp == 7) & (frame.source_id == "uav_2"), ["value_0", "value_1"]] = np.nan
    data = Windows(frame, ["uav_0", "uav_1", "uav_2"], Standardizer().fit(frame), 3)
    result = score_windows(CountingStream(), data, torch.device("cpu"))
    assert len(result[result.timestamp == 7]) == 2
    assert not result[["value_0", "value_1"]].isna().all(axis=1).any()


def test_validation_rng_is_repeatable_without_changing_training_rng():
    class StochasticLoss(torch.nn.Module):
        def loss(self, x, mask):
            return torch.rand(())
    data = [{"x": torch.zeros(2, 3, 2, 1), "mask": torch.ones(2, 3, 2, 1)}]
    torch.manual_seed(32)
    previous = torch.random.get_rng_state()
    one = validation_loss(StochasticLoss(), data, torch.device("cpu"), seed=11)
    torch.testing.assert_close(previous, torch.random.get_rng_state())
    two = validation_loss(StochasticLoss(), data, torch.device("cpu"), seed=11)
    assert one == two
