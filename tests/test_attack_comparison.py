import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from uav_abnormality.attack_comparison import analyze_attack_comparison


def pair(tmp_path, seed=0, algorithm="tsadc", dataset="taxibj"):
    roots = [tmp_path / "no_attack", tmp_path / "attack"]
    config = dict(algorithm=algorithm, seed=seed, data={"path": "data/fixed.csv"},
                  split={"ratios": [.4, .1, .3, .2]}, window={"window_size": 2},
                  training={"epochs": 20, "patience": 5}, calibration={"alpha": .05},
                  evaluation={"attack_harm_objective": "instantaneous_error", "attack_min_harm_rounds": 1})
    base = pd.DataFrame(dict(sequence_id=["s"] * 4, task_id=["t"] * 4, timestamp=range(4),
                             source_id=["a"] * 4, value_0=[1., 2., 3., 4.], baseline_0=[1., 2., 3., 4.],
                             original_label=[-1] * 4, label=[0] * 4, injection_label=[0] * 4,
                             condition_x=[.1] * 4, truth_0=[2.] * 4))
    for index, root in enumerate(roots):
        run = root / dataset / algorithm / str(seed)
        (run / "splits").mkdir(parents=True)
        run_config = copy.deepcopy(config)
        if index == 0:
            run_config["attack_mode"] = "none"
            run_config["corruption"] = {stage: {"protocol": "none", "assume_unlabelled_normal": True}
                                        for stage in ("train", "validation", "calibration", "test")}
        else:
            run_config["corruption"] = {stage: {"protocol": "targeted_report_attack", "kinds": ["target_bias"]}
                                        for stage in ("train", "validation", "calibration", "test")}
        (run / "config.resolved.yaml").write_text(yaml.safe_dump(run_config))
        torch.save({"input_sha256": "unchanged-original-input"}, run / "checkpoint.pt")
        frame = base.copy()
        if index:
            frame.loc[1, ["value_0", "injection_label", "label"]] = [2.5, 1, 1]
        for stage in ("train", "validation", "calibration", "test"):
            frame.to_csv(run / "splits" / (stage + ".csv"), index=False)
        frame["score"] = [.1, .2, .3, .4]
        frame["prediction"] = [0, 1, 0, 0] if index == 0 else [1, 1, 0, 0]
        frame.to_csv(run / "predictions.csv", index=False)
        estimates = pd.DataFrame(dict(sequence_id=["s"] * 3, task_id=["t"] * 3,
                                      timestamp=[1, 2, 3], reference_0=[2., 2., 2.],
                                      filtered_median_0=[1., 2., np.nan] if index == 0 else [2., 4., 100.]))
        estimates.to_csv(run / "environment_estimates.csv", index=False)
        if algorithm == "condition_alignment":
            estimates = estimates.rename(columns={"filtered_median_0": "estimate_0"})
            estimates["query_id"] = "task_query"
            estimates.to_csv(run / "native_environment.csv", index=False)
        metrics = dict(algorithm=algorithm, input_sha256=hashlib.sha256((run / "splits/test.csv").read_bytes()).hexdigest(), reference_evaluation={"environment": {
            "method": "common_report_median_postprocessing"}})
        (run / "metrics.json").write_text(json.dumps(metrics))
    return roots


def analyze(roots, **kwargs):
    return analyze_attack_comparison(roots[0], {"taxibj": roots[1]}, expected_datasets=["taxibj"],
                                     expected_algorithms=kwargs.pop("algorithms", ["tsadc"]),
                                     expected_seeds=kwargs.pop("seeds", [0]), **kwargs)


def test_fpr_uses_same_untampered_reports_and_mae_same_output_intersection(tmp_path):
    roots = pair(tmp_path)
    result = analyze(roots)
    assert result["completion"] == {"expected": 1, "valid": 1, "invalid": 0, "missing": 0}
    stats = result["groups"][0]["metrics"]
    assert stats["no_attack_false_positive_rate"]["mean"] == 0
    assert stats["no_attack_all_background_fpr"]["mean"] == .25
    assert stats["attack_false_positive_rate"]["mean"] == pytest.approx(1 / 3)
    assert stats["no_attack_common_mae"]["mean"] == .5
    assert stats["attack_common_mae"]["mean"] == 1.
    assert stats["delta_common_mae"]["mean"] == .5
    assert stats["attack_common_all_support_mae"]["mean"] == pytest.approx(100 / 3)
    assert stats["no_attack_common_coverage"]["mean"] == .5
    assert stats["attack_common_coverage"]["mean"] == .75
    assert stats["no_attack_common_matched_coverage"]["mean"] == .5
    row = pd.read_csv(roots[0] / "comparison/per_seed.csv").iloc[0]
    assert row.paired_background_reports == 3
    assert row.common_paired_outputs == 2
    assert row.common_full_test_outputs == 4


def test_delta_std_is_computed_within_seed_not_from_marginal_stds(tmp_path):
    roots = pair(tmp_path)
    pair(tmp_path, seed=42)
    for root in roots:
        path = root / "taxibj/tsadc/42/environment_estimates.csv"
        frame = pd.read_csv(path)
        frame.loc[:1, "filtered_median_0"] += 5
        frame.to_csv(path, index=False)
    result = analyze(roots, seeds=[0, 42])
    metrics = result["groups"][0]["metrics"]
    # First pair: .5 -> 1.; second: 4.5 -> 6.; paired changes .5, 1.5.
    assert metrics["delta_common_mae"] == {"mean": 1., "std": pytest.approx(np.std([.5, 1.5], ddof=1)), "n": 2}
    assert metrics["no_attack_common_mae"]["std"] > metrics["delta_common_mae"]["std"]


@pytest.mark.parametrize("mutation, reason", [
    ("budget", "training.epochs"), ("value", "original report identity/value"),
    ("condition", "original report identity/value"), ("natural_label", "original natural-fault labels"),
    ("hash", "original input_sha256"), ("injection", "contains injection"),
    ("disabled", "attack_mode"), ("reference", "reference differs")])
def test_incompatible_pairs_never_enter_aggregates(tmp_path, mutation, reason):
    roots = pair(tmp_path)
    run = roots[0] / "taxibj/tsadc/0"
    if mutation in ("budget", "disabled"):
        path = run / "config.resolved.yaml"
        config = yaml.safe_load(path.read_text())
        if mutation == "budget":
            config["training"]["epochs"] = 1
        else:
            del config["attack_mode"]
        path.write_text(yaml.safe_dump(config))
    elif mutation in ("value", "condition", "natural_label", "injection"):
        path = run / "splits/train.csv"
        frame = pd.read_csv(path)
        column = {"value": "baseline_0", "condition": "condition_x", "natural_label": "original_label", "injection": "injection_label"}[mutation]
        frame.loc[0, column] = 1.234
        frame.to_csv(path, index=False)
    elif mutation == "hash":
        torch.save({"input_sha256": "different"}, run / "checkpoint.pt")
    else:
        path = run / "environment_estimates.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "reference_0"] = 99
        frame.to_csv(path, index=False)
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert reason in result["pairs"][0]["reason"]
    assert all(metric["n"] == 0 for metric in result["groups"][0]["metrics"].values())


def test_allowed_attack_diagnostics_do_not_hide_budget_changes(tmp_path):
    roots = pair(tmp_path)
    path = roots[1] / "taxibj/tsadc/0/config.resolved.yaml"
    config = yaml.safe_load(path.read_text())
    config["evaluation"].update(attack_harm_objective="mechanism_specific", attack_min_harm_rounds=2)
    path.write_text(yaml.safe_dump(config))
    result = analyze(roots)
    assert result["completion"]["valid"] == 1
    assert "evaluation.attack_harm_objective" in result["pairs"][0]["allowed_config_differences"]


def test_missing_declared_incomplete_and_invalid_counts_remain_visible(tmp_path):
    roots = pair(tmp_path)
    pair(tmp_path, seed=42)
    (roots[0] / "run_status.json").write_text(json.dumps({"runs": [
        {"dataset": "taxibj", "algorithm": "tsadc", "seed": 42, "status": "running"}]}))
    result = analyze(roots, seeds=[0, 42, 46])
    assert result["completion"] == {"expected": 3, "valid": 1, "missing": 2, "invalid": 0}
    assert len(pd.read_csv(roots[0] / "comparison/per_seed.csv")) == 3
    assert result["groups"][0]["metrics"]["delta_common_mae"]["n"] == 1
    json.dumps(result, allow_nan=False)


def test_native_is_separate_and_report_has_exactly_three_tables(tmp_path):
    roots = pair(tmp_path, algorithm="condition_alignment")
    result = analyze(roots, algorithms=["condition_alignment"])
    metrics = result["groups"][0]["metrics"]
    assert metrics["delta_native_mae"]["mean"] == .5
    report = (roots[0] / "comparison/REPORT.md").read_text()
    assert sum(line.startswith("| ---") for line in report.splitlines()) == 3
    assert "整个实验流程" in report
    assert "同一受污染训练的冻结 checkpoint" in report
    assert "原始观测" in report
    assert "target_bias" in report
    assert sorted(path.name for path in (roots[0] / "comparison").iterdir()) == ["REPORT.md", "per_seed.csv", "summary.csv", "summary.json"]


def test_cli_preserves_explicit_expected_grid(tmp_path, capsys):
    from scripts.analyze_attack_comparison import main
    roots = pair(tmp_path)
    result = main(["--no-attack", str(roots[0]), "--attack", "taxibj=" + str(roots[1]),
                   "--datasets", "taxibj", "--algorithms", "tsadc", "--seeds", "0", "42"])
    assert result["completion"]["expected"] == 2
    assert json.loads(capsys.readouterr().out)["valid"] == 1


def test_cross_seed_budget_mix_invalidates_aggregation_even_if_each_pair_matches(tmp_path):
    roots = pair(tmp_path)
    pair(tmp_path, seed=42)
    for root in roots:
        path = root / "taxibj/tsadc/42/config.resolved.yaml"
        config = yaml.safe_load(path.read_text())
        config["training"]["epochs"] = 1
        path.write_text(yaml.safe_dump(config))
    result = analyze(roots, seeds=[0, 42])
    assert result["completion"]["valid"] == 2
    assert result["groups"][0]["aggregation_status"] == "invalid"
    assert result["groups"][0]["metrics"]["delta_common_mae"]["n"] == 0
    assert any("across seeds" in warning for warning in result["warnings"])


def test_zero_joint_output_coverage_is_not_zero_mae(tmp_path):
    roots = pair(tmp_path)
    path = roots[1] / "taxibj/tsadc/0/environment_estimates.csv"
    frame = pd.read_csv(path)
    frame.loc[:1, "filtered_median_0"] = np.nan
    frame.to_csv(path, index=False)
    result = analyze(roots)
    metrics = result["groups"][0]["metrics"]
    assert metrics["delta_common_mae"] == dict(mean=None, std=None, n=0)
    assert metrics["no_attack_common_matched_coverage"]["mean"] == 0
    assert metrics["attack_common_coverage"]["mean"] == .25


def test_stale_evaluation_hash_invalidates_pair(tmp_path):
    roots = pair(tmp_path)
    path = roots[0] / "taxibj/tsadc/0/metrics.json"
    metrics = json.loads(path.read_text())
    metrics["input_sha256"] = "previous-evaluation"
    path.write_text(json.dumps(metrics))
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert "evaluated test input hash" in result["pairs"][0]["reason"]


def test_query_hash_and_full_test_query_coverage_are_verified(tmp_path):
    roots = pair(tmp_path)
    query_path = tmp_path / "queries.csv"
    pd.DataFrame(dict(sequence_id=["s"] * 8, task_id=["t"] * 8, timestamp=np.repeat(range(4), 2),
                      query_id=["q1", "q2"] * 4, reference_0=[2.] * 8)).to_csv(query_path, index=False)
    fingerprint = hashlib.sha256(query_path.read_bytes()).hexdigest()
    for root in roots:
        run = root / "taxibj/tsadc/0"
        config_path = run / "config.resolved.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["evaluation"]["queries_path"] = str(query_path)
        config_path.write_text(yaml.safe_dump(config))
        metrics_path = run / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["query_evaluation"] = {"method": "fixed_inverse_distance_common_postprocessing"}
        metrics_path.write_text(json.dumps(metrics))
        frame = pd.read_csv(run / "environment_estimates.csv").rename(columns={"filtered_median_0": "filtered_estimate"})
        frame["query_id"] = "q1"
        frame.to_csv(run / "query_estimates.csv", index=False)
        (root / "run_status.json").write_text(json.dumps({"datasets": {"taxibj": {"input_sha256": {"queries": fingerprint}}}}))
    result = analyze(roots)
    assert result["completion"]["valid"] == 1
    assert result["groups"][0]["metrics"]["no_attack_common_coverage"]["mean"] == .25
    query_path.write_text(query_path.read_text() + "\n")
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert "query input_sha256" in result["pairs"][0]["reason"]


def test_preserved_natural_fault_is_not_relabelled_normal(tmp_path):
    roots = pair(tmp_path)
    for root in roots:
        path = root / "taxibj/tsadc/0/splits/train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "original_label"] = 1
        # The no-injection arm incorrectly labels this original natural fault 0.
        frame.to_csv(path, index=False)
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert "original natural-fault labels" in result["pairs"][0]["reason"]


def test_same_no_attack_root_cannot_masquerade_as_attack_control(tmp_path):
    roots = pair(tmp_path)
    result = analyze([roots[0], roots[0]])
    assert result["completion"]["invalid"] == 1
    assert "attack arm cannot declare no-attack mode" in result["pairs"][0]["reason"]
    assert result["groups"][0]["metrics"]["delta_false_positive_rate"]["n"] == 0


def test_declared_attack_protocol_requires_real_training_injection(tmp_path):
    roots = pair(tmp_path)
    path = roots[1] / "taxibj/tsadc/0/splits/train.csv"
    frame = pd.read_csv(path)
    frame["value_0"], frame["injection_label"], frame["label"] = frame.baseline_0, 0, 0
    frame.to_csv(path, index=False)
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert "realized training and test injection" in result["pairs"][0]["reason"]


def test_attack_arm_disabling_one_split_is_rejected(tmp_path):
    roots = pair(tmp_path)
    path = roots[1] / "taxibj/tsadc/0/config.resolved.yaml"
    config = yaml.safe_load(path.read_text())
    config["corruption"]["calibration"] = {"protocol": "none"}
    path.write_text(yaml.safe_dump(config))
    result = analyze(roots)
    assert result["completion"]["invalid"] == 1
    assert "targeted_report_attack in calibration" in result["pairs"][0]["reason"]
