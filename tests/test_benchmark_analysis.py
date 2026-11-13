import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.benchmark_analysis import analyze_benchmarks


def write_run(root, dataset="taxibj", algorithm="tsadc", seed=0, **changes):
    directory = root / dataset / algorithm / str(seed)
    directory.mkdir(parents=True)
    metrics = dict(algorithm=algorithm, scored_reports=8, input_reports=10, tp=1, fn=1, fp=1, tn=5,
                   average_precision=.5, auroc=.6, f1=.5, recall=.5, false_positive_rate=1 / 6)
    metrics.update(changes)
    (directory / "metrics.json").write_text(json.dumps(metrics))
    return directory


def test_seed_statistics_do_not_pool_and_use_sample_standard_deviation(tmp_path):
    write_run(tmp_path, seed=0, average_precision=.2, scored_reports=10)
    write_run(tmp_path, seed=42, average_precision=.8, scored_reports=1000)
    summary = analyze_benchmarks(tmp_path)
    stats = summary["groups"][0]["metrics"]["average_precision"]
    assert stats == {"mean": .5, "std": pytest.approx(np.std([.2, .8], ddof=1)), "n": 2}
    assert summary["completion"] == dict(expected=2, completed=2, failed=0, pending=0)
    assert sorted(p.name for p in (tmp_path / "analysis").iterdir()) == ["REPORT.md", "per_seed.csv", "summary.csv", "summary.json"]
    csv = pd.read_csv(tmp_path / "analysis/summary.csv")
    assert csv.iloc[0].average_precision_n == 2
    assert csv.iloc[0].average_precision_std == pytest.approx(stats["std"])


def test_single_class_is_not_perfect_or_zero_detection_and_one_seed_std_missing(tmp_path):
    write_run(tmp_path, dataset="crowd_temperature", tp=0, fn=0, tn=8, fp=0, f1=0., recall=0.,
              average_precision=None, auroc=None, false_positive_rate=0.,
              calibration_summary={"retained": 3, "min_p": .25, "alarms_attainable": False})
    summary = analyze_benchmarks(tmp_path)
    group = summary["groups"][0]
    for name in ("average_precision", "auroc", "recall", "f1"):
        assert group["metrics"][name] == dict(mean=None, std=None, n=0)
    assert group["metrics"]["false_positive_rate"] == dict(mean=0., std=None, n=1)
    assert group["alarm_capable_seeds"] == 0
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "无法在指定 alpha 下告警" in report
    assert "无正例" in report
    assert "NaN" not in (tmp_path / "analysis/summary.json").read_text()


def test_plan_keeps_missing_failed_running_and_corrupt_runs_visible(tmp_path):
    (tmp_path / "run_plan.json").write_text(json.dumps(dict(datasets=["taxibj"], algorithms=["tsadc"], seeds=[0, 42, 46, 47, 1024])))
    write_run(tmp_path, seed=0)
    write_run(tmp_path, seed=42)
    write_run(tmp_path, seed=47)
    corrupt = write_run(tmp_path, seed=1024)
    (corrupt / "metrics.json").write_text("broken json")
    (tmp_path / "run_status.json").write_text(json.dumps({"runs": [
        dict(dataset="taxibj", algorithm="tsadc", seed=42, status="failed", error="out of memory"),
        dict(dataset="taxibj", algorithm="tsadc", seed=47, status="running")]}))
    summary = analyze_benchmarks(tmp_path)
    assert summary["completion"] == dict(expected=5, completed=1, failed=2, pending=2)
    rows = pd.read_csv(tmp_path / "analysis/per_seed.csv")
    assert dict(zip(rows.seed, rows.status)) == {0: "completed", 42: "failed", 46: "missing", 47: "running", 1024: "invalid"}
    assert rows[rows.seed == 42].iloc[0].reason == "out of memory"
    assert summary["groups"][0]["metrics"]["f1"]["n"] == 1


def test_environment_coverage_includes_warmup_and_native_stays_separate(tmp_path):
    directory = write_run(tmp_path, algorithm="condition_alignment", reference_evaluation={"environment": {
        "method": "common_report_median_postprocessing", "task_rounds": 3,
        "0": {"all": {"n": 3, "mae": 2.}, "filtered": {"n": 2, "mae": 1.}}}},
        native_environment={"query_rounds": 5, "coverage": .6, "0": {"n": 3, "mae": .7}})
    (directory / "splits").mkdir()
    pd.DataFrame(dict(sequence_id=["x"] * 5, task_id=["a"] * 5, timestamp=range(5))).to_csv(directory / "splits/test.csv", index=False)
    stats = analyze_benchmarks(tmp_path)["groups"][0]["metrics"]
    assert stats["common_coverage"]["mean"] == .4
    assert stats["common_conditional_coverage"]["mean"] == 2 / 3
    assert stats["common_mae"]["mean"] == 1.
    assert stats["native_mae"]["mean"] == .7
    assert stats["native_coverage"]["mean"] == .6


def test_idw_coverage_uses_only_full_test_queries(tmp_path):
    directory = write_run(tmp_path, dataset="intel_berkeley", query_evaluation={
        "method": "fixed_inverse_distance_common_postprocessing", "query_rounds": 2,
        "all": {"n": 2, "mae": 1.}, "filtered": {"n": 1, "mae": .4}})
    (directory / "splits").mkdir()
    pd.DataFrame(dict(sequence_id=["x"] * 3, task_id=["a"] * 3, timestamp=range(3))).to_csv(directory / "splits/test.csv", index=False)
    query_path = tmp_path / "queries.csv"
    pd.DataFrame(dict(sequence_id=["x"] * 8, task_id=["a"] * 8, timestamp=[0, 0, 1, 1, 2, 2, 3, 3])).to_csv(query_path, index=False)
    (directory / "config.resolved.yaml").write_text("evaluation:\n  queries_path: " + str(query_path) + "\n")
    stats = analyze_benchmarks(tmp_path)["groups"][0]["metrics"]
    assert stats["environment_rounds"]["mean"] == 6
    assert stats["common_coverage"]["mean"] == 1 / 6


def test_protection_averages_seeds_not_events_and_retains_actual_denominators(tmp_path):
    for seed, effective, protected in ((0, 1, 1), (42, 9, 0)):
        write_run(tmp_path, seed=seed, attack_evaluation={"attempt_events": 12, "feasible_events": 10,
            "assessed_events": 11, "effective_events": effective,
            "defended": {"protected_effective_events": protected, "protection_rate_on_effective_attempts": protected / effective}})
    group = analyze_benchmarks(tmp_path)["groups"][0]
    assert group["metrics"]["common_protection_rate"]["mean"] == .5
    assert group["event_totals"]["effective_events"] == 10
    assert group["event_totals"]["common_protected_events"] == 1
    assert group["metrics"]["native_mae"]["n"] == 0


def test_missing_reference_or_no_effective_attacks_never_becomes_zero(tmp_path):
    write_run(tmp_path, attack_evaluation={"attempt_events": 3, "effective_events": 0,
        "defended": {"protected_effective_events": 0, "protection_rate_on_effective_attempts": 0.}})
    stats = analyze_benchmarks(tmp_path)["groups"][0]["metrics"]
    assert stats["common_mae"] == dict(mean=None, std=None, n=0)
    assert stats["common_protection_rate"] == dict(mean=None, std=None, n=0)


def test_nonfinite_values_do_not_contaminate_json_and_mismatched_inputs_are_reported(tmp_path):
    write_run(tmp_path, algorithm="tsadc", average_precision=float("nan"), input_sha256="one")
    write_run(tmp_path, algorithm="prbtd", average_precision=float("inf"), input_sha256="two")
    result = analyze_benchmarks(tmp_path)
    assert all(group["metrics"]["average_precision"]["n"] == 0 for group in result["groups"])
    assert any("哈希不一致" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_explicit_expected_grid_preserves_entire_unavailable_dataset(tmp_path):
    write_run(tmp_path)
    summary = analyze_benchmarks(tmp_path, expected_datasets=["taxibj", "intel_berkeley"],
                                 expected_algorithms=["tsadc", "prbtd"], expected_seeds=[0, 42])
    assert summary["completion"] == dict(expected=8, completed=1, failed=0, pending=7)
    assert len(summary["groups"]) == 4
    intel = [g for g in summary["groups"] if g["dataset"] == "intel_berkeley"]
    assert all(g["metrics"]["false_positive_rate"]["n"] == 0 for g in intel)
    assert len(pd.read_csv(tmp_path / "analysis/per_seed.csv")) == 8


def test_cli_produces_the_same_report_from_an_existing_run(tmp_path, capsys):
    from scripts.analyze_report_benchmarks import main
    write_run(tmp_path)
    result = main(["--input", str(tmp_path), "--seeds", "0", "42"])
    printed = json.loads(capsys.readouterr().out)
    assert result["completion"]["expected"] == 2
    assert printed["completed"] == 1
    assert printed["report"] == str(tmp_path / "analysis/REPORT.md")


def test_report_exposes_downstream_degradation_and_deduplicates_contamination_seeds(tmp_path):
    write_run(tmp_path, algorithm="condition_alignment", reference_evaluation={"environment": {
        "method": "common_report_median_postprocessing", "task_rounds": 3,
        "0": {"all": {"n": 3, "mae": .2}, "filtered": {"n": 3, "mae": .7}}}})
    for algorithm in ("condition_alignment", "tsadc"):
        for seed in (0, 42):
            path = write_run(tmp_path, dataset="crowd_temperature", algorithm=algorithm, seed=seed,
                             tp=0, fn=0, fp=0, tn=8)
            (path / "split_manifest.json").write_text(json.dumps({"corruptions": {
                "validation": {"injected_reports": 0}, "calibration": {"injected_reports": 0 if seed == 0 else 2}}}))
    summary = analyze_benchmarks(tmp_path)
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "MAE 平均恶化" in report
    assert "FPR（%）=" in report
    assert "验证阶段 2/2 个种子实际篡改报告数为 0" in report
    assert "校准阶段 1/2 个种子实际篡改报告数为 0" in report
    assert summary["contamination"]["crowd_temperature"]["validation"]["known_seeds"] == 2


def test_mixed_harm_objectives_invalidate_entire_seed_group(tmp_path):
    legacy = write_run(tmp_path, seed=0, attack_evaluation={"attempt_events": 3})
    # Current config cannot retrospectively change the semantics of saved metrics.
    (legacy / "config.resolved.yaml").write_text("evaluation:\n  attack_harm_objective: instantaneous_error\n")
    write_run(tmp_path, seed=42, attack_evaluation={"harm_objective": "instantaneous_error", "attempt_events": 3})
    result = analyze_benchmarks(tmp_path)
    group = result["groups"][0]
    assert result["completion"]["completed"] == 2
    assert result["invalid_aggregation_groups"] == 1
    assert group["aggregation_status"] == "invalid"
    assert group["aggregation_reason"] == "mixed_harm_objectives"
    assert group["harm_objectives"] == ["instantaneous_error", "mechanism_specific"]
    assert group["harm_objective"] is None
    assert all(metric["n"] == 0 for metric in group["metrics"].values())
    assert group["event_totals"]["attempt_events"] is None
    csv = pd.read_csv(tmp_path / "analysis/summary.csv")
    assert csv.aggregation_status.iloc[0] == "invalid"
    seeds = pd.read_csv(tmp_path / "analysis/per_seed.csv").set_index("seed")
    assert seeds.loc[0, "harm_objective"] == "mechanism_specific"
    assert seeds.loc[42, "harm_objective"] == "instantaneous_error"
    assert "整组聚合无效" in (tmp_path / "analysis/REPORT.md").read_text()


def test_crowd_proxy_table_uses_seed_means_and_deduplicates_query_rounds(tmp_path):
    for seed, count, shifted, all_delta, filtered_delta in (
            (0, 2, 1, [0., 2.], [1., 3.]), (42, 4, 3, [4., 8.], [0., 2.])):
        proxy = {"is_truth_error": False, "channels": {
            "undefended": {"assessed_events": count, "shifted_events": shifted,
                           "shifted_assessed_event_rate": shifted / count},
            "defended": {"assessed_events": count, "shifted_events": shifted,
                         "shifted_assessed_event_rate": shifted / count}}}
        directory = write_run(tmp_path, dataset="crowd_temperature", seed=seed,
                              attack_evaluation={"harm_objective": "instantaneous_error", "attempt_events": count,
                                                 "assessed_events": 0, "effective_events": 0,
                                                 "displacement_proxy": proxy})
        pd.DataFrame({"sequence_id": ["s"] * 3, "task_id": ["t"] * 3,
                      "timestamp": [0., 1., 0.], "query_id": ["q"] * 3,
                      "undefended_displacement": all_delta + [all_delta[0]],
                      "defended_displacement": filtered_delta + [filtered_delta[0]]}).to_csv(
                          directory / "attack_rounds.csv", index=False)
        (directory / "split_manifest.json").write_text(json.dumps({"corruptions": {
            stage: {"injected_reports": shifted} for stage in ("train", "validation", "calibration", "test")}}))
    result = analyze_benchmarks(tmp_path)
    group = result["groups"][0]
    assert group["harm_objective"] == "instantaneous_error"
    assert group["metrics"]["common_mae"]["n"] == 0
    assert group["metrics"]["common_protection_rate"]["n"] == 0
    assert group["metrics"]["undefended_attack_mean_absolute_displacement"] == {
        "mean": 3.5, "std": pytest.approx(np.std([1., 6.], ddof=1)), "n": 2}
    assert group["metrics"]["common_attack_mean_absolute_displacement"]["mean"] == 1.5
    assert group["metrics"]["common_proxy_shifted_assessed_event_rate"]["mean"] == .625
    assert group["event_totals"]["common_proxy_shifted_events"] == 4
    assert group["event_totals"]["common_proxy_assessed_events"] == 6
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "Crowd 无独立参考时的位移代理" in report
    assert "不是对真实温度的误差，也不是保护率" in report
    assert "位移事件率只以各自可判定事件为分母" in report
    assert "不能据不同覆盖集合下较小的位移均值推断防护更好" in report
    assert "危害及保护率未知" in report
    assert "62.50 ± 17.68 (n=2/2)" in report
    assert "Crowd 全阶段污染" in report
    assert "训练、验证、校准、测试均实际产生了篡改报告" in report
    assert "Crowd 数据限制" not in report
    csv = pd.read_csv(tmp_path / "analysis/summary.csv")
    assert csv.harm_objective.iloc[0] == "instantaneous_error"
    assert csv.common_proxy_shifted_events_sum.iloc[0] == 4
    per_seed = pd.read_csv(tmp_path / "analysis/per_seed.csv")
    assert per_seed.warnings.str.contains("harm_unassessed").all()
    assert not per_seed.warnings.str.contains("no_effective_events").any()
    json.dumps(result, allow_nan=False)


def test_assessed_but_ineffective_attacks_are_distinct_from_unknown_harm(tmp_path):
    write_run(tmp_path, attack_evaluation={"harm_objective": "instantaneous_error", "attempt_events": 3,
                                          "assessed_events": 3, "effective_events": 0})
    analyze_benchmarks(tmp_path)
    row = pd.read_csv(tmp_path / "analysis/per_seed.csv").iloc[0]
    assert "no_effective_events" in row.warnings
    assert "harm_unassessed" not in row.warnings


def test_legacy_results_are_not_reinterpreted_as_new_proxy_data(tmp_path):
    directory = write_run(tmp_path, dataset="crowd_temperature", attack_evaluation={"attempt_events": 1})
    pd.DataFrame({"sequence_id": ["s"], "task_id": ["t"], "timestamp": [0.], "query_id": ["q"],
                  "undefended_displacement": [100.]}).to_csv(directory / "attack_rounds.csv", index=False)
    result = analyze_benchmarks(tmp_path)
    group = result["groups"][0]
    assert group["harm_objective"] == "mechanism_specific"
    assert group["metrics"]["undefended_attack_mean_absolute_displacement"]["n"] == 0
    assert "### 3a." not in (tmp_path / "analysis/REPORT.md").read_text()


def test_unknown_persisted_harm_objective_is_invalid(tmp_path):
    write_run(tmp_path, attack_evaluation={"harm_objective": "unknown_rule"})
    result = analyze_benchmarks(tmp_path)
    assert result["completion"]["failed"] == 1
    assert result["groups"][0]["metrics"]["average_precision"]["n"] == 0


def test_intentional_no_attack_report_focuses_on_fpr_and_sensing(tmp_path):
    (tmp_path / "run_plan.json").write_text(json.dumps({"datasets": ["taxibj"],
        "algorithms": ["condition_alignment"], "seeds": [0, 42], "attack_mode": "none"}))
    for seed, fp in ((0, 1), (42, 2)):
        directory = write_run(tmp_path, algorithm="condition_alignment", seed=seed, tp=0, fn=0, fp=fp,
            tn=8 - fp, average_precision=None, auroc=None, f1=0., recall=0., false_positive_rate=fp / 8,
            reference_evaluation={"environment": {"method": "common_report_median_postprocessing", "task_rounds": 3,
                "0": {"all": {"n": 3, "mae": .2}, "filtered": {"n": 3, "mae": .3}}}},
            native_environment={"query_rounds": 3, "coverage": 1., "0": {"mae": .4}},
            calibration_summary={"retained": 40, "min_p": 1 / 41, "alarms_attainable": True})
        (directory / "config.resolved.yaml").write_text("attack_mode: none\n")
        (directory / "split_manifest.json").write_text(json.dumps({"corruptions": {
            stage: {"injected_reports": 0} for stage in ("train", "validation", "calibration", "test")}}))
    result = analyze_benchmarks(tmp_path)
    assert result["attack_mode"] == "none"
    assert result["completion"]["completed"] == 2
    assert not result["warnings"]
    group = result["groups"][0]
    assert group["attack_mode"] == "none"
    for metric in ("average_precision", "auroc", "f1", "recall"):
        assert group["metrics"][metric] == {"mean": None, "std": None, "n": 0}
    assert group["metrics"]["false_positive_rate"] == {"mean": .1875, "std": pytest.approx(np.std([.125, .25], ddof=1)), "n": 2}
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "没有注入正例是本次对照的设计，不是实验失败" in report
    assert "全阶段无攻击与全阶段受攻击的差异包含训练、校准和测试污染的总影响" in report
    assert "## 3. 攻击危害与防护" not in report
    assert "原生保护 %" not in report
    assert "训练数据解释为干净" not in report
    assert "过滤后−无过滤 MAE" in report
    assert pd.read_csv(tmp_path / "analysis/summary.csv").attack_mode.tolist() == ["none"]
    rows = pd.read_csv(tmp_path / "analysis/per_seed.csv")
    assert rows.notes.str.contains("no_attacks_by_design").all()
    assert rows.warnings.isna().all()


def test_clean_mode_is_persisted_without_plan_and_does_not_claim_crowd_pollution(tmp_path):
    directory = write_run(tmp_path, dataset="crowd_temperature", tp=0, fn=0, tn=8, fp=0)
    (directory / "config.resolved.yaml").write_text("attack_mode: none\n")
    (directory / "split_manifest.json").write_text(json.dumps({"corruptions": {
        stage: {"injected_reports": 0} for stage in ("train", "validation", "calibration", "test")}}))
    result = analyze_benchmarks(tmp_path)
    assert result["attack_mode"] == "none"
    report = (tmp_path / "analysis/REPORT.md").read_text()
    assert "训练、验证、校准、测试均不注入攻击" in report
    assert "Crowd 缺少独立温度参考" in report
    assert "这些阶段执行了攻击调度" not in report
    assert "位移代理表" not in report
    assert not any("无正例" in warning or "保护率" in warning for warning in result["warnings"])


def test_clean_and_attacked_seeds_must_not_be_averaged(tmp_path):
    directory = write_run(tmp_path, seed=0, tp=0, fn=0, tn=8, fp=0)
    (directory / "config.resolved.yaml").write_text("attack_mode: none\n")
    write_run(tmp_path, seed=42)
    result = analyze_benchmarks(tmp_path)
    group = result["groups"][0]
    assert group["aggregation_status"] == "invalid"
    assert group["aggregation_reason"] == "mixed_attack_modes"
    assert all(metric["n"] == 0 for metric in group["metrics"].values())
