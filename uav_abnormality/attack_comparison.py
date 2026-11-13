"""Paired whole-pipeline comparisons, with immutable-input compatibility checks.

The no-injection arm is retrained and recalibrated. It is deliberately distinct
from baseline_predictions.csv, which replays an attacked checkpoint. A seed is
the statistical unit, and each difference uses identical report/output support.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .benchmark_analysis import DATASET_NAMES, METHOD_NAMES, _number, _statistics, _table

KEYS = ["sequence_id", "task_id", "timestamp", "source_id"]
ROUND = KEYS[:3]
SPLITS = ("train", "validation", "calibration", "test")
BASE_METRICS = ("false_positive_rate", "all_background_fpr", "scoring_coverage",
                "common_mae", "common_all_support_mae", "common_coverage", "common_matched_coverage",
                "native_mae", "native_all_support_mae", "native_coverage", "native_matched_coverage")
METRICS = tuple(side + "_" + metric for metric in BASE_METRICS for side in ("no_attack", "attack", "delta"))
IGNORED_CONFIG_PATHS = ("attack_mode", "corruption", "protocol_feasibility.min_scored_attacks_per_split",
                        "evaluation.attack_harm_objective", "evaluation.attack_min_harm_rounds")


def _json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _csv(path):
    frame = pd.read_csv(path, dtype={key: str for key in ("sequence_id", "task_id", "source_id", "query_id")})
    if "timestamp" in frame and not pd.api.types.is_numeric_dtype(frame.timestamp):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).astype("int64") / 1e9
    return frame


def _config_signature(config):
    result = copy.deepcopy(config)
    for path in IGNORED_CONFIG_PATHS:
        parent = result
        parts = path.split(".")
        for part in parts[:-1]:
            parent = parent.get(part, {})
        parent.pop(parts[-1], None)
    return result


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [p for key in sorted(set(left) | set(right)) for p in
                _differences(left.get(key), right.get(key), prefix + ("." if prefix else "") + key)]
    return [] if left == right else [prefix]


def _original_reports(frame):
    original = frame.copy()
    for col in [c for c in frame if c.startswith("value_")]:
        baseline = "baseline_" + col[len("value_"):]
        if baseline not in frame:
            raise ValueError("missing preserved original values: " + baseline)
        original[col] = frame[baseline]
    if "original_label" not in frame:
        raise ValueError("missing original_label; natural labels cannot be audited")
    original["label"] = frame.original_label
    drop = [c for c in original if c.startswith(("baseline_", "attack_", "fault_")) or
            c in ("original_label", "injection_label", "window_contains_attack", "evaluation_phase")]
    original = original.drop(columns=drop)
    if original.duplicated(KEYS).any():
        raise ValueError("duplicate report identities")
    return original.sort_values(KEYS).reset_index(drop=True).sort_index(axis=1)


def _checkpoint_hash(path):
    # These are project-generated local checkpoints, never downloaded pickles.
    import torch
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    result = checkpoint.get("input_sha256")
    if not result:
        raise ValueError("checkpoint missing original input_sha256")
    return result


def _query_path(config, run):
    value = config.get("evaluation", {}).get("queries_path")
    if not value:
        return None
    path = Path(value)
    return next((p for p in (path, run / path, Path(__file__).resolve().parents[1] / path) if p.is_file()), path)


def _validate_pair(no_run, attack_run, no_config, attack_config, no_status, attack_status, dataset):
    """Fail closed on training budgets, data changes, or non-injection mutations."""
    if no_config.get("attack_mode") != "none":
        raise ValueError("no-attack arm must explicitly declare attack_mode: none")
    if attack_config.get("attack_mode", "configured") != "configured":
        raise ValueError("attack arm cannot declare no-attack mode")
    for stage in SPLITS:
        if no_config.get("corruption", {}).get(stage, {}).get("protocol") != "none":
            raise ValueError("no-attack arm must disable injection in " + stage)
        if attack_config.get("corruption", {}).get(stage, {}).get("protocol") != "targeted_report_attack":
            raise ValueError("attack arm must configure targeted_report_attack in " + stage)
    difference = _differences(_config_signature(no_config), _config_signature(attack_config))
    if difference:
        raise ValueError("incompatible non-attack config: " + ", ".join(difference))
    raw_hash = _checkpoint_hash(no_run)
    if raw_hash != _checkpoint_hash(attack_run):
        raise ValueError("original input_sha256 differs")
    audits = {}
    for stage in SPLITS:
        clean, attack = _csv(no_run / "splits" / (stage + ".csv")), _csv(attack_run / "splits" / (stage + ".csv"))
        if "injection_label" not in clean or not clean.injection_label.eq(0).all():
            raise ValueError("no-attack arm contains injection or missing labels: " + stage)
        original_clean, original_attack = _original_reports(clean), _original_reports(attack)
        expected_labels = clean.original_label.replace(-1, 0) if no_config["corruption"][stage].get("assume_unlabelled_normal") else clean.original_label
        if not clean.label.equals(expected_labels.astype(clean.label.dtype)):
            raise ValueError("no-attack arm changed original natural-fault labels: " + stage)
        try:
            pd.testing.assert_frame_equal(original_clean, original_attack, check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
            for col in [c for c in clean if c.startswith("value_")]:
                np.testing.assert_allclose(clean[col], clean["baseline_" + col[len("value_"):]],
                                           rtol=0, atol=0, equal_nan=True)
        except AssertionError as exc:
            raise ValueError("original report identity/value/condition/reference differs: " + stage) from exc
        audits[stage] = {"reports": len(clean), "original_reports_equal": True,
                         "no_attack_injected_reports": 0,
                         "attack_injected_reports": int(attack.injection_label.eq(1).sum())}
    if any(audits[stage]["attack_injected_reports"] == 0 for stage in ("train", "test")):
        raise ValueError("attack arm requires realized training and test injection")
    query_hash = None
    if _query_path(no_config, no_run) is not None:
        fingerprints = []
        for status in (no_status, attack_status):
            fingerprint = status.get("datasets", {}).get(dataset, {}).get("input_sha256", {}).get("queries")
            if not fingerprint:
                raise ValueError("missing persisted query input_sha256")
            fingerprints.append(fingerprint)
        current_hash = hashlib.sha256(_query_path(no_config, no_run).read_bytes()).hexdigest()
        if fingerprints[0] != fingerprints[1] or fingerprints[0] != current_hash:
            raise ValueError("query input_sha256 differs or current query sidecar changed")
        query_hash = current_hash
    return dict(original_input_sha256=raw_hash, query_input_sha256=query_hash, splits=audits,
                allowed_config_differences=_differences(no_config, attack_config))


def _put(row, name, no_attack, attack):
    no_attack, attack = _number(no_attack), _number(attack)
    row["no_attack_" + name], row["attack_" + name] = no_attack, attack
    row["delta_" + name] = attack - no_attack if no_attack is not None and attack is not None else None


def _predictions(run):
    frame = _csv(run / "predictions.csv")
    valid = np.isfinite(pd.to_numeric(frame.score, errors="coerce")) & frame.prediction.isin([0, 1])
    if "evaluated" in frame:
        valid &= frame.evaluated.eq(True)
    frame = frame[valid]
    if frame.duplicated(KEYS).any():
        raise ValueError("duplicate scored report identities")
    return frame


def _fpr_pair(row, no_run, attack_run, test_reports):
    clean, attack = _predictions(no_run), _predictions(attack_run)
    if not clean.injection_label.eq(0).all():
        raise ValueError("no-attack predictions contain injected positives")
    _put(row, "all_background_fpr", clean.loc[clean.label.eq(0), "prediction"].mean(),
         attack.loc[attack.label.eq(0), "prediction"].mean())
    _put(row, "scoring_coverage", len(clean) / test_reports, len(attack) / test_reports)
    joined = clean.merge(attack, on=KEYS, suffixes=("_no_attack", "_attack"), validate="one_to_one")
    background = joined[joined.injection_label_attack.eq(0) & joined.label_attack.eq(0) & joined.label_no_attack.eq(0)]
    row["paired_background_reports"] = len(background)
    _put(row, "false_positive_rate", background.prediction_no_attack.mean(), background.prediction_attack.mean())


def _estimates(run, channel, metrics):
    if channel == "native":
        path, estimate, keys = run / "native_environment.csv", "estimate_0", ROUND + ["query_id"]
    elif metrics.get("query_evaluation", {}).get("method") == "fixed_inverse_distance_common_postprocessing":
        path, estimate, keys = run / "query_estimates.csv", "filtered_estimate", ROUND + ["query_id"]
    elif metrics.get("reference_evaluation", {}).get("environment", {}).get("method") == "common_report_median_postprocessing":
        path, estimate, keys = run / "environment_estimates.csv", "filtered_median_0", ROUND
    else:
        return None, None
    if not path.is_file():
        if channel == "native" and metrics.get("algorithm") != "condition_alignment":
            return None, None
        raise ValueError("missing persisted environment estimates: " + str(path))
    frame = _csv(path)
    if frame.duplicated(keys).any():
        raise ValueError("duplicate environment output identities")
    result = frame[keys].copy()
    result["estimate"] = pd.to_numeric(frame[estimate], errors="coerce")
    result["reference"] = pd.to_numeric(frame["reference_0"], errors="coerce") if "reference_0" in frame else np.nan
    return result, keys


def _environment_pair(row, channel, no_run, attack_run, no_metrics, attack_metrics, config):
    clean, keys = _estimates(no_run, channel, no_metrics)
    attack, attack_keys = _estimates(attack_run, channel, attack_metrics)
    if clean is None and attack is None:
        return
    if clean is None or attack is None or keys != attack_keys:
        raise ValueError("incompatible " + channel + " output estimator")
    test_rounds = _csv(no_run / "splits/test.csv")[ROUND].drop_duplicates()
    if "query_id" in keys and _query_path(config, no_run) is not None:
        queries = _csv(_query_path(config, no_run))
        denominator = len(queries.merge(test_rounds, on=ROUND, validate="many_to_one"))
    else:
        denominator = len(test_rounds)
    if denominator == 0:
        raise ValueError("empty environment coverage denominator")
    valid = [np.isfinite(frame.estimate) & np.isfinite(frame.reference) for frame in (clean, attack)]
    available = [np.isfinite(frame.estimate) for frame in (clean, attack)]
    _put(row, channel + "_coverage", available[0].sum() / denominator, available[1].sum() / denominator)
    _put(row, channel + "_all_support_mae", *((frame.loc[mask, "estimate"] - frame.loc[mask, "reference"]).abs().mean()
                                             for frame, mask in zip((clean, attack), valid)))
    merged = clean.merge(attack, on=keys, suffixes=("_no_attack", "_attack"), validate="one_to_one")
    references = np.isfinite(merged.reference_no_attack) & np.isfinite(merged.reference_attack)
    if not np.allclose(merged.loc[references, "reference_no_attack"], merged.loc[references, "reference_attack"],
                       rtol=1e-12, atol=1e-12):
        raise ValueError("reference differs between environment arms")
    matched = merged[references & np.isfinite(merged.estimate_no_attack) & np.isfinite(merged.estimate_attack)]
    row[channel + "_paired_outputs"] = len(matched)
    row[channel + "_full_test_outputs"] = denominator
    _put(row, channel + "_matched_coverage", len(matched) / denominator, len(matched) / denominator)
    _put(row, channel + "_mae", (matched.estimate_no_attack - matched.reference_no_attack).abs().mean(),
         (matched.estimate_attack - matched.reference_attack).abs().mean())


def _cell(group, metric, percent=False):
    stats = group["metrics"][metric]
    if not stats["n"]:
        return "— (n=0)"
    scale, digits = (100, 2) if percent else (1, 3)
    mean = format(stats["mean"] * scale, "." + str(digits) + "f")
    std = format(stats["std"] * scale, "." + str(digits) + "f") if stats["std"] is not None else "—"
    suffix = "" if stats["n"] == group["expected_seeds"] else " (n=" + str(stats["n"]) + ")"
    return mean + " ± " + std + suffix


def _report(summary):
    counts, groups = summary["completion"], summary["groups"]
    lines = ["# 无注入与有攻击的配对比较", "",
             "有效配对 **{valid}/{expected}**；不兼容或无效 {invalid}；缺失或未完成 {missing}。".format(**counts),
             "种子：`" + ", ".join(map(str, summary["seeds"])) + "`。各项为逐种子均值 ± 样本标准差（ddof=1）。"
             "差值固定为 **有攻击 − 无注入**：先计算每个 seed 的差值，再求均值和标准差；FPR 差值以百分点表示。"
             "未标 n 的单元格包含全部计划种子，缺失指标不补零。", "",
             "无注入组在训练、验证、校准、测试四阶段都关闭人工篡改，并重新训练和校准；"
             "原始观测及其中未经认证的自然故障保留。故这里衡量的是**整个实验流程切换攻击开关后的系统差异**，"
             "不是仅在测试时移除攻击的直接效应。已有 baseline_predictions.csv 使用同一受污染训练的冻结 checkpoint，"
             "属于另一种反事实，未拿来代替本次重新训练的无注入组。", "",
             "## 1. 同一批未篡改报告的 FPR", "",
             "双方均评分、攻击侧未篡改且双方协议标签为正常的同一报告集合。"
             "未注入背景按协议视为正常，并不代表已认证没有自然异常。"
             "无注入组全量背景 FPR 另存 CSV/JSON 的 all_background_fpr，不能与此处的配对分母混用。", ""]
    rows = [[DATASET_NAMES.get(g["dataset"], g["dataset"]), METHOD_NAMES.get(g["algorithm"], g["algorithm"])] +
            [_cell(g, side + "_false_positive_rate", True) for side in ("no_attack", "attack", "delta")]
            for g in groups]
    lines += _table(["数据集", "方法", "无注入 FPR %", "有攻击 FPR %", "配对 Δ 百分点"], rows)
    for channel, title in (("common", "2. 公共估计器：共同输出集合的 MAE"),
                           ("native", "3. Condition alignment 原生估计器：共同输出集合的 MAE")):
        lines += ["", "## " + title, "",
                  "MAE 在双方均有估计、参考值一致的相同查询/任务时刻交集上计算。"
                  "覆盖率依次为无注入／有攻击／可计算配对误差的交集，分母包括整个测试集的预热轮次。"
                  "TaxiBJ 单位为流量；Intel 为 °C，参考是留出传感器观测代理。Crowd 无独立环境参考，不报告 MAE。", ""]
        rows = []
        for group in groups:
            if channel == "native" and group["algorithm"] != "condition_alignment":
                continue
            if group["dataset"] == "crowd_temperature":
                continue
            identity = [DATASET_NAMES.get(group["dataset"], group["dataset"])]
            if channel == "common":
                identity += [METHOD_NAMES.get(group["algorithm"], group["algorithm"])]
            rows.append(identity + [_cell(group, side + "_" + channel + "_mae") for side in ("no_attack", "attack", "delta")] +
                        [" / ".join((_cell(group, "no_attack_" + channel + "_coverage", True),
                                     _cell(group, "attack_" + channel + "_coverage", True),
                                     _cell(group, "no_attack_" + channel + "_matched_coverage", True)))])
        lines += _table(["数据集"] + (["方法"] if channel == "common" else []) +
                        ["无注入 MAE", "有攻击 MAE", "配对 Δ", "覆盖率 %：无注入 / 有攻击 / 交集"], rows)
    lines += ["", "公共估计器固定为 TaxiBJ 同任务中位数、Intel 固定 IDW。原生估计器另含质量、信誉与条件对齐，"
              "不与公共估计器混为同一排名。上述结果是描述统计，不代表统计显著性。", "", "对照来源及实际攻击协议：", ""]
    for dataset in summary["datasets"]:
        kinds = sorted({kind for r in summary["pairs"] if r["dataset"] == dataset for kind in r.get("attack_kinds", [])})
        persistence = sorted({mode for r in summary["pairs"] if r["dataset"] == dataset for mode in r.get("attack_persistence", [])})
        lines.append("- " + DATASET_NAMES.get(dataset, dataset) + "：`" + summary["attack_roots"].get(dataset, "未指定") +
                     "`；" + ", ".join(kinds) + "；持续方式 " + ", ".join(persistence) + "。")
    lines += ["", "配对前核验原始输入哈希、四个 split 的报告身份/原始值/条件/参考/原始标签，以及非攻击配置。"
              "训练预算、模型、划分、窗口、评分、校准设置不允许变化；仅攻击开关、攻击覆盖检查与危害诊断配置允许不同。"
              "指标有效种子数、配对支持数、全部输出上的指标和逐对核验结论均保存在 summary.csv/json 与 per_seed.csv。"
              "无正例时不使用 AP、Recall 或 F1 评价检测能力。", ""]
    for warning in summary["warnings"]:
        lines.append("- " + warning)
    return "\n".join(lines) + "\n"


def analyze_attack_comparison(no_attack_root, attack_roots, expected_datasets=None,
                              expected_algorithms=None, expected_seeds=None, output=None):
    """Write three paper-facing paired tables plus CSV/JSON with missing counts."""
    no_attack_root = Path(no_attack_root)
    output = Path(output) if output is not None else no_attack_root / "comparison"
    plan = _json(no_attack_root / "run_plan.json", {})
    datasets = list(expected_datasets if expected_datasets is not None else plan.get("datasets", attack_roots))
    algorithms = list(expected_algorithms if expected_algorithms is not None else plan.get("algorithms", METHOD_NAMES))
    seeds = list(dict.fromkeys(int(s) for s in (expected_seeds if expected_seeds is not None else plan.get("seeds", [0, 42, 46, 47, 1024]))))
    no_status = _json(no_attack_root / "run_status.json", {})
    status_cache = {str(Path(root)): _json(Path(root) / "run_status.json", {}) for root in attack_roots.values()}
    pairs, rows = [], []
    for dataset, algorithm, seed in itertools.product(datasets, algorithms, seeds):
        no_run = no_attack_root / dataset / algorithm / str(seed)
        attack_root = Path(attack_roots[dataset]) if dataset in attack_roots else None
        attack_run = attack_root / dataset / algorithm / str(seed) if attack_root is not None else None
        row = dict(dataset=dataset, algorithm=algorithm, seed=seed, status="missing", reason="",
                   no_attack_path=str(no_run), attack_path=str(attack_run) if attack_run is not None else None,
                   **{name: None for name in METRICS})
        audit = dict(dataset=dataset, algorithm=algorithm, seed=seed)
        try:
            if attack_run is None:
                raise FileNotFoundError("attack root unspecified")
            attack_status = status_cache[str(attack_root)]
            for root_status, path in ((no_status, no_run), (attack_status, attack_run)):
                declared = next((r for r in root_status.get("runs", []) if
                                 (r.get("dataset"), r.get("algorithm"), r.get("seed")) == (dataset, algorithm, seed)), {})
                if declared.get("status") not in (None, "completed", "complete", "success", "ok"):
                    raise FileNotFoundError(str(path) + ": declared " + declared["status"])
            configs = [yaml.safe_load((path / "config.resolved.yaml").read_text()) for path in (no_run, attack_run)]
            metrics = [_json(path / "metrics.json") for path in (no_run, attack_run)]
            if any(not isinstance(m, dict) for m in metrics):
                raise FileNotFoundError("metrics.json missing")
            for path, config, metric in zip((no_run, attack_run), configs, metrics):
                if config.get("algorithm") != algorithm or config.get("seed") != seed or metric.get("algorithm") != algorithm:
                    raise ValueError("run identity differs from declared dataset/method/seed path")
                if metric.get("input_sha256") != hashlib.sha256((path / "splits/test.csv").read_bytes()).hexdigest():
                    raise ValueError("evaluated test input hash differs from persisted split")
            audit.update(_validate_pair(no_run, attack_run, *configs, no_status, attack_status, dataset))
            audit["attack_kinds"] = sorted({kind for stage in SPLITS for kind in configs[1].get("corruption", {}).get(stage, {}).get("kinds", [])})
            audit["attack_persistence"] = sorted({configs[1].get("corruption", {}).get(stage, {}).get("persistence", "finite_episode") for stage in SPLITS})
            signature = _config_signature(configs[1])
            signature.pop("seed", None)
            signature["corruption"] = configs[1].get("corruption")
            signature["original_input_sha256"] = audit["original_input_sha256"]
            audit["aggregation_signature"] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            _fpr_pair(row, no_run, attack_run, audit["splits"]["test"]["reports"])
            for channel in ("common", "native"):
                _environment_pair(row, channel, no_run, attack_run, *metrics, configs[0])
            row["status"] = "valid"
        except FileNotFoundError as exc:
            row["reason"] = str(exc)
        except (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError, EOFError, pd.errors.ParserError) as exc:
            row.update(status="invalid", reason=str(exc))
        if row["status"] != "valid":
            row.update({name: None for name in METRICS})
        audit.update(status=row["status"], reason=row["reason"])
        rows.append(row)
        pairs.append(audit)
    groups, warnings = [], []
    for dataset, algorithm in itertools.product(datasets, algorithms):
        subset = [r for r in rows if r["dataset"] == dataset and r["algorithm"] == algorithm]
        valid = [r for r in subset if r["status"] == "valid"]
        signatures = {a["aggregation_signature"] for a in pairs if a["dataset"] == dataset and
                      a["algorithm"] == algorithm and a["status"] == "valid"}
        aggregatable = valid if len(signatures) <= 1 else []
        if len(signatures) > 1:
            warnings.append(dataset + "/" + algorithm + ": inconsistent input/training/calibration/attack protocol across seeds; group not aggregated")
        groups.append(dict(dataset=dataset, algorithm=algorithm, expected_seeds=len(seeds), valid_pairs=len(valid),
                           invalid_pairs=sum(r["status"] == "invalid" for r in subset),
                           missing_pairs=sum(r["status"] == "missing" for r in subset),
                           aggregation_status="valid" if len(signatures) <= 1 else "invalid",
                           metrics={metric: _statistics([r[metric] for r in aggregatable]) for metric in METRICS}))
    summary = dict(schema_version=1, datasets=datasets, algorithms=algorithms, seeds=seeds,
                   no_attack_root=str(no_attack_root), attack_roots={k: str(v) for k, v in attack_roots.items()},
                   comparison="whole_pipeline_retrained_and_recalibrated", delta="attack minus no_attack, within seed",
                   fpr_support="same unchanged protocol-normal reports scored by both arms",
                   mae_support="same output identity with finite estimate in both arms and matching reference",
                   natural_faults="preserved; no-injection does not certify naturally clean observations",
                   statistics={"unit": "seed", "std": "sample standard deviation (ddof=1)", "missing": "null, never zero-imputed"},
                   compatibility_ignored_config_paths=list(IGNORED_CONFIG_PATHS),
                   completion=dict(expected=len(rows), valid=sum(r["status"] == "valid" for r in rows),
                                   invalid=sum(r["status"] == "invalid" for r in rows),
                                   missing=sum(r["status"] == "missing" for r in rows)),
                   groups=groups, pairs=pairs,
                   warnings=warnings + [r["dataset"] + "/" + r["algorithm"] + "/" + str(r["seed"]) + ": " + r["reason"]
                                        for r in rows if r["status"] != "valid"])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    flat = []
    for group in groups:
        row = {key: value for key, value in group.items() if key != "metrics"}
        for metric, stats in group["metrics"].items():
            row.update({metric + "_" + key: value for key, value in stats.items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    (output / "REPORT.md").write_text(_report(summary))
    return summary
