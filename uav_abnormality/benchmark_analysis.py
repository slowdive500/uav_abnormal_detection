"""Paper-facing, seed-wise summaries of report benchmarks.

This module never pools report-level predictions across seeds. A seed is the
statistical unit; unavailable metrics stay unavailable, including single-class
detection experiments and experiments without an environmental reference.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROUND = ["sequence_id", "task_id", "timestamp"]
METRICS = (
    "average_precision", "auroc", "f1", "recall", "false_positive_rate",
    "scoring_coverage", "positive_reports", "negative_reports",
    "common_mae", "common_unfiltered_mae", "common_mae_change", "common_coverage",
    "common_conditional_coverage", "common_estimated_rounds", "environment_rounds",
    "native_mae", "native_coverage", "attempt_events", "feasible_events",
    "assessed_events", "effective_events", "effective_assessed_event_rate",
    "detected_event_rate", "detected_feasible_event_rate", "common_protection_rate",
    "common_protected_events", "common_assessed_events", "native_protection_rate",
    "native_protected_events", "background_fpr", "recovery_fpr",
    "calibration_pool", "calibration_min_p", "alarms_attainable",
    "train_injected_reports", "validation_injected_reports", "calibration_injected_reports", "test_injected_reports",
    *tuple(prefix + "_" + metric for prefix in ("undefended", "common", "native") for metric in (
        "proxy_assessed_events", "proxy_shifted_events", "proxy_shifted_assessed_event_rate",
        "attack_mean_absolute_displacement")),
)
DATASET_NAMES = {"crowd_temperature": "Crowd", "taxibj": "TaxiBJ", "intel_berkeley": "Intel"}
METHOD_NAMES = {"condition_alignment": "Condition alignment", "prbtd": "PRBTD", "tsadc": "TSADC"}


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _ratio(numerator, denominator):
    numerator, denominator = _number(numerator), _number(denominator)
    return numerator / denominator if numerator is not None and denominator and denominator > 0 else None


def _round_table(path):
    frame = pd.read_csv(path, usecols=ROUND, dtype={"sequence_id": str, "task_id": str})
    if not pd.api.types.is_numeric_dtype(frame.timestamp):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise").astype("int64") / 1e9
    return frame


def _environment_rounds(run_path, metrics, common_method):
    """Use the entire test split, including warmup, as the coverage denominator."""
    test_path = run_path / "splits" / "test.csv"
    if not test_path.is_file():
        return _number(_get(metrics, "native_environment", "query_rounds"))
    raw_rounds = _round_table(test_path).drop_duplicates()
    if common_method == "task_median":
        return float(len(raw_rounds))
    if common_method == "fixed_inverse_distance":
        config_path = run_path / "config.resolved.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
        query_path = _get(config, "evaluation", "queries_path")
        if query_path:
            candidate = Path(query_path)
            candidates = [candidate, run_path / candidate, Path(__file__).resolve().parents[1] / candidate]
            candidate = next((p for p in candidates if p.is_file()), None)
            if candidate is not None:
                queries = _round_table(candidate)
                return float(len(queries.merge(raw_rounds, on=ROUND, how="inner", validate="many_to_one")))
        return _number(_get(metrics, "native_environment", "query_rounds"))
    return None


def _harm_objective(metrics):
    attack = metrics.get("attack_evaluation")
    if not isinstance(attack, dict) or not attack or attack.get("status") == "not_enabled":
        return None
    # Old persisted results used mechanism-specific gates. Never relabel them
    # according to a current config, which may have changed since evaluation.
    objective = attack.get("harm_objective", "mechanism_specific")
    if objective not in ("mechanism_specific", "instantaneous_error"):
        raise ValueError("Unknown persisted attack harm_objective: " + str(objective))
    return objective


def _displacement_proxy(run_path, attack, values, warnings):
    proxy = attack.get("displacement_proxy", {})
    if proxy.get("is_truth_error") is not False:
        return
    channels = (("undefended", "undefended"), ("common", "defended"), ("native", "native"))
    for prefix, channel in channels:
        for metric in ("assessed_events", "shifted_events", "shifted_assessed_event_rate"):
            values[prefix + "_proxy_" + metric] = _number(_get(proxy, "channels", channel, metric))
    path = run_path / "attack_rounds.csv"
    if not path.is_file():
        return
    try:
        keys = ROUND + ["query_id"]
        frame = pd.read_csv(path, usecols=lambda name: name in keys or name.endswith("_displacement"),
                            dtype={"sequence_id": str, "task_id": str, "query_id": str})
        if frame.empty:
            return
        for prefix, channel in channels:
            column = channel + "_displacement"
            if column not in frame:
                continue
            # Events can overlap. Count each sensing output once rather than
            # giving overlapping events extra weight in a seed's mean.
            if (frame.groupby(keys)[column].nunique(dropna=False) > 1).any():
                warnings.append("inconsistent_displacement_for_same_query_round:" + channel)
                continue
            displacement = pd.to_numeric(frame.drop_duplicates(keys)[column], errors="coerce")
            displacement = displacement[np.isfinite(displacement)]
            values[prefix + "_attack_mean_absolute_displacement"] = (
                float(displacement.abs().mean()) if len(displacement) else None)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        warnings.append("displacement_proxy_read_error:" + str(exc))


def _attack_mode(run_path, planned_mode="configured"):
    config_path = run_path / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    mode = config.get("attack_mode", planned_mode)
    if mode not in ("configured", "none"):
        raise ValueError("Unknown persisted attack_mode: " + str(mode))
    if planned_mode == "none" and mode != "none":
        raise ValueError("Saved run attack_mode disagrees with the no-attack run plan")
    return mode


def _extract(run_path, metrics, attack_mode="configured"):
    values = {name: None for name in METRICS}
    warnings = []
    for name in ("average_precision", "auroc", "f1", "recall", "false_positive_rate", "scoring_coverage"):
        values[name] = _number(metrics.get(name))
    if values["scoring_coverage"] is None:
        values["scoring_coverage"] = _ratio(metrics.get("scored_reports"), metrics.get("input_reports"))
    tp, fn, fp, tn = [_number(metrics.get(k)) for k in ("tp", "fn", "fp", "tn")]
    positive = tp + fn if tp is not None and fn is not None else None
    negative = fp + tn if fp is not None and tn is not None else None
    values.update(positive_reports=positive, negative_reports=negative)
    if positive == 0:
        if attack_mode != "none":
            warnings.append("no_scored_positives")
        for name in ("average_precision", "auroc", "f1", "recall"):
            values[name] = None
    if negative == 0:
        warnings.append("no_scored_negatives")
        values["false_positive_rate"] = values["auroc"] = None
    common_method = None
    query = metrics.get("query_evaluation", {})
    environment = _get(metrics, "reference_evaluation", "environment") or {}
    if query.get("method") == "fixed_inverse_distance_common_postprocessing":
        common_method, common = "fixed_inverse_distance", query
        scored_rounds = query.get("query_rounds")
    elif environment.get("method") == "common_report_median_postprocessing":
        common_method, common = "task_median", environment.get("0", {})
        scored_rounds = environment.get("task_rounds")
    else:
        common, scored_rounds = {}, None
        warnings.append("no_environment_reference")
    values["common_mae"] = _number(_get(common, "filtered", "mae"))
    values["common_unfiltered_mae"] = _number(_get(common, "all", "mae"))
    if values["common_mae"] is not None and values["common_unfiltered_mae"] is not None:
        values["common_mae_change"] = values["common_mae"] - values["common_unfiltered_mae"]
    values["common_estimated_rounds"] = _number(_get(common, "filtered", "n"))
    values["common_conditional_coverage"] = _ratio(values["common_estimated_rounds"], scored_rounds)
    if common_method:
        try:
            values["environment_rounds"] = _environment_rounds(run_path, metrics, common_method)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            warnings.append("environment_denominator_error:" + str(exc))
        values["common_coverage"] = _ratio(values["common_estimated_rounds"], values["environment_rounds"])
        if values["environment_rounds"] is None:
            warnings.append("full_test_environment_denominator_unavailable")
    native = metrics.get("native_environment", {})
    values["native_mae"] = _number(_get(native, "0", "mae"))
    values["native_coverage"] = _number(native.get("coverage"))
    attack = metrics.get("attack_evaluation", {})
    _displacement_proxy(run_path, attack, values, warnings)
    for name in ("attempt_events", "feasible_events", "assessed_events", "effective_events", "effective_assessed_event_rate"):
        values[name] = _number(attack.get(name))
    for name in ("detected_event_rate", "detected_feasible_event_rate"):
        values[name] = _number(_get(attack, "detection", name))
    for prefix, channel in (("common", "defended"), ("native", "native")):
        values[prefix + "_protection_rate"] = _number(_get(attack, channel, "protection_rate_on_effective_attempts"))
        values[prefix + "_protected_events"] = _number(_get(attack, channel, "protected_effective_events"))
    values["common_assessed_events"] = _number(_get(attack, "defended", "assessed_events"))
    if values["effective_events"] == 0:
        values["common_protection_rate"] = values["native_protection_rate"] = None
        if values["assessed_events"] == 0:
            warnings.append("harm_unassessed")
        elif values["assessed_events"] is not None and values["assessed_events"] > 0:
            warnings.append("no_effective_events")
    for phase in ("background", "recovery"):
        values[phase + "_fpr"] = _number(_get(attack, "false_positive_rate", phase, "fpr"))
    calibration = metrics.get("calibration_summary") or _read_json(run_path / "calibration.json", {})
    values["calibration_pool"] = _number(calibration.get("retained"))
    values["calibration_min_p"] = _number(calibration.get("min_p"))
    values["alarms_attainable"] = _number(calibration.get("alarms_attainable"))
    if values["alarms_attainable"] == 0:
        warnings.append("calibration_cannot_alarm")
    manifest = _read_json(run_path / "split_manifest.json", {})
    for stage in ("train", "validation", "calibration", "test"):
        values[stage + "_injected_reports"] = _number(_get(manifest, "corruptions", stage, "injected_reports"))
    return values, common_method, warnings


def _statistics(values):
    available = [value for value in (_number(v) for v in values) if value is not None]
    return {"mean": float(np.mean(available)) if available else None,
            "std": float(np.std(available, ddof=1)) if len(available) > 1 else None,
            "n": len(available)}


def _cell(group, metric, percent=False, digits=3, show_n=False):
    stats = group["metrics"][metric]
    if not stats["n"]:
        return "— (n=0/" + str(group["expected_seeds"]) + ")"
    scale = 100 if percent else 1
    digits = 2 if percent else digits
    mean = format(stats["mean"] * scale, "." + str(digits) + "f")
    std = format(stats["std"] * scale, "." + str(digits) + "f") if stats["std"] is not None else "—"
    note = (" (n=" + str(stats["n"]) + "/" + str(group["expected_seeds"]) + ")"
            if show_n or stats["n"] < group["expected_seeds"] else "")
    return mean + " ± " + std + note


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows]


def _concise_warnings(warnings):
    """Combine the same limitation across methods instead of repeating prose."""
    grouped = {}
    for warning in warnings:
        label, separator, message = warning.partition("：")
        if separator:
            grouped.setdefault(message, []).append(label)
        else:
            grouped.setdefault(warning, [])
    by_labels = {}
    for message, labels in grouped.items():
        by_labels.setdefault(tuple(labels), []).append(message)
    return [("、".join(labels) + "：" if labels else "") + " ".join(messages) for labels, messages in by_labels.items()]


def _no_attack_report(summary):
    groups = summary["groups"]
    lines = ["# 无攻击对照：跨种子实验结果", "",
             "实验完成 **{completed}/{expected}**；失败 {failed}，缺失或未完成 {pending}。".format(**summary["completion"]),
             "种子：`" + ", ".join(map(str, summary["seeds"])) + "`。各单元格为逐种子指标的**均值 ± 样本标准差**（ddof=1）。"
             "未标注 n 的指标使用全部 " + str(len(summary["seeds"])) + " 个种子；n=1 时标准差不可估计，— 表示不可计算，绝不补零。",
             "训练、验证、校准、测试均不注入攻击。未注入背景按协议视作正常，原始自然异常未经额外认证。"
             "因此这里的 FPR 是该背景假设下的误报率，不能证明原始数据绝对干净。",
             "没有注入正例是本次对照的设计，不是实验失败。AP、AUROC、F1、Recall 不适用，CSV/JSON 保持 null；"
             "攻击事件、危害和保护率也不适用，不输出空表。", "",
             "## 1. 无攻击时的误报与评分覆盖", "",
             "FPR 越低越好；覆盖率是可评分报告/全部测试报告。百分数列的标准差以百分点计。", ""]
    lines += _table(["数据集", "方法", "FPR % ↓", "评分覆盖 %"], [
        [DATASET_NAMES.get(g["dataset"], g["dataset"]), METHOD_NAMES.get(g["algorithm"], g["algorithm"]),
         _cell(g, "false_positive_rate", True), _cell(g, "scoring_coverage", True)] for g in groups])
    lines += ["", "## 2. 相同后处理下的联合感知", "",
              "TaxiBJ 使用同任务中位数，Intel 使用固定 IDW。MAE 单位分别为流量、温度 °C，只用成功估计轮次。"
              "覆盖率分母包括整个测试集的预热轮次。无过滤列保留相同可评分轮次上的全部报告；"
              "过滤后−无过滤 MAE 的差值先逐种子计算。低 MAE 必须结合覆盖率和估计时刻集合判断。", ""]
    rows = [[DATASET_NAMES.get(g["dataset"], g["dataset"]), METHOD_NAMES.get(g["algorithm"], g["algorithm"]),
             _cell(g, "common_unfiltered_mae"), _cell(g, "common_mae"), _cell(g, "common_mae_change"),
             _cell(g, "common_coverage", True)] for g in groups
            if g["metrics"]["common_mae"]["n"] or g["metrics"]["common_unfiltered_mae"]["n"]]
    lines += (_table(["数据集", "方法", "无过滤 MAE", "过滤后 MAE ↓", "过滤后−无过滤 MAE", "全测试轮覆盖 %"], rows)
              if rows else ["本次运行没有可用独立环境参考，故不列 MAE。"])
    if "crowd_temperature" in summary["datasets"]:
        lines += ["", "Crowd 缺少独立温度参考，不能计算真实温度 MAE，也不能把节点一致性当作感知准确率。"]
    aligned = [g for g in groups if g["algorithm"] == "condition_alignment"]
    if aligned:
        lines += ["", "## 3. Condition alignment 原生感知与校准", "",
                  "原生估计器包含质量与信誉加权，与公共后处理不同，单独报告。", ""]
        lines += _table(["数据集", "原生 MAE ↓", "全测试轮覆盖 %", "校准池单元", "可告警种子/已知种子"], [
            [DATASET_NAMES.get(g["dataset"], g["dataset"]), _cell(g, "native_mae"), _cell(g, "native_coverage", True),
             _cell(g, "calibration_pool", digits=1),
             str(g["alarm_capable_seeds"]) + "/" + str(g["calibration_known_seeds"])] for g in aligned])
    lines += ["", "## 比较口径", "",
              "与攻击实验比较时，应按相同数据集、方法、种子配对，检查数据、划分、窗口、模型和训练预算一致。"
              "全阶段无攻击与全阶段受攻击的差异包含训练、校准和测试污染的总影响，不能解释成仅测试时攻击的因果效应。"
              "配对差值的标准差应由每个种子的差值计算，不能用两组标准差相减。"]
    lines += ["- " + warning for warning in _concise_warnings(summary["warnings"])]
    lines += ["", "直接阅读本页；`summary.csv` / `summary.json` 保存均值、标准差、有效种子数及 attack_mode，"
              "`per_seed.csv` 保存配对分析所需的逐种子指标。", ""]
    return "\n".join(lines)


def _report(summary):
    if summary.get("attack_mode") == "none":
        return _no_attack_report(summary)
    completion = summary["completion"]
    groups = summary["groups"]
    lines = ["# 跨种子实验结果", "", "实验完成 **{completed}/{expected}**；失败 {failed}，缺失或未完成 {pending}。".format(**completion),
             "种子：`" + ", ".join(map(str, summary["seeds"])) + "`。各单元格是逐种子指标的**均值 ± 样本标准差**（ddof=1），n 是该指标有效种子数。",
             "未标注 n 的单元格使用全部 " + str(len(summary["seeds"])) + " 个种子；仅不足时附注 n/计划种子数，完整 n 同时保存在 CSV/JSON 中。",
             "n=1 时标准差不可估计；— 表示指标不可计算，绝不以 0 替代。百分数列的标准差也以百分点计。", "",
             "## 1. 报告级异常检测", "", "AP、AUROC、F1、Recall 越高越好，FPR 越低越好。覆盖率是可评分报告/全部测试报告；不做 point adjustment。正例是注入篡改的报告，是否达到任务危害门槛另见表 3；未注入背景按实验协议视作正常，原始自然异常未额外认证。", ""]
    rows = []
    for group in groups:
        rows.append([DATASET_NAMES.get(group["dataset"], group["dataset"]), METHOD_NAMES.get(group["algorithm"], group["algorithm"])] +
                    [_cell(group, name, name in ("recall", "false_positive_rate", "scoring_coverage")) for name in
                     ("average_precision", "auroc", "f1", "recall", "false_positive_rate", "scoring_coverage")])
    lines += _table(["数据集", "方法", "AP ↑", "AUROC ↑", "F1 ↑", "Recall % ↑", "FPR % ↓", "评分覆盖 %"], rows)
    lines += ["", "## 2. 相同后处理下的联合感知", "", "TaxiBJ 使用同任务中位数，Intel 使用固定 IDW。MAE 使用原始单位（TaxiBJ 流量，Intel 温度 °C），只在成功估计的轮次计算；覆盖率分母包括整个测试集的预热轮次。无过滤列是相同可评分轮次上保留全部报告的对照。低 MAE 必须结合覆盖率判断，覆盖集合不同不能直接推断整体更优。", ""]
    rows = []
    for group in groups:
        if group["metrics"]["common_mae"]["n"] or group["metrics"]["common_unfiltered_mae"]["n"]:
            rows.append([DATASET_NAMES.get(group["dataset"], group["dataset"]), METHOD_NAMES.get(group["algorithm"], group["algorithm"]),
                         _cell(group, "common_unfiltered_mae"), _cell(group, "common_mae"), _cell(group, "common_coverage", True)])
    if rows:
        lines += _table(["数据集", "方法", "无过滤 MAE", "过滤后 MAE ↓", "全测试轮覆盖 %"], rows)
    else:
        lines += ["本次运行没有可用的独立环境参考，故不列 MAE 表；Crowd 的输出变化见下方位移代理表。"]
    lines += ["", "## 3. 攻击危害与防护", "", "事件数为各有效种子的总和，比例仍先逐种子计算再求均值和标准差。有效事件指在固定无防御估计器上达到危害门槛；保护率以全部有效事件为分母，无法输出估计不算保护成功。可评危害事件为 0 时，有效事件 0 仅表示没有已确认事件，实际危害仍未知。检出率分母是所有尝试（含不可行尝试），所以它不等同于报告 Recall。", ""]
    rows = []
    for group in groups:
        totals = group["event_totals"]
        counts = "/".join(str(totals.get(key)) if totals.get(key) is not None else "—" for key in ("attempt_events", "feasible_events", "assessed_events", "effective_events"))
        protection = _cell(group, "common_protection_rate", True)
        if totals["effective_events"]:
            protection += "；" + str(totals["common_protected_events"]) + "/" + str(totals["effective_events"])
        rows.append([DATASET_NAMES.get(group["dataset"], group["dataset"]), METHOD_NAMES.get(group["algorithm"], group["algorithm"]),
                     counts, _cell(group, "detected_event_rate", True), protection,
                     _cell(group, "background_fpr", True), _cell(group, "recovery_fpr", True)])
    lines += _table(["数据集", "方法", "尝试/可行/可评危害/有效", "事件检出 %", "公共估计器保护 % (保护/有效)", "背景 FPR %", "恢复 FPR %"], rows)
    objectives = sorted({g["harm_objective"] for g in groups if g["harm_objective"]})
    if objectives:
        descriptions = {"instantaneous_error": "instantaneous_error：当轮绝对估计误差增加达到门槛",
                        "mechanism_specific": "mechanism_specific：旧版危害定义，额外要求攻击方向或时间响应满足对应机制"}
        lines += ["", "危害口径：" + "；".join(descriptions[key] for key in objectives) +
                  "。各组口径保存在 summary.csv/json；同组跨种子口径混合时整组不聚合。"]
    proxy_groups = [g for g in groups if g["dataset"] == "crowd_temperature"
                    and g["metrics"]["undefended_proxy_assessed_events"]["n"]]
    if proxy_groups:
        lines += ["", "### 3a. Crowd 无独立参考时的位移代理", "",
                  "这里只衡量相对未注入对照的输出位移，**不是对真实温度的误差，也不是保护率**。"
                  "位移事件使用与危害评估相同的幅度门槛和最少轮数；事件数跨种子求和。"
                  "平均绝对位移先在每个种子的攻击事件覆盖时刻计算（含退出阶段，同一任务/查询时刻去重），再报告种子间均值 ± 标准差及 n。"
                  "过滤后对照是在同一冻结模型下重放未注入报告，未注入报告不等于已认证真值。", "",
                  "各通道可评事件和可用估计时刻可能不同：已经观察到足够位移可判定达到门槛，"
                  "但输出缺失且未观察到足够位移的事件仍未知，不计作未达到。位移事件率只以各自可判定事件为分母；"
                  "位移均值只用该通道同时存在注入及未注入估计的时刻。不能据不同覆盖集合下较小的位移均值推断防护更好。", ""]
        rows = []
        for group in proxy_groups:
            totals = group["event_totals"]
            counts = []
            for prefix in ("undefended", "common"):
                counts.append("/".join(str(totals.get(prefix + "_proxy_" + suffix))
                                        if totals.get(prefix + "_proxy_" + suffix) is not None else "—"
                                        for suffix in ("shifted_events", "assessed_events")))
            rows.append([METHOD_NAMES.get(group["algorithm"], group["algorithm"]), *counts,
                         _cell(group, "common_proxy_shifted_assessed_event_rate", True, show_n=True),
                         _cell(group, "undefended_attack_mean_absolute_displacement", show_n=True),
                         _cell(group, "common_attack_mean_absolute_displacement", show_n=True)])
        lines += _table(["方法", "G0 位移事件/可评", "过滤后位移事件/可评", "过滤后位移事件率 %",
                         "G0 平均绝对位移 °C", "过滤后平均绝对位移 °C"], rows)
    lines += ["", "## 4. Condition alignment 原生感知与校准", "", "原生估计器包含质量与信誉加权，与表 2 的公共估计器不同，单独报告，不能混作相同估计器的排名。原生保护率仍以固定无防御估计器判定的有效事件为分母。", ""]
    rows = []
    for group in groups:
        if group["algorithm"] == "condition_alignment":
            protection = _cell(group, "native_protection_rate", True)
            totals = group["event_totals"]
            if totals["effective_events"]:
                protection += "；" + str(totals["native_protected_events"]) + "/" + str(totals["effective_events"])
            rows.append([DATASET_NAMES.get(group["dataset"], group["dataset"]), _cell(group, "native_mae"),
                         _cell(group, "native_coverage", True), protection,
                         str(group["alarm_capable_seeds"]) + "/" + str(group["calibration_known_seeds"])])
    lines += _table(["数据集", "原生 MAE ↓", "全测试轮覆盖 %", "原生保护 % (保护/有效)", "可告警种子/已知种子"], rows)
    lines += ["", "## 如何用于论文", ""]
    for dataset in summary["datasets"]:
        relevant = [g for g in groups if g["dataset"] == dataset]
        usable = [g for g in relevant if g["metrics"]["average_precision"]["n"] == len(summary["seeds"])]
        observations = []
        if len(usable) > 1:
            best = max(usable, key=lambda g: g["metrics"]["average_precision"]["mean"])
            observations.append("完整种子中 AP 均值最高为 " + METHOD_NAMES.get(best["algorithm"], best["algorithm"]) + "（" + _cell(best, "average_precision") + "）。")
        aligned = next((g for g in relevant if g["algorithm"] == "condition_alignment"), None)
        if aligned and aligned["metrics"]["common_mae_change"]["n"]:
            change = aligned["metrics"]["common_mae_change"]
            direction = "改善" if change["mean"] < 0 else "恶化" if change["mean"] > 0 else "持平"
            observations.append("Condition alignment 相对同种子的无过滤对照，公共估计器 MAE 平均" + direction +
                                "（过滤后−无过滤=" + _cell(aligned, "common_mae_change") + "），FPR（%）=" + _cell(aligned, "false_positive_rate", True) + "。")
        if observations:
            lines.append("- " + DATASET_NAMES.get(dataset, dataset) + "：" + " ".join(observations) + " 这些是描述统计，不代表统计显著性；MAE 变化还需结合覆盖率与输出轮次集合判断。")
    crowd = summary["contamination"].get("crowd_temperature", {})
    if crowd:
        details = []
        stages = (("train", "训练"), ("validation", "验证"), ("calibration", "校准"), ("test", "测试"))
        for stage, label in stages:
            counts = crowd.get(stage, {})
            if counts.get("known_seeds") and counts.get("zero_injection_seeds"):
                details.append(label + "阶段 " + str(counts["zero_injection_seeds"]) + "/" + str(counts["known_seeds"]) + " 个种子实际篡改报告数为 0")
        if details:
            lines.append("- Crowd 数据限制：" + "，".join(details) + "；这些阶段执行了攻击调度，但受稀疏报告与约束影响，不等同于保证每个阶段都能产生篡改。")
        elif all(crowd.get(stage, {}).get("known_seeds") == len(summary["seeds"]) for stage, _ in stages):
            ranges = [label + " " + str(crowd[stage]["min"]) + "–" + str(crowd[stage]["max"]) for stage, label in stages]
            lines.append("- Crowd 全阶段污染：全部 " + str(len(summary["seeds"])) +
                         " 个种子的训练、验证、校准、测试均实际产生了篡改报告；每种子报告数范围为" + "，".join(ranges) + "。")
    for warning in _concise_warnings(summary["warnings"]):
        lines.append("- " + warning)
    if not summary["warnings"]:
        lines.append("- 所有计划运行完成，各主指标有效种子数完整。")
    lines += ["", "统计单位是一次完整 seed 运行，不能把跨种子的所有报告当作独立重复，也不能把同一攻击在不同方法中的重复评估当作额外攻击样本。训练、验证、校准、测试均沿用配置中的污染协议；本报告不把训练数据解释为干净数据。", "",
              "直接使用本页的四张主表" + ("与 Crowd 位移代理附表" if proxy_groups else "") +
              "阅读结果；`summary.csv` 提供每数据集/方法一行的均值、标准差、n 和危害口径，`summary.json` 提供相同结构化汇总，`per_seed.csv` 仅用于复核单个种子和失败原因。无需阅读运行内部审计文件。", ""]
    return "\n".join(lines)


def analyze_benchmarks(root, expected_datasets=None, expected_algorithms=None, expected_seeds=None, output=None):
    """Write four compact artifacts and return a JSON-serializable summary.

    Explicit expected axes override ``run_plan.json``; absent a plan, axes are
    discovered from dataset/method/seed directories and the compact run status.
    A declared failed/running run is never counted as completed even if stale
    metrics happen to exist. Missing combinations remain visible in per_seed.
    """
    root = Path(root)
    output = Path(output) if output is not None else root / "analysis"
    plan = _read_json(root / "run_plan.json", {})
    status = _read_json(root / "run_status.json", {})
    statuses = {(str(r["dataset"]), str(r["algorithm"]), int(r["seed"])): r for r in status.get("runs", [])}
    discovered = set(statuses)
    for path in root.glob("*/*/*"):
        if path.is_dir():
            try:
                discovered.add((path.parent.parent.name, path.parent.name, int(path.name)))
            except ValueError:
                pass
    datasets = list(expected_datasets if expected_datasets is not None else plan.get("datasets", sorted({r[0] for r in discovered})))
    algorithms = list(expected_algorithms if expected_algorithms is not None else plan.get("algorithms", sorted({r[1] for r in discovered})))
    seeds = list(dict.fromkeys(int(s) for s in (expected_seeds if expected_seeds is not None else plan.get("seeds", sorted({r[2] for r in discovered})))))
    if not datasets or not algorithms or not seeds:
        raise ValueError("No benchmark plan found; provide expected datasets, algorithms and seeds")
    rows = []
    for dataset, algorithm, seed in itertools.product(datasets, algorithms, seeds):
        path = root / dataset / algorithm / str(seed)
        declared = statuses.get((dataset, algorithm, seed), {})
        run_status = declared.get("status")
        row = dict(dataset=dataset, algorithm=algorithm, seed=seed, status="missing", reason="metrics.json missing",
                   run_path=str(path), warnings="", notes="", common_estimator=None, input_sha256=None, harm_objective=None,
                   attack_mode=plan.get("attack_mode", "configured"))
        row.update({name: None for name in METRICS})
        if run_status and run_status not in ("completed", "complete", "success", "ok"):
            row.update(status=run_status, reason=str(declared.get("error") or declared.get("reason") or run_status))
        elif (path / "metrics.json").is_file():
            try:
                metrics = _read_json(path / "metrics.json")
                if not isinstance(metrics, dict) or not ("scored_reports" in metrics or "average_precision" in metrics):
                    raise ValueError("metrics.json does not contain evaluation metrics")
                mode = _attack_mode(path, plan.get("attack_mode", "configured"))
                values, common_method, warnings = _extract(path, metrics, mode)
                row.update(values)
                row.update(status="completed", reason="", common_estimator=common_method,
                           input_sha256=metrics.get("input_sha256"), harm_objective=_harm_objective(metrics),
                           warnings=";".join(warnings), attack_mode=mode,
                           notes="no_attacks_by_design;detection_metrics_not_applicable" if mode == "none" else "")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                row.update(status="invalid", reason=str(exc))
        rows.append(row)
    groups = []
    warnings = []
    for dataset, algorithm in itertools.product(datasets, algorithms):
        selected = [r for r in rows if r["dataset"] == dataset and r["algorithm"] == algorithm]
        completed = [r for r in selected if r["status"] == "completed"]
        objectives = sorted({r["harm_objective"] for r in completed if r["harm_objective"] is not None})
        attack_modes = sorted({r["attack_mode"] for r in completed})
        incompatible = len(objectives) > 1 or len(attack_modes) > 1
        incompatibility = "mixed_attack_modes" if len(attack_modes) > 1 else "mixed_harm_objectives" if incompatible else ""
        aggregatable = [] if incompatible else completed
        group = dict(dataset=dataset, algorithm=algorithm, expected_seeds=len(seeds), completed_seeds=len(completed),
                     attack_mode=attack_modes[0] if len(attack_modes) == 1 else plan.get("attack_mode", "configured") if not attack_modes else "mixed",
                     attack_modes=attack_modes,
                     harm_objective=objectives[0] if len(objectives) == 1 else None,
                     harm_objectives=objectives, aggregation_status="invalid" if incompatible else "ok",
                     aggregation_reason=incompatibility,
                     metrics={name: _statistics([r[name] for r in aggregatable]) for name in METRICS})
        event_metrics = ("attempt_events", "feasible_events", "assessed_events", "effective_events",
                         "common_protected_events", "native_protected_events") + tuple(
                             prefix + "_proxy_" + name for prefix in ("undefended", "common", "native")
                             for name in ("assessed_events", "shifted_events"))
        group["event_totals"] = {key: int(sum(r[key] for r in aggregatable if r[key] is not None))
                                 if any(r[key] is not None for r in aggregatable) else None for key in event_metrics}
        group["alarm_capable_seeds"] = sum(r["alarms_attainable"] == 1 for r in aggregatable)
        group["calibration_known_seeds"] = sum(r["alarms_attainable"] is not None for r in aggregatable)
        group["no_positive_seeds"] = sum(r["positive_reports"] == 0 for r in aggregatable)
        groups.append(group)
        label = DATASET_NAMES.get(dataset, dataset) + "/" + METHOD_NAMES.get(algorithm, algorithm)
        if incompatible:
            if len(attack_modes) > 1:
                warnings.append(label + "：跨种子混合无攻击与配置攻击模式，整组聚合无效，指标未平均。请分别分析两种实验。")
            else:
                warnings.append(label + "：跨种子混合危害口径（" + ", ".join(objectives) +
                                "），整组聚合无效，指标未平均。请分开旧运行与新协议后重新分析。")
            continue
        if len(completed) < len(seeds):
            warnings.append(label + "：仅完成 " + str(len(completed)) + "/" + str(len(seeds)) + " 个种子，未完成运行没有被删除或补零。")
        if group["no_positive_seeds"] and group["attack_mode"] != "none":
            warnings.append(label + "：" + str(group["no_positive_seeds"]) + " 个种子在可评分报告中无正例，AP/AUROC/F1/Recall 不用于检测能力结论。")
        cannot_alarm = sum(r["alarms_attainable"] == 0 for r in completed)
        if cannot_alarm:
            warnings.append(label + "：" + str(cannot_alarm) + " 个种子的校准池无法在指定 alpha 下告警，零 FPR 不能解释为成功防御。")
        if completed and not group["metrics"]["common_mae"]["n"]:
            warnings.append(label + "：缺少可用公共感知参考，不能宣称改善真实环境精度。")
        if completed and group["attack_mode"] != "none" and not group["metrics"]["common_protection_rate"]["n"]:
            if group["event_totals"]["assessed_events"] == 0:
                warnings.append(label + "：尚无可评估危害的事件，危害及保护率未知；有效事件计数 0 不代表已证实没有有害攻击。")
            else:
                warnings.append(label + "：保护率缺少可用分母或估计，不能将空值解读为零攻击或完全防护。")
        partial = [name for name in ("average_precision", "auroc", "f1", "recall", "common_mae", "common_protection_rate")
                   if 0 < group["metrics"][name]["n"] < len(seeds)]
        if partial:
            warnings.append(label + "：" + ", ".join(partial) + " 的有效种子不足 " + str(len(seeds)) + " 个，以表中 n 为准。")
    for dataset, seed in itertools.product(datasets, seeds):
        hashes = {r["input_sha256"] for r in rows if r["dataset"] == dataset and r["seed"] == seed and r["status"] == "completed" and r["input_sha256"]}
        if len(hashes) > 1:
            warnings.append(dataset + "/seed=" + str(seed) + "：方法间测试输入哈希不一致，不可视为公平配对比较。")
        objectives = {r["harm_objective"] for r in rows if r["dataset"] == dataset and r["seed"] == seed
                      and r["status"] == "completed" and r["harm_objective"] is not None}
        if len(objectives) > 1:
            warnings.append(dataset + "/seed=" + str(seed) + "：方法间危害口径不一致，不能直接比较攻击危害或保护率。")
    contamination = {}
    for dataset in datasets:
        contamination[dataset] = {}
        for stage in ("train", "validation", "calibration", "test"):
            counts = []
            for seed in seeds:
                available = {r[stage + "_injected_reports"] for r in rows if r["dataset"] == dataset and r["seed"] == seed
                             and r["status"] == "completed" and r[stage + "_injected_reports"] is not None}
                if len(available) == 1:
                    counts.append(next(iter(available)))
                elif len(available) > 1:
                    warnings.append(dataset + "/seed=" + str(seed) + "：" + stage + " 阶段的方法间篡改报告数不同，未合并为单个污染计数。")
            contamination[dataset][stage] = dict(known_seeds=len(counts), zero_injection_seeds=sum(n == 0 for n in counts),
                                                 min=int(min(counts)) if counts else None, max=int(max(counts)) if counts else None)
    completed = sum(r["status"] == "completed" for r in rows)
    failed = sum(r["status"] in ("failed", "error", "invalid", "unavailable") for r in rows)
    modes = {g["attack_mode"] for g in groups}
    summary = dict(schema_version=3, datasets=datasets, algorithms=algorithms, seeds=seeds,
                   attack_mode=next(iter(modes)) if len(modes) == 1 else "mixed",
                   completion=dict(expected=len(rows), completed=completed, failed=failed, pending=len(rows) - completed - failed),
                   invalid_aggregation_groups=sum(g["aggregation_status"] == "invalid" for g in groups),
                   statistics=dict(unit="seed", std="sample standard deviation (ddof=1)", missing="null; never imputed as zero", n="valid seed count per metric"),
                   groups=groups, contamination=contamination, warnings=warnings)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    flat = []
    for group in groups:
        row = {k: group[k] for k in ("dataset", "algorithm", "expected_seeds", "completed_seeds", "alarm_capable_seeds", "calibration_known_seeds", "no_positive_seeds",
                                    "attack_mode", "harm_objective", "aggregation_status", "aggregation_reason")}
        row["harm_objectives"] = json.dumps(group["harm_objectives"])
        for name, stats in group["metrics"].items():
            row.update({name + "_" + k: value for k, value in stats.items()})
        row.update({name + "_sum": value for name, value in group["event_totals"].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary
