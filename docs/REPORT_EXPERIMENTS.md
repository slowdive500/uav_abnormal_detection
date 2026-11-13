# 运行实验与读取论文结果

多种子实验完成后，直接打开输出目录下的 **`analysis/REPORT.md`**。它按该目录的实验计划汇总数据集和方法，不需要逐个读取单次运行文件。攻击前后对照集中在无攻击运行目录的 **`comparison/REPORT.md`**。方法与实验设置分别见 [METHOD.md](METHOD.md)、[DATA_AND_PROTOCOL.md](DATA_AND_PROTOCOL.md) 和 [ATTACK_PROTOCOL.md](ATTACK_PROTOCOL.md)。

## 无攻击实验与攻击前后对照

三个数据集、三个方法、五个种子的完整无注入攻击实验共 45 次，保留原训练上限 20 轮和早停规则：

```bash
python scripts/run_report_benchmarks.py \
  --no-attacks \
  --datasets crowd_temperature taxibj intel_berkeley \
  --algorithms condition_alignment prbtd tsadc \
  --seeds 0 42 46 47 1024 \
  --output runs/no_attacks_5seeds_20261113
```

`--no-attacks` 等价于 `--attack-mode none`，覆盖训练、验证、校准和测试全部阶段，分别重新训练模型、拟合标准化参数和校准阈值。原报告值、条件、参考和划分不变；原始自然故障不会被清洗。未知原标签保存在 `original_label`，误报率沿用此前“未注入背景视为正常”的实验口径，不把真实来源质量解释为已认证正常。四个阶段实际注入数为零，不会生成攻击事件。

运行仍保留原 Label-Trim 核验方式、预算与时间间隔。Crowd 仅取消“必须有可评分攻击”的要求，仍检查报告可评分和至少 40 个有效校准单元。恢复运行时添加 `--resume`，脚本拒绝将有攻击和无攻击计划混用。

无攻击结果没有注入正例，AP、AUROC、Recall、F1 无法评估。`analysis/REPORT.md` 只呈现误报率、评分覆盖、公共感知 MAE 和我们方法的原生感知结果；不输出空的攻击成功率表。

完成后生成配对比较：

```bash
python scripts/analyze_attack_comparison.py \
  --no-attack runs/no_attacks_5seeds_20261113 \
  --attack crowd_temperature=runs/crowd_sustained_5seeds_20261113 \
  --attack taxibj=runs/targeted_attacks_5seeds_20261113 \
  --attack intel_berkeley=runs/targeted_attacks_5seeds_20261113 \
  --output runs/no_attacks_5seeds_20261113/comparison
```

比较器先核对原始输入哈希、四份划分的身份与未注入数值，以及模型、窗口、训练和校准配置。Crowd 对照是最新两类持续攻击；TaxiBJ、Intel 对照是历史三类有限时长攻击，包含 `false_trend`。后两者的非攻击配置和输入与当前无攻击实验一致，可以比较 FPR、MAE；不能将其称作已完成当前两类攻击的新实验。

`comparison/REPORT.md` 的差值方向统一为“有攻击 − 无攻击”。FPR 在双方共同评分、且攻击侧未篡改的同一批报告上比较；MAE 在双方共同输出且参考一致的同一批任务/查询时刻上比较，并附全测试覆盖和交集覆盖。先在每个种子内作差，再计算差值的均值与样本标准差；不能用两个标准差相减。完整无攻击样本上的 FPR/MAE 另见 `analysis/REPORT.md`。

这是两组分别训练、校准、测试后的整流程差异，包含训练与校准污染影响。已有攻击运行中的 `baseline_predictions.csv` 则固定同一个含污染训练 checkpoint，只移除测试阶段的篡改；二者回答不同问题。Crowd 没有独立温度参考，不能提供真实环境 MAE 对照。

## 1. 只需看这些结果

| 文件 | 用途 |
|---|---|
| `analysis/REPORT.md` | 阅读跨种子结果、标准差和有效样本数；作为论文表格的起点 |
| `analysis/summary.csv` | 每个数据集、方法一行，导入 Excel 或绘图脚本 |
| `analysis/per_seed.csv` | 可选：查看某个种子的指标，分析波动或异常运行 |

`analysis/summary.json` 提供同一汇总的程序接口。`benchmark_status.json` 和每次运行的审计文件用于排查问题，不是常规结果阅读入口。

报告集中呈现以下主表，并对 Crowd 补充位移代理表：

1. **报告检测**：AP、AUROC、F1、Recall、FPR，以及评分覆盖率。
2. **公共环境估计**：三个方法使用同一估计器时的 MAE 与估计覆盖率，附无防御对照。
3. **攻击影响与保护**：尝试及有效事件数量、事件检出、公共估计保护率、普通背景与恢复窗口的 FPR。
4. **我们的方法的原生估计**：条件、质量与信誉加权估计的 MAE、覆盖率和保护率，以及校准规则可产生告警的种子数量。
5. **Crowd 位移代理**：相对于未注入对照的输出位移，以及达到预设位移门槛的事件数；该表不代表真实环境误差或已证实的保护成功。

跨种子统计为 **均值 ± 样本标准差**，标准差使用 `ddof=1`，每个种子权重相同，不把五个种子的报告混成一个大样本。每项指标都保留有效种子数 `n`；本次计划五个种子，表中未标注 n 的单元格使用全部五个种子，不足时才附注 `n/5`。只有一个有效值时可报告均值，标准差记为缺失；无定义或不可评估的值不补成零。`summary.csv` 中对应字段为 `<指标>_mean`、`<指标>_std` 和 `<指标>_n`。

结果解释保留三项必要限制：

- 检测指标的分母是能够评分的报告，须同时看评分覆盖率。无正例时，AP、AUROC、F1、Recall 不作为有意义的检测结论；校准规则无法产生告警时，零 FPR 也不能单独说明方法有效。
- TaxiBJ 用原交通场作为参考，Intel 用留出节点测量作为代理参考。Crowd 缺少独立参考，只能衡量相对于未篡改聚合的位移，不能宣称真实环境误差或保护成功。
- 公共估计表用于三个方法的同口径比较。我们的方法的原生加权估计单列；缺失估计不算零误差或成功保护。一次实际篡改也不必然达到危害门槛，攻击有效数量和覆盖分母会一并列出。

## 2. 本轮 Crowd v2 比较与历史结果

所有命令均在仓库根目录、`UAV_Abnormality_Detect` 环境中执行。本轮 Crowd v2 比较使用三个方法、五个种子，共 **15 次训练、校准和测试**：

```bash
conda activate UAV_Abnormality_Detect
cd /home/hhzhang/projects/UAV_abnormality_detection

python scripts/run_report_benchmarks.py \
  --datasets crowd_temperature \
  --algorithms condition_alignment prbtd tsadc \
  --prepare --seeds 0 42 46 47 1024 \
  --output runs/crowd_sustained_5seeds_20261113
```

不传 `--epochs` 时使用配置的完整训练预算：上限 20 轮、早停耐心值 5。脚本运行结束后自动生成 `analysis/` 中的汇总文件。同一数据集和种子下，三个方法共享划分与攻击报告；跨种子变化包含模型随机性和攻击轨迹。数据划分以及受控来源选择种子 `source_selection_seed: 6113` 固定。每次新实验使用新的输出目录，不能用当前配置覆盖历史结果。

意外中断后，在同一条运行命令末尾添加 `--resume`，即可按照原实验计划继续；已完成结果保留，失败尝试也留有记录。

Crowd v2 使用 **164 个训练前缀来源、2×2 网格、30 分钟桶、3,106 份报告**，新数据保存为 `data/processed/crowd_temperature_v2.csv`。训练 / 验证 / 校准 / 测试为 75 / 19 / 57 / 38 个时间片，对应可评分报告 1,042 / 186 / 731 / 630。半小时报告是实际区间观测的均值，不是瞬时温度真值。

四个阶段默认使用定向偏置和变化压制，均在受污染报告上运行；约束参数只从训练划分拟合。Crowd 使用持续模式，首轮预热后按真实到场的受控来源实施，允许暂时缺席，不补造参与。五个种子 `0、42、46、47、1024` 的预检测试正例分别为 **63、63、63、39、36** 份，四个划分均有实际且可评分的篡改。校准覆盖可用 51 个单元，最坏去除 2 个后为 49；运行严格检查实际保留至少 40。以上是可评估性检查，不是检测性能，最终指标以完整运行生成的汇总为准。

当前三套实验配置均只默认使用这两类攻击，并将危害目标改为当轮绝对估计误差增量；本轮新增持续模式仅用于 Crowd。TaxiBJ、Intel 仍保留原有有限时长攻击日历和训练预算，**本轮未按变更后的配置重跑它们**，因此目前不能声称已有新协议的完整 45 次结果。

| 结果目录 | 协议及用途 |
|---|---|
| `runs/crowd_sustained_5seeds_20261113` | 本轮 Crowd v2：三个方法 × 五个种子，持续两类攻击；位移代理与误差危害明确区分 |
| `runs/targeted_attacks_5seeds_20261113` | 历史 45 次比较：旧三类攻击、机制相关危害判据、至少两轮危害；Crowd 为旧 572 报告配置 |
| `runs/targeted_attacks_20261113` | 更早的单种子目标攻击结果 |
| `runs/method_comparison_20261113` | 更早的故障注入协议结果 |

历史 45 次中的 Crowd 五个种子均无进入评分的正例，其 AP、AUROC、F1、Recall 为不可评估，CA 校准也无法产生告警。这些旧结论保留，不能用 v2 的覆盖数据重新解释。不同数据版本、攻击日历、危害定义的结果不得混入同一跨种子均值。

已有实验可以单独重新生成汇总，无需重新训练：

```bash
python scripts/analyze_report_benchmarks.py \
  --input runs/crowd_sustained_5seeds_20261113
```

分析旧目录时可显式指定计划的实验组合，以便保留缺失种子的统计口径：

```bash
python scripts/analyze_report_benchmarks.py \
  --input runs/targeted_attacks_5seeds_20261113 \
  --datasets crowd_temperature taxibj intel_berkeley \
  --algorithms condition_alignment prbtd tsadc \
  --seeds 0 42 46 47 1024
```

重新生成汇总只读取各目录保存的结果，不会把旧攻击判据变成当前判据，也不会产生尚未运行的新结果。单次结果与跨种子统计不能互相替代。

## 3. 准备数据与单次检查

本机已经有准备好的报告表。仅当首次准备数据，或修改数据日期、来源、任务网格后，执行：

```bash
python scripts/run_report_benchmarks.py \
  --datasets crowd_temperature taxibj intel_berkeley \
  --prepare-only --output runs/data_preparation
```

报告表保存在 `data/processed/`；Crowd 当前写入 `crowd_temperature_v2.csv`，保留旧 `crowd_temperature.csv`；Intel 同时生成查询节点参考表 `intel_berkeley.queries.csv`。新机器可添加 `--download` 下载固定版本原始文件；Crowd 原版 CSV 的手动导入见 [数据页](crowd_temperature.md)。`--skip-unavailable` 允许跳过尚未准备好的数据集。

仅检查完整流程时，可缩短训练；这些结果不作为正式比较：

```bash
python scripts/run_report_benchmarks.py \
  --datasets intel_berkeley --algorithms condition_alignment prbtd tsadc \
  --epochs 2 --seeds 42 --output runs/intel_targeted_attack_check
```

单次完整训练、校准、测试也可直接调用：

```bash
python -m uav_abnormality run \
  --config configs/experiments/intel_berkeley.yaml \
  --algorithm condition_alignment --output runs/intel_ours
```

`configs/experiments/<数据集>.yaml` 控制划分、窗口、训练、模型、校准、四阶段攻击及危害评估门槛。当前 `attack_harm_objective: instantaneous_error` 比较同轮攻击后与未注入对照的绝对参考误差，至少 **一个**任务轮次的增量达到门槛即可。Crowd/Intel 门槛为 0.2 °C，TaxiBJ 为 1 个流量单位；这是实验效应定义，不是业务报警线。Crowd 没有独立参考，沿用这些门槛计算位移代理，真实误差与保护率仍不可评估。旧 checkpoint 缺少新字段时保留原机制判据；详细规则见 [攻击协议](ATTACK_PROTOCOL.md)。

## 4. 重新评测和问题追溯

需要重新评测已保存的模型时：

```bash
python -m uav_abnormality evaluate \
  --checkpoint runs/intel_ours/checkpoint.pt \
  --reports runs/intel_ours/splits/test.csv --output runs/intel_re_evaluation

python -m uav_abnormality infer \
  --checkpoint runs/intel_ours/checkpoint.pt \
  --reports data/processed/intel_berkeley.csv --output runs/intel_inference
```

`evaluate` 计算标签指标，`infer` 输出评分与判定。两者沿用 checkpoint 保存的模型和校准规则，并按时间推进状态；恢复信誉时需从独立序列的起点重放。

只有需要追溯某个异常指标时，才进入 `<数据集>/<方法>/<种子>/`：`metrics.json` 和 `predictions.csv` 定位检测结果，`attack_events.csv` 与 `attack_rounds.csv` 定位攻击影响，`attack_audits/<阶段>/` 核查四阶段注入。配置、划分、训练期约束和 checkpoint 也随运行保存，支持复现。

`baseline_predictions.csv` 是同一个含污染训练模型在移除测试篡改后的重新评分；我们的方法还保存 `baseline_native_environment.csv`。这些反事实重新推进信誉状态，不重新训练干净模型，也不消除原数据中的未知自然故障。重新评测应保留 `baseline_*` 和攻击事件字段，才能重建这组对照。

日常验证使用 `python scripts/doctor.py` 和 `pytest -q`。已执行的实验规模及限制记录在 [VALIDATION.md](VALIDATION.md)。
