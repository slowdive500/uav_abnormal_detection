# UAV Abnormality Detection

这个项目研究无人机群智感知中的**报告质量评价、异常检测、环境估计和来源信誉维护**。平台收到各来源的数值报告及采集条件后，判断哪些贡献可靠，并用可靠贡献估计目标位置的环境状态。

项目使用 PyTorch，当前比较三种算法：

| 算法 | 在项目中的作用 | 主要机制 |
|---|---|---|
| `condition_alignment` | 我们的方法 | 跨来源条件对齐、留一鲁棒参照、连续质量、Label-Trim 校准、质量与信誉加权估计 |
| `prbtd` | 群智感知基线 | CFSTTN 预测、报告关联、迭代质量与信誉 |
| `tsadc` | 时序异常检测基线 | 扩散去污染、时空依赖建模、双分支重建误差 |

建议先读本页，运行一次实验，再看 [我们的方法](docs/METHOD.md)。调整数据和实验设置时，查阅 [数据与评测协议](docs/DATA_AND_PROTOCOL.md) 和 [运行指南](docs/REPORT_EXPERIMENTS.md)。

## 一次实验做什么

```text
原始数据 → 来源/任务/时间/数值/条件组成的报告表
        → 划分训练、验证、校准、测试 → 四阶段均按统一约束尝试报告攻击
        → 训练模型并用验证集早停 → 冻结模型并校准判定规则
        → 按时间评分 → 环境估计与信誉更新 → 汇总检测和估计指标
```

三个算法使用相同的报告、划分和攻击样本。当前默认采用**定向偏置、变化压制**两类少数来源篡改，四个阶段均允许污染。攻击意图与实际危害分开：有独立参考时，危害按当前轮次估计的绝对误差是否增加判定，连续上升的轨迹本身不算危害。完整定义与不可行尝试的处理见 [攻击协议](docs/ATTACK_PROTOCOL.md)。我们的方法额外读取采集条件，按照 proposal 完成从条件对齐到信誉更新的完整流程。当前比较包含这三种完整算法；线性模型和 MLP 是同一个条件模型的配置选项。

## 开始运行

环境已准备在 `UAV_Abnormality_Detect`，原始数据已放入 `data/raw/`。在项目根目录执行：

```bash
conda activate UAV_Abnormality_Detect
cd /home/hhzhang/projects/UAV_abnormality_detection

# 先用两轮训练检查三套数据、三个算法的完整流程。
python scripts/run_report_benchmarks.py \
  --datasets crowd_temperature taxibj intel_berkeley \
  --algorithms condition_alignment prbtd tsadc \
  --prepare --epochs 2 --seeds 42 --output runs/three_methods_check

# Crowd v2 有攻击比较：三个方法 × 五个种子，共15次完整运行。
python scripts/run_report_benchmarks.py \
  --datasets crowd_temperature \
  --algorithms condition_alignment prbtd tsadc \
  --prepare --seeds 0 42 46 47 1024 \
  --output runs/crowd_sustained_5seeds_20261113

# 全部三个数据集：四阶段均无注入攻击，共45次完整运行。
python scripts/run_report_benchmarks.py \
  --no-attacks --seeds 0 42 46 47 1024 \
  --output runs/no_attacks_5seeds_20261113
```

每次新实验使用新的输出目录，已有结果会保留；上述目录已有本轮任务时应续跑而非覆盖，中断续跑见 [运行指南](docs/REPORT_EXPERIMENTS.md)。默认训练上限为 20 轮、早停耐心值为 5；脚本完成后自动生成跨种子分析与论文表格。运行使用下表中的配置数据范围，不自动扩展为全部原始记录。`device: auto` 自动选择可用设备，当前机器使用 CPU。新机器可运行 `bash scripts/setup_env.sh` 建立环境，再按运行指南下载、导入和准备数据。

历史目录 `runs/targeted_attacks_5seeds_20261113` 保存旧三类攻击协议的 45 次运行，Crowd 的新有攻击结果另存在 `runs/crowd_sustained_5seeds_20261113`。无攻击实验重新训练、校准和测试全部 45 个组合，原始观测与自然故障保留。配对比较复用非攻击配置及原始数据完全相同的上述攻击对照，并明确 TaxiBJ、Intel 对照包含历史 `false_trend`，不将其称为当前两类攻击的新结果。

只运行我们的方法：

```bash
python -m uav_abnormality run \
  --config configs/experiments/intel_berkeley.yaml \
  --algorithm condition_alignment --output runs/intel_condition_alignment
```

## 三套实验分别观察什么

| 数据 | 默认实验规模 | 主要观察点 |
|---|---|---|
| [crowd_temperature v2](docs/crowd_temperature.md) | 164 个训练前缀来源，2×2 网格，189 个有报告的半小时时隙，3,106 份报告 | 移动来源的位置差异、动态参与及持续报告攻击 |
| [TaxiBJ](docs/taxibj.md) | BJ15 八天、4 网格、8 来源，9,814 份报告 | 真实交通变化下的报告污染、持续质量和信誉 |
| [Intel Berkeley](docs/intel_berkeley.md) | 八天、8 个报告节点、2 个查询节点，18,370 份报告 | 空间条件差异，以及目标位置的温度估计 |

三套实验把原数据组织为虚拟无人机执行的数值报告任务。完整原始文件与默认子集分别保存；数据含义、条件字段、标签和校准规模集中说明在 [数据与评测协议](docs/DATA_AND_PROTOCOL.md)。另保留 [LAPSE-RATE 与 UAV3D 接口](docs/datasets.md)，用于后续接入飞行观测或视觉报告。

## 结果从哪里看

直接打开输出目录中的 **`analysis/REPORT.md`**，即可阅读检测、公共环境估计、攻击影响与保护，以及我们的方法的原生估计表；Crowd 另附位移代理表。论文结果不需要逐个查看运行文件。

无攻击运行的汇总只保留误报率、覆盖率与感知误差；无正例时不解释 AP、Recall 或 F1。攻击前后对照直接查看 `runs/no_attacks_5seeds_20261113/comparison/REPORT.md`，生成命令见 [运行指南](docs/REPORT_EXPERIMENTS.md)。差值先在同种子、相同报告或输出集合内计算，再汇总均值与样本标准差。

| 文件 | 用途 |
|---|---|
| `analysis/REPORT.md` | 跨种子论文表：均值、标准差、有效种子数与主要限制 |
| `analysis/summary.csv` | 每个数据集、方法一行，供 Excel 和绘图使用 |
| `analysis/per_seed.csv` | 可选：逐种子指标，用于检查波动 |

统计采用各种子指标的均值与样本标准差（`ddof=1`），同时报告有效 `n/5`。无定义值不补零；只有一个有效种子时不报告标准差。`analysis/summary.json` 提供程序接口，`benchmark_status.json` 和底层审计文件仅用于排查。

已有运行可单独更新汇总，无需重新训练：

```bash
python scripts/analyze_report_benchmarks.py \
  --input runs/crowd_sustained_5seeds_20261113
```

Crowd 缺少独立环境参考，其位移不能解释为真实环境误差。v2 按有效校准轮数规划了 30% 的校准时间片，覆盖允许抽取 51 个单元，最多去除 2 个；运行仍须核查实际保留数至少 40，以及四个阶段都有可评分篡改。半小时报告是区间观测均值，不是瞬时温度真值。详细指标口径见 [运行指南](docs/REPORT_EXPERIMENTS.md)，实际运行与限制见 [验证记录](docs/VALIDATION.md)。

## 代码从哪里读

| 路径 | 内容 |
|---|---|
| `uav_abnormality/models/` | 三种算法；我们的条件模型和基线网络 |
| `uav_abnormality/data/` | 原始数据下载、任务转换和报告格式 |
| `uav_abnormality/protocol.py` | 时间划分、标准化和窗口 |
| `uav_abnormality/attacks.py`、`attack_pipeline.py` | 两类默认目标攻击、持续与有限时长日历、共享训练期约束与四阶段注入；保留旧趋势类型兼容 |
| `uav_abnormality/corruption.py` | 注入入口及旧六类故障兼容实现 |
| `uav_abnormality/experiment.py` | 训练、校准、顺序评分和保存加载 |
| `uav_abnormality/evaluation.py` | 报告参考与查询位置的估计评测 |
| `uav_abnormality/attack_evaluation.py` | 全部攻击尝试、危害与防御效果的反事实审计 |
| `configs/datasets/`、`configs/experiments/` | 数据任务定义和训练参数 |
| `scripts/run_report_benchmarks.py` | 多数据集、多算法、多种子实验入口 |
| `scripts/analyze_report_benchmarks.py` | 汇总逐种子指标、均值与样本标准差，生成论文表格 |
| `tests/`、`third_party/` | 回归测试、基线源代码与版本来源 |

基线结构及适配细节分别见 [PRBTD](docs/prbtd.md)、[TSAD-C](docs/tsadc.md)；代码与数据来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
