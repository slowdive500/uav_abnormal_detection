# 我们的方法：跨来源条件对齐与动态信誉

算法名为 `condition_alignment`，对应[方法原稿](PROPOSAL.md)中的五个步骤：**条件对齐 → 留一参照 → 质量与异常判定 → 目标条件下的环境估计 → 信誉更新**。同一任务中，来源位置不同可能带来合理的观测差异；方法先学习这种差异，再判断某份报告偏离其他独立报告的程度。

## 1. 输入、输出与模型

输入是一组同期可比较报告：来源 $i$ 提交数值 $x_{i,k}$ 和采集条件 $c_{i,k}$。三个实验当前都使用一个标量观测，例如温度或区域流量。条件由数据配置明确选择：crowd 使用经纬度，Intel 使用 x/y，TaxiBJ 使用网格位置及周期时间。

条件模型 $h_\theta(c)$ 用线性层或轻量 MLP 表示采集条件带来的系统差异。将来源 $j$ 的报告换算到来源 $i$ 的条件：

$$
\Phi_\theta(x_j,c_j,c_i)=x_j+h_\theta(c_i)-h_\theta(c_j).
$$

因此，同条件转换自动保持原值。共同环境变化由当时收到的报告提供，模型学习的是条件之间的修正关系。报告和条件各自使用训练集拟合的标准化参数，并随 checkpoint 保存。

每份报告的主要输出是偏差 `error`、连续质量 `quality`、参照离散度 `dispersion`、检验分数 `score`、`p_value`、异常判定 `prediction`，以及信誉更新前后的 `reputation_prior` 和 `reputation`。

`aligned_reference_0` 记录当前时刻换算回原始单位的留一参照，可直接与 `value_0` 对照；报告级 `error` 则综合整个历史窗口的共同有效位置。

## 2. 用其他来源构造参照并训练

为评估来源 $i$，根据任务、采集条件和有效观测选取其他来源 $\mathcal N_i$。来源自身始终留出，默认至少需要两个其他来源。参照选择规则和模型在训练、校准、测试时保持一致。

运行时沿用实验配置的历史窗口。参照按末端条件和有效观测选取，条件转换应用于窗口中各时间位置，报告级距离取共同有效位置上的标准化平均绝对偏差。末端需有待检报告及至少两个有效参照来源。

对每个参照完成条件转换，再取中位数：

$$
\widehat x_{i\leftarrow j}=\Phi_\theta(x_j,c_j,c_i),\qquad
\widetilde x_i=\operatorname{median}_{j\in\mathcal N_i}\widehat x_{i\leftarrow j}.
$$

训练使用报告与参照之间的 Huber 损失：

$$
\mathcal L(\theta)=\sum_{k,i}\rho_\kappa\bigl(\operatorname{dist}(x_{i,k},\widetilde x_{i,k})\bigr).
$$

当前三套标量实验使用中位数和绝对距离；多特征输入使用几何中位数和按特征数归一的欧氏距离，完整坐标共同构成一次有效观测。

中位数降低参照中极端报告的影响，Huber 限制大目标残差对梯度的影响。训练信号来自跨来源报告的一致性，异常标签和环境参考保留给校准核验及评测。验证损失用于选择 checkpoint。

## 3. 质量和异常证据分开计算

固定训练尺度下的偏差决定质量：

$$
e_i=\operatorname{dist}(x_i,\widetilde x_i),\qquad q_i=\frac{1}{1+e_i}.
$$

同样大小的偏差始终得到同样的质量，便于跨任务积累信誉。异常分数再考虑其他来源之间的分歧：

$$
\sigma_i=\operatorname{median}_{j\in\mathcal N_i}
\operatorname{dist}(\widehat x_{i\leftarrow j},\widetilde x_i),\qquad
s_i=\frac{e_i}{\max(\sigma_i,\tau_0)}.
$$

参照一致时，偏离更有异常证据；参照分歧大时，异常证据相应减弱。`tau0` 是固定下限，防止参照完全一致时除零。质量 $q$ 用于估计和信誉，分数 $s$ 用于异常判定。

举例：某目标报告为 30，两份其他报告经条件对齐后都是 24，训练确定的距离尺度为 2。则 $e=3$、$q=0.25$、$\sigma=0$；若 `tau0=0.1`，则 $s=30$。是否告警由下一步的校准分数分布决定。

## 4. Label-Trim 校准

模型训练结束后冻结参数。校准任务按预设间隔和规则抽取，每个任务选一份待检报告及其参照，并计算分数。

1. 在 $n$ 个校准分数中选出最高的 $m$ 个，预算满足 $m\le\alpha(n+1)$。
2. 核验这些报告，移除确认异常者，保留核验正常者及其余报告。
3. 保存保留集合 $\mathcal D_{\mathrm{cal}}^{\mathrm{LT}}$，测试时计算包含并列分数的精确秩 p 值：

$$
p_i=\frac{1+\sum_{z\in\mathcal D_{\mathrm{cal}}^{\mathrm{LT}}}\mathbf 1[s_z\ge s_i]}
{|\mathcal D_{\mathrm{cal}}^{\mathrm{LT}}|+1},\qquad
A_i=\mathbf 1[p_i\le\alpha].
$$

当前可复现实验使用 `verification: injected_fault_oracle`，由选中报告的注入记录回答核验结果。该入口模拟有预算的核验；已选之外的标签不参与修剪。`calibration_audit.csv` 记录选择、核验与移除过程，`calibration.json` 保存最终分数、预算和规则。时序任务及小规模校准的解释统一见 [数据协议第 4 节](DATA_AND_PROTOCOL.md#4-校准怎样工作)。

接入人工或外部核验时，将 `verification` 改为 `external`，设置 `external_verification_csv`。该 CSV 用 `sequence_id, task_id, timestamp, source_id` 对应报告，并提供 `verified_label`：0 为正常，1 为异常，-1 为尚无核验结论。系统仍按相同预算读取选中报告的核验。

Label-Trim 依据 [Bashari、Sesia、Romano，ICML 2025](https://proceedings.mlr.press/v267/bashari25a.html) 的第 3.3 节。这里按照 proposal 使用其核验与秩检验流程。

## 5. 估计目标条件下的环境

给定目标条件 $c^*$，把通过评分且可与目标比较的报告转换到目标条件。权重同时使用本次判定、上一轮信誉和本次质量：

$$
w_i=(1-A_i)R_i^{\mathrm{old}}q_i,\qquad
\widehat z^*=\operatorname{weighted\ median}_i
\bigl(\Phi_\theta(x_i,c_i,c^*);w_i\bigr).
$$

告警贡献权重为零；历史可靠且当前质量高的贡献获得更大权重。默认至少需要三个有效来源，来源不足时保存该次估计缺失状态。

Intel 的目标条件来自留出查询节点的位置，查询温度在估计完成后用于计算误差。TaxiBJ 使用对应网格条件及原流量参考。结果写入 `native_environment.csv`；统一的公共后处理仍保存在 `environment_estimates.csv` 或 `query_estimates.csv`，两者便于分别阅读方法自身估计和单纯过滤效用。

## 6. 环境估计之后更新信誉

每个来源初始信誉为 0.5。质量高于旧信誉时慢速上升，低于旧信誉时快速下降：

$$
R_i^{\mathrm{new}}=R_i^{\mathrm{old}}+
\begin{cases}
\eta_\uparrow(q_i-R_i^{\mathrm{old}}),&q_i\ge R_i^{\mathrm{old}},\\
\eta_\downarrow(q_i-R_i^{\mathrm{old}}),&q_i<R_i^{\mathrm{old}}.
\end{cases}
$$

默认 `eta_up=0.05`、`eta_down=0.25`。上述例子中，旧信誉为 0.5、质量为 0.25，更新后信誉为 0.4375。异常判定影响当前聚合，连续质量决定信誉更新。

每个有效任务更新一次，按 `(sequence_id, timestamp, task_id)` 排序。同期多个任务按固定任务 ID 顺序处理，当前任务先使用旧信誉完成估计，再更新已评估来源；缺席或参照不足的来源保留原信誉。独立序列和数据划分分别从初始信誉开始。

## 7. 配置与代码对应

三套 `configs/experiments/*.yaml` 已配置条件字段。主要参数如下：

```yaml
models:
  condition_alignment:
    architecture: mlp       # linear 或 mlp，选择 h(c) 的参数化
    hidden_dim: 32
    min_references: 2
    tau0: 0.1
    huber_delta: 1.0
method:
  condition_columns: [condition_x, condition_y]  # Intel 示例
  reputation:
    initial: 0.5
    eta_up: 0.05
    eta_down: 0.25
  environment:
    min_sources: 3
    max_neighbors: 8
    radius: null
  calibration:
    alpha: 0.05
    budget: auto
    verification: injected_fault_oracle
    selection_seed: 42
```

校准的 `separation_seconds` 采用窗口长度乘数据时间间隔；Intel 默认为 3,600 秒。模型可配置 `max_neighbors` 和 `neighbor_radius` 选择参照，环境估计对应 `max_neighbors` 和 `radius`；距离使用标准化条件空间。改变条件字段时，需要重新训练和校准。

以下路径均相对于 `uav_abnormality/`：

| 文件与入口 | 对应的实现 |
|---|---|
| `models/condition_alignment.py`：`ConditionAlignment` | $h(c)$、条件转换、留一中位数、Huber、$e/q/\sigma/s$ |
| `condition_data.py`：`ConditionScaler`、`ConditionWindows` | 条件字段、训练集条件标准化、有效观测组织 |
| `calibration.py`：`fit_label_trim`、`label_trim_pvalues` | 核验选择、Label-Trim、精确 p 值和审计 |
| `condition_runtime.py`：`apply_reputation_and_estimate` | 目标条件估计、加权中位数、逐任务信誉更新 |
| `condition_experiment.py`：`train_condition_experiment` | 训练、校准、保存加载与评测衔接 |

直接运行：

```bash
python -m uav_abnormality run \
  --config configs/experiments/intel_berkeley.yaml \
  --algorithm condition_alignment --output runs/intel_condition_alignment
```

阅读一次结果时，先检查 `predictions.csv` 中 `error → quality` 与 `dispersion → score → p_value` 的计算链，再将 `native_environment.csv` 和 `reputation.csv` 按任务时间对应起来，就能看到一份报告如何影响环境估计与之后的信誉。
