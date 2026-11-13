# PRBTD：预测、报告关联与信誉迭代

PRBTD 是本项目的群智感知基线，输出逐报告质量、异常分数和来源信誉。模型由 CFSTTN 预测网络与基于报告关系的质量/信誉迭代组成。代码入口为 `models/prbtd.py` 和 `models/prbtd_cfsttn.py`。

依据为 [论文](https://arxiv.org/abs/2405.18725) 和 [作者仓库](https://github.com/glab2019/PRBTD/tree/078bf86f09fdb17dafe90d305986559f22efd4a8)，固定版本 `078bf86f09fdb17dafe90d305986559f22efd4a8`；原始文件及校验值在 `third_party/PRBTD/PROVENANCE.json`。

## 一次评分的过程

1. CFSTTN 使用窗口前 `W-1` 个时刻，预测各来源在最后时刻的数值。
2. 将预测残差分成个体与环境两部分，以二者构造报告特征。
3. 对不同来源的报告计算余弦相似度，形成正、负关联关系。
4. 交替更新来源信誉和报告质量，最后以 `1-quality` 作为异常分数。

训练使用最后时刻有效报告上的 MAE。关联缓冲使用最近五个已评分时刻，环境残差缓冲使用最近一百个；同来源的报告从相互支持关系中排除。默认最多迭代 30 次，信誉收敛阈值为 `1e-4`，沿用上游 `alpha=0.018, beta=0.06, gamma=0.5, rho=0.002`。

## 统一报告接口中的适配

| 部分 | 本项目设置 |
|---|---|
| 输入布局 | 每个任务形成 `[时间, 来源, 特征]` 面板，来源按固定顺序排成 1×S 索引网格 |
| 预测目标 | CFSTTN 对来源位置产生下一步预测，多个标准化特征共用网络 |
| 残差 | 默认使用标准化数值的有符号差；`residual_mode: relative` 支持有分母下限的相对残差 |
| 输出 | 标准化值可为负，因此去掉最后 sigmoid |
| 缺测 | 输入补标准化零，损失与报告关系使用有效掩码 |
| 多特征 | 一来源的一组特征仍作为一份报告，在共同有效特征上计算关系 |
| 容量 | 默认 `d_model=16, d_inner=32, growth_rate=4`，可在验证阶段调整 |

这些设置将原始区域预测任务转换为来源报告面板。来源索引卷积采用固定顺序，真实位置仍保存在数据条件字段中，供我们的方法进行条件对齐。

## 来源信誉怎样延续

三套数值实验使用 `scoring.reputation_scope: source`：同一独立序列内，来源信誉跨任务延续；各任务保留自己的残差和关联缓冲。同一时刻多个任务读取相同旧信誉，完成后平均该来源的更新提议，缺席来源保持原值。独立序列和数据划分分别重置。

旧配置默认 `task` 范围，信誉随每个任务序列独立维护。`stream_score` 用于顺序推理，模型权重保存后可通过重放序列恢复运行状态。`score` 是无状态辅助接口，主要用于隔离测试。

## 运行与结果

```bash
python -m uav_abnormality run \
  --config configs/experiments/intel_berkeley.yaml \
  --algorithm prbtd --output runs/intel_prbtd
```

`predictions.csv` 中 `quality_native` 是原生报告质量，`reputation_prior/reputation` 是更新前后信誉。独立校准集确定固定阈值，后续报告用同一规则判定。

`tests/test_prbtd.py` 覆盖预测因果性、掩码、顺序评分、保存加载，并对固定上游版本核对 CFSTTN 运算和常规单变量的质量/信誉公式。上游快照保留其原许可状态，详情见 [第三方来源](../THIRD_PARTY_NOTICES.md)。
