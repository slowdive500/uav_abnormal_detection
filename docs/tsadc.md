# TSAD-C：扩散去污染与时空依赖重建

TSAD-C 是本项目的时序异常检测基线，输出每个来源的异常分数。它首先学习去污染重建，再用时空依赖模型重建报告，两种重建误差共同决定分数。

实现依据 [UAI 2025 论文](https://proceedings.mlr.press/v286/ho25a.html)、[完整方法](https://arxiv.org/html/2308.12563v5) 和 [作者代码](https://github.com/Armanfard-Lab/TSADC/tree/7013d7d2053044acdd3aa980bcf309d48f94de0d)，固定版本 `7013d7d2053044acdd3aa980bcf309d48f94de0d`。代码位于 `models/tsadc.py` 与 `models/_tsadc_s4.py`。

## 模型怎样工作

1. **S4 条件扩散去污染器**：训练时遮挡部分有效观测，学习噪声预测和恢复；推理时完成 DDPM 反向采样。
2. **时空依赖模型**：S4 编码时间关系，注意力与余弦近邻建立来源图，再用 GINE 消息传递和重建器估计报告。
3. **双分支评分**：对扩散恢复与依赖重建分别计算 RMSE，并按原方法权重组合。

训练目标包含噪声预测、图平滑/连通/稀疏正则及重建误差。S4 使用 HiPPO 初始化和 NPLR 核，图层采用 dense PyTorch 实现，便于在当前少量来源任务上运行。

## 报告接口和适配

输入为 `[B, W, S, F]` 的标准化报告及同形观测 mask。一个图节点对应一个来源，`F` 是其报告特征数。损失只计算实际观测位置，完全缺席的来源从图边和图正则中排除。

原方法聚合整个窗口的分数，本项目按来源评估窗口最后时刻，返回 `[B, S]`。扩散条件隐藏窗口末端，使各来源都有重建误差；依赖重建分支使用当前已经收到的窗口。三种训练遮挡策略为 RandM (`rm`)、RandBM (`rbm`) 和 BoM (`bom`)，默认 RandBM。

推理使用完整的 T 到 1 反向采样，将多次采样误差平均。固定独立随机数使同一个窗口的分数与批大小、处理顺序无关。独立校准集确定固定阈值，输出的异常分数越高，报告越反常。

## 配置和计算成本

默认参数为 `hidden_dim=16`、`state_dim=8`、`diffusion_steps=10`、`score_samples=2`，适合当前 CPU 实验。增加扩散步数、采样数、S4 状态和残差块会提高计算量，按验证集结果选择配置。

移植时保留作者的 Laplacian 正则及训练噪声恢复方式，并修正设备硬编码、模块注册和批次形状问题。S4 参数的 `_optim` 元数据由训练器保留。实现依赖 PyTorch、NumPy、SciPy、einops 和 opt_einsum。

## 运行与结果

```bash
python -m uav_abnormality run \
  --config configs/experiments/intel_berkeley.yaml \
  --algorithm tsadc --output runs/intel_tsadc
```

`predictions.csv` 保存来源异常分数和校准判定，`metrics.json` 保存报告检测及公共下游估计结果。TSAD-C 的输出重点是异常评分，信誉维护由我们的方法和 PRBTD 各自实现。

测试覆盖 S4、GINE 消息公式、掩码、梯度、固定采样及保存加载。移植代码遵循上游 **CC BY-NC-SA 3.0**，见 `third_party/TSADC/LICENSE.txt` 和 `PROVENANCE.md`。
