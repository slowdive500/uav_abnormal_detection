# TaxiBJ：动态交通场上的群智报告

这套实验重点观察**真实环境变化、个体报告污染和持续来源信誉**。原始交通流量构成时变环境场，虚拟 UAV 来源执行区域交通统计任务，提交带轻微测量误差的数值报告。

## 数据来源和格式

TaxiBJ 来自 Zhang、Zheng、Qi 的 AAAI 2017 工作 *Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction*。原始说明见 [数据文档](https://github.com/amirkhango/DeepST/blob/master/data/TaxiBJ/ReadMe.md)，本机文件来自固定版本的 [spatial-temporal-transformer 数据镜像](https://github.com/pengzhangzhi/spatial-temporal-transformer/tree/b17760e01eb08b9b4340e9dd6cbc7cd3caac2b97/data/TaxiBJ)，并逐文件匹配原始 MD5。

HDF5 的 `data[T,2,32,32]` 保存流入和流出两通道，`date` 使用 `YYYYMMDDss`，`ss` 为 01–48，分别代表北京时间一天内的半小时区间。

| 已下载文件 | 时间片 | 完整天数 | 保存范围 |
|---|---:|---:|---|
| BJ13 | 4,888 | 101 | 2013070101 – 2013102948 |
| BJ14 | 4,780 | 91 | 2014030101 – 2014062724 |
| BJ15 | 5,596 | 115 | 2015030101 – 2015063048 |

三个流量文件及气象、节假日、说明和校验文件共 259,812,395 字节，保存在 `data/raw/taxibj/`。默认采用 BJ15；适配器也接受相同格式的 BJ16 文件。下载清单为 `data/manifests/taxibj.json`，审计信息为原始目录中的 `DATA_AUDIT.json`。

## 从交通场生成报告

`configs/datasets/taxibj.yaml` 默认选择 BJ15 前八个完整天、四个相邻网格和八个稳定来源。每个来源在每个任务时隙以 0.8 概率参与，随后补足至少三个来源；一共生成 9,814 份报告、1,536 个任务时隙。

正常报告在交通场 (z) 上加入来源偏差和轻微噪声：

```text
x_i = max(0, z × (1 + bias_i) + sigma_i(z) × epsilon)
sigma_i(z) = (absolute_std + relative_std × z) × source_scale_i
```

偏差和噪声尺度按来源固定种子生成。默认 `relative_std=0.02`、`absolute_std=0.25`、`source_bias_std=0.005`。额外攻击由共享实验流程在时间划分之后追加，三个算法读取同样的报告。

报告时刻取半小时窗口右端点，转换为 UTC 秒，并保存原时间区间。网格和周期时间作为采集条件，原始交通场放入 `truth_*` 供估计误差评测。默认条件下的模型解释见 [数据协议](DATA_AND_PROTOCOL.md#1-报告与任务)。

## 与 PRBTD 原实验的关系

上游 PRBTD 的 [报告生成脚本](https://github.com/glab2019/PRBTD/blob/main/Code/CreateNoise/add_noise.py) 从 BJ16 流入数据选择 32 区域、120 时隙，为 100 用户各分配 30 次报告，并给恶意来源加入相对噪声。

本项目沿用“真实交通场 + 持续来源 + 动态参与”的任务形式，采用 BJ15 默认子集，为所有来源加入正常测量误差，再按 [目标导向攻击协议](ATTACK_PROTOCOL.md) 在训练、验证、校准和测试中尝试定向偏置、变化压制和虚假趋势。少数攻击者只修改报告值，`truth_*` 不向攻击生成器开放，留给评测器检查联合感知是否受到实质影响。已有运行仍按其保存的故障配置解释。

## 运行与扩大规模

```bash
python -m uav_abnormality.data.taxibj \
  --manifest data/manifests/taxibj.json --destination data/raw/taxibj
python scripts/run_report_benchmarks.py --datasets taxibj \
  --algorithms condition_alignment prbtd tsadc \
  --prepare --epochs 2 --seeds 42 --output runs/taxibj_check
```

修改数据配置可使用更多完整天数、网格或通道：

```yaml
files:
  - BJ13_M32x32_T30_InOut.h5
  - BJ14_M32x32_T30_InOut.h5
  - BJ15_M32x32_T30_InOut.h5
max_days: null
sequence_mode: campaign
channels: [inflow]       # 可增加 outflow
```

`cells: all` 使用 1,024 个网格。`sequence_mode: campaign` 保留跨日上下文；`day` 按日期分序列。下载的气象和节假日文件作为扩展数据保留，当前数值报告配置使用交通观测、网格和时间。
