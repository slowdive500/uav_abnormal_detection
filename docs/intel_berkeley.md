# Intel Berkeley：空间温度场与目标位置估计

这套实验重点观察**不同位置的合理温差、条件对齐，以及留出位置的温度估计**。我们保留真实传感器温度和实际平面坐标，将每个节点映射为一个固定任务位置上的虚拟 UAV 来源。

## 数据与默认选择

[MIT CSAIL 官方页面](https://db.csail.mit.edu/labdata/labdata.html)提供 Intel Berkeley 实验室 54 个节点的温度、湿度、光照、电压及位置。原始日志约每 31 秒记录一次，包含 2,313,682 行，压缩文件为 34,422,518 字节。温度单位为摄氏度，x/y 坐标单位为米。

本机完整日志、节点坐标和说明页保存在 `data/raw/intel_berkeley/`。默认配置 `configs/datasets/intel_berkeley.yaml` 选择 2004-03-01 至 03-09（左闭右开）的八天数据：

| 用途 | 节点 | 结果 |
|---|---|---|
| 报告来源 | 1、2、3、4、6、7、8、11 | 18,370 份五分钟温度报告 |
| 留出查询 | 9、10 | 4,605 份查询位置参考测量 |

节点 5 在所选时段缺少有效温度，因此默认来源采用节点 11。报告来源与原始 `raw_node_id` 一一对应；查询节点单独保存，训练和校准使用报告节点。

## 报告和条件

默认将整个选定区域视为一个温度场任务。每五分钟对各节点实际存在的观测求均值，以桶右端点作为提交时刻。`condition_x`、`condition_y` 是节点实际坐标，允许模型学习不同位置的系统温差。`sample_start/end/count` 记录本次平均对应的观测；缺失时段保留为空。

每个原始日期形成一个独立序列，默认按 4/1/1/2 天分成训练、验证、校准、测试。原数据的钟表时间没有官方时区定义，统一表使用 nominal UTC 编码来排序。`simulated_operation_height_m` 是任务设置中的固定高度，条件模型默认使用 x/y。

可通过 `node_ids`、`query_node_ids`、`start/end` 改变范围；`spatial_regions` 支持以圆心和半径创建局部任务。完整节点坐标使这些设置可直接复用。

## 查询位置怎样评测

CLI 将查询测量写入 `data/processed/intel_berkeley.queries.csv`。每条查询包含任务、时刻、位置、节点 ID 和 `reference_0`。模型先在查询位置估计温度，再与该节点的独立测量比较。

我们的方法把报告对齐到查询 x/y，按质量与旧信誉聚合；公共下游评测采用固定邻居的反距离加权。两类估计分别记录误差和有效覆盖。查询测量是独立传感器参考，原始天然报告的故障状态保留为未知，额外故障使用单独的注入标签。

## 下载和运行

```bash
python -m uav_abnormality.data.download \
  --manifest data/manifests/intel_berkeley.json \
  --destination data/raw/intel_berkeley --max-bytes 40000000
python scripts/run_report_benchmarks.py --datasets intel_berkeley \
  --algorithms condition_alignment prbtd tsadc \
  --prepare --epochs 2 --seeds 42 --output runs/intel_check
```

下载清单固定文件版本，`download_receipt.json` 保存校验信息。适配器分块读取 gzip，并按原日志的单空格分隔保留缺测空字段，保证温度、湿度和电压正确对应。

数据使用致谢遵循官方说明：Peter Bodik、Wei Hong、Carlos Guestrin、Sam Madden、Mark Paskin、Romain Thibaux，以及相关 TinyOS/Intel 团队。
