# 扩展数据接口：LAPSE-RATE 与 UAV3D

当前实验主线为 [crowd_temperature](crowd_temperature.md)、[TaxiBJ](taxibj.md) 和 [Intel Berkeley](intel_berkeley.md)。本页介绍保留的两个扩展入口：实测无人机风场和仿真多机视觉报告。统一报告格式与划分约定见 [DATA_AND_PROTOCOL.md](DATA_AND_PROTOCOL.md)。

## LAPSE-RATE：实测无人机风场

来源为 [University of Kentucky 数据论文](https://essd.copernicus.org/articles/12/1759/2020/) 和 [固定 Zenodo 版本 3923056](https://zenodo.org/records/3923056)。完整清单含 180 个文件、约 5.48 GB，路径为 `configs/datasets/lapse_rate_manifest.json`。本机已下载 14 个文件、约 58 MB，供真实 NetCDF 读取和任务覆盖检查。

### 报告怎样生成

| 内容 | 原始字段与转换 |
|---|---|
| 来源 | 文件名中的物理平台，同平台的不同航次保留同一 ID |
| 观测 | 默认水平风分量 `u/v`，单位 m/s |
| 时间 | UTC 日期和时刻，保留小数秒 |
| 任务位置 | `Lat/Long/Alt` 形成统一全局三维网格，`Alt` 为海拔 |
| 条件 | `z` 为离地高度，另保留位置、地速和姿态 |
| 缺测 | `-9999.9` 转为有效掩码中的缺测 |

默认按 5 秒、250 米水平和 50 米垂直网格组织任务，同来源同任务时段的观测取平均，同一 UTC 日期作为序列。窗口和来源覆盖由准备阶段输出统计。

当前下载子集产生 1,120 份风矢量报告，其中 13 个任务时隙有两个来源，其余为单来源；可用于读取和单来源时序通路。开展多来源条件对齐时，需要选择更多实际重叠的航次，任务重叠统计位于准备结果的 `source_overlap`。

### 准备与运行

```bash
python -m uav_abnormality.data.download \
  --manifest configs/datasets/lapse_rate_pilot_manifest.json \
  --destination data/raw/lapse_rate --max-bytes 60000000
python -m uav_abnormality prepare --dataset lapse-rate \
  --config configs/datasets/lapse_rate.json --output data/processed/lapse_rate.csv
python -m uav_abnormality inspect --reports data/processed/lapse_rate.csv
```

`configs/lapse_rate_io_smoke.yaml` 用于当前文件的单来源通路，`configs/lapse_rate.yaml` 是多来源训练入口。需要完整数据时可用下载器的 `--pattern` 选择航次。温度或湿度通过适配器的 `features` 配置切换。

## UAV3D：多机视觉报告

[UAV3D 官方仓库](https://github.com/huiyegit/UAV3D)提供仿真多机 3D 感知数据，五架 UAV 共享场景。项目从每机独立感知输出开始，将目标位置转换到全局坐标，再按目标和时间组织报告。

本机已取得 mini 的六张元数据表，包含 20 场景、400 个同步样本、五架 UAV 和 25 个相机通道；图像需另行下载。完整 mini ZIP 约 7.17 GB，固定清单为 `configs/datasets/uav3d_mini_manifest.json`。

```bash
python scripts/fetch_uav3d_metadata.py \
  --destination data/raw/uav3d_metadata --max-download-bytes 30000000
python -m uav_abnormality.data.download \
  --manifest configs/datasets/uav3d_mini_manifest.json \
  --destination data/raw/uav3d_archives --max-bytes 8000000000
```

### 三种报告入口

| 输入 | 需要的字段或配置 |
|---|---|
| 已统一全局坐标的 CSV/JSONL | `timestamp, sequence_id, source_id, x, y, z`，可提供统一 `task_id` |
| 保留相机样本 token 的 CSV/JSONL | `sample_data_token, x, y, z, coordinate_frame`，由元数据补齐时间和来源 |
| 每机独立的 nuScenes `results` JSON | 用 `report_files` 指定各文件及对应来源 |

同机多个相机归属同一来源。默认每机使用一个 BOTTOM 相机；已有多相机结果先去重。元数据时间戳单位是纳秒，场景内采样间隔约 2.65–4.15 秒，默认关联和窗口间隔上限采用 5 秒。

原始 UAV3D 使用 `metadata_coordinate_convention: uav3d_carla`；已转成标准 nuScenes 的数据使用 `nuscenes_optical`。两种坐标链由转换器分别处理。提供统一目标 ID 时直接使用，否则按类别、距离门限与运动轨迹做跨机关联。

```bash
python -m uav_abnormality prepare --dataset uav3d \
  --config configs/datasets/uav3d.json --output data/processed/uav3d.csv
python -m uav_abnormality inspect --reports data/processed/uav3d.csv
```

配置示例位于 `configs/datasets/uav3d.json`。另有 [固定 RGB 前端](perception.md)，输出地面接触点报告；3D 中心与地面接触点分别作为各自的报告语义。`oracle_semisynthetic` 模式使用标注加噪声，适合数据通路调试，其来源标记随输出保存。

视觉实验的异常注入位于报告生成及目标关联之后；因此与当前数值主线一致，评估提交数值的质量。原始检测 AP、自然漏检/误检和报告异常检测分别对应不同评测任务。
