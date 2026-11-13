# 可选 RGB 前端：从 UAV3D 图像生成报告

`uav_abnormality.data.perception` 将每架 UAV 的 RGB 图像转换成车辆地面接触点报告。它使用固定的 torchvision **SSDLite320 + MobileNetV3 Large COCO_V1**，检测 car、bus、truck，然后把检测框底边中点投影到给定地面平面。当前三套数值实验直接读取报告，这个前端用于后续视觉任务。

输出表示水平地面假设下的 2.5D 位置，`z` 等于场景地面高度。所需输入是原始图片、相机元数据和已知地面平面；完整 3D 中心/尺寸/姿态任务可接入其他固定检测器。

## 安装与导出

当前环境已具备前端依赖和 COCO_V1 权重缓存。其他环境可在仓库根目录安装：

```bash
python -m pip install -e '.[perception]'
python -m uav_abnormality.data.perception \
  --raw-dir data/raw/uav3d \
  --output data/interim/uav3d_ssdlite_reports.jsonl \
  --ground-z 0 --device cpu --max-frames 25
```

将 `--ground-z` 改为仿真场景或独立地图给定的全局地面高度。`--max-frames 25` 便于先检查少量图片，完整导出时去掉该参数。默认每架 UAV 选择 BOTTOM 相机，来源 ID 对应 UAV。权重默认缓存于 `data/weights/torch`，可用 `--weights-cache` 更改位置。

模型与权重说明见 [torchvision 文档](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)。环境使用 `torchvision==0.19.1`，与 PyTorch 2.4.1 配套。

## 投影和坐标

原始 UAV3D 默认选择 `--camera-convention uav3d_carla`：相机平移采用元数据中的全局位置，四元数按照 CARLA 约定转换，再将机体坐标转成光学坐标。正规化为 nuScenes 的数据选择 `nuscenes_optical`，通过 sensor→ego→global 坐标链处理。

依据是作者的 [相机可视化](https://github.com/huiyegit/UAV3D/blob/main/visualization/functions.py)、[数据转换器](https://github.com/huiyegit/UAV3D/blob/main/perception/bevfusion/tools/data_converter/uav_converter_mini.py) 和 [投影代码](https://github.com/huiyegit/UAV3D/blob/main/perception/bevfusion/mmdet3d/datasets/nuscenes_dataset.py)。

投影先用相机内参计算像素射线，转到全局后与地面平面求交。默认保留斜距 250 米以内、向下方向余弦至少 0.05 的有效交点。该设置适合平坦场景；坡地和多层道路可改用具有地形或深度信息的前端。

## 接入报告质检

JSONL/CSV 输出包含样本 token、来源、时间、相机通道、全局 x/y/z、类别、置信度及 `position_semantics=bbox_bottom_center_on_ground_plane`。旁边的 `.manifest.json` 保存权重、参数、处理帧数和文件哈希，多个质检算法共用同一份输出。

UAV3D 准备配置示例：

```yaml
raw_dir: data/raw/uav3d
reports_path: data/interim/uav3d_ssdlite_reports.jsonl
mode: detector_reports
coordinate_frame: global
metadata_coordinate_convention: uav3d_carla
association_gate_m: 5.0
association_max_gap_seconds: 5.0
metadata_timestamp_unit: ns
min_physical_sources: 2
```

适配器按观测位置完成跨机目标关联，再输出统一报告表。原报告保留未知故障标签，随后可使用共享故障注入模块测试报告污染。`tests/test_perception.py` 验证投影、来源映射、过滤规则和输出协议。
