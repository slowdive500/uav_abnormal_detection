# 上游来源与许可

本项目实现跨来源条件对齐方法，并基于 PRBTD、TSAD-C 的上游代码构建两个比较算法。
固定上游版本、原始文件校验值和许可证保存在 `third_party/`。

- PRBTD：glab2019/PRBTD，commit `078bf86f09fdb17dafe90d305986559f22efd4a8`。上游快照没有明确许可证文件，原 README 的版权说明保留。未给这部分代码另行授予开放源码许可证。
- TSAD-C：Armanfard-Lab/TSADC，commit `7013d7d2053044acdd3aa980bcf309d48f94de0d`。上游 CC BY-NC-SA 3.0，许可全文在 `third_party/TSADC/LICENSE.txt`；S4 移植代码和衍生模块保留其约束。
- UAV3D 原始数据、LAPSE-RATE 原始数据以及 torchvision 预训练权重分别受原发布方条款约束，未把下载资源重新授权为本项目代码。

目前用于本地研究开发。新增训练与数据接口不改变上游代码的权利归属。

## Report-level datasets

- **queensu/crowd_temperature**: Mohannad A. Alswailim, Hossam S. Hassanein, Mohammad Zulkernine, CRAWDAD v2015-11-20, DOI 10.15783/C7CG65. Original CSV supplied by the user and verified against the published MD5. Real taxi participation with author-generated temperatures, organized here as virtual UAV report tasks. Metadata: https://ieee-dataport.org/open-access/crawdad-queensucrowdtemperature .
- **TaxiBJ**: Junbo Zhang, Yu Zheng, Dekang Qi, *Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction*, AAAI 2017. Original matrices from a public fixed-commit mirror, verified against the original MD5 listing. The data are ground taxi aggregate flows, used to generate explicitly simulated UAV reports. Dataset and mirror provenance are pinned in data/manifests/taxibj.json; repository code licenses do not establish a separate dataset license.
- **Intel Berkeley Research Lab**: February–April 2004 sensor deployment. Source https://db.csail.mit.edu/labdata/labdata.html . Please acknowledge Intel Research Berkeley, UC Berkeley, and MIT when publishing results using these data, as requested by the source. Original observations and node locations are retained; virtual UAV identities are an experimental mapping.

Raw data and run artifacts are local resources, excluded from Git. Each original file is accompanied by acquisition checksums and a receipt.
