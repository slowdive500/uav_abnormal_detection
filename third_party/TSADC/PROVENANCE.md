# TSADC provenance

- Authors: Thi Kieu Khanh Ho and Narges Armanfard.
- Repository: https://github.com/Armanfard-Lab/TSADC
- Pinned commit: `7013d7d2053044acdd3aa980bcf309d48f94de0d`.
- Paper: https://proceedings.mlr.press/v286/ho25a.html
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported; unchanged upstream text in `LICENSE.txt`.
- Accessed: 2026-11-12.

Files in this directory are an unchanged upstream reference subset (`README.md`, `LICENSE.txt`, `args.py`, `main_run.py`, `model/TSADC.py`, `model/model_dependency.py`, `utils/model_utils.py`). They are not imported by the executable framework and do not constitute a standalone installation of the upstream repository. Upstream README and code are preserved as source evidence, not as instructions to run its EEG-specific script or to install its complete dependency stack.

Every retained upstream file was verified against the Git blob SHA from the pinned commit. `UPSTREAM_HASHES.json` records those blob hashes and local SHA-256 hashes.

Executable derivatives are `uav_abnormality/models/tsadc.py` and `_tsadc_s4.py`, distributed under the same CC BY-NC-SA 3.0 license. `_tsadc_s4.py` retains the final S4 definitions and diffusion residual blocks from upstream `model/model_dependency.py`, removing the older overwritten definitions, optional compiled/KeOps kernels and Lightning logging. A pure PyTorch Cauchy/Vandermonde path is used. Diffusion time embeddings allocate on the input device; channel dropout uses Dropout1d; the configured S4 state dimension is forwarded instead of being silently ignored. Upstream credits https://github.com/AI4HealthUOL/SSSD and https://github.com/tsy935/graphs4mer for these components; those attributions are preserved.

`tsadc.py` adapts orchestration, masks and tensor layouts to observed `[batch, window, physical source, feature]` data. It retains S4 diffusion, learned attention/cosine graphs, GINE message equations, graph regularizers, and the two-component anomaly score. Dense GINE replaces PyG storage only. Numerical, missing-data and source-endpoint scoring adaptations are documented in `docs/tsadc.md`. No claim is made that the compact default configuration reproduces the paper's reported results.
