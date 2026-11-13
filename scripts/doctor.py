"""Check installed packages and CPU/GPU readiness without changing the machine."""
import json
import sys
import importlib.metadata as metadata
import torch
packages = ['torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'xarray', 'netCDF4', 'einops', 'opt-einsum']
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': versions,
                  'cuda_available': torch.cuda.is_available(), 'torch_cuda_runtime': torch.version.cuda,
                  'gpu_count': torch.cuda.device_count()}, indent=2))
