#!/usr/bin/env bash
set -euo pipefail
uav_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uav_conda_bin="${CONDA_EXE:-conda}"
uav_environment="UAV_Abnormality_Detect"
if ! "$uav_conda_bin" run -n "$uav_environment" python -c 'import sys' >/dev/null 2>&1; then
  "$uav_conda_bin" create -n "$uav_environment" --clone RL_Unlearning -y
fi
"$uav_conda_bin" run -n "$uav_environment" python -m pip install -r "$uav_repo_root/requirements-core.txt" -r "$uav_repo_root/requirements-perception.txt"
# Some RL_Unlearning installations contain stale NumPy distribution metadata.
if ! "$uav_conda_bin" run -n "$uav_environment" python -c 'import importlib.metadata as m, sys; d=[p for p in m.distributions() if p.metadata.get("Name", "").lower()=="numpy"]; sys.exit(0 if len(d)==1 and d[0].version=="1.24.3" else 1)'; then
  "$uav_conda_bin" run -n "$uav_environment" python -m pip install --force-reinstall --no-deps numpy==1.24.3
fi
"$uav_conda_bin" run -n "$uav_environment" python -m pip install --no-deps --no-build-isolation -e "$uav_repo_root"
"$uav_conda_bin" run -n "$uav_environment" python -m pip check
