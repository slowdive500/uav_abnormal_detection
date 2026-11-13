"""Run only the two external algorithms under identical data/protocol seeds."""
import argparse
import json
from pathlib import Path
import copy
import numpy as np
import yaml
from uav_abnormality.experiment import train_experiment

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 71])
a = p.parse_args()
config = yaml.safe_load(Path(a.config).read_text())
rows = []
for algorithm in ("prbtd", "tsadc"):
    for seed in a.seeds:
        c = copy.deepcopy(config)
        c["algorithm"], c["seed"] = algorithm, seed
        c["model"] = c.get("models", {}).get(algorithm, {})
        result = train_experiment(c, Path(a.output) / algorithm / str(seed))
        rows.append(dict(result, seed=seed))
summary = {}
for algorithm in ("prbtd", "tsadc"):
    summary[algorithm] = {}
    for metric in ("average_precision", "auroc", "f1", "false_positive_rate"):
        values = [r[metric] for r in rows if r["algorithm"] == algorithm and r.get(metric) is not None]
        summary[algorithm][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)} if values else None
Path(a.output, "comparison.json").write_text(json.dumps({"runs": rows, "summary": summary}, indent=2, allow_nan=False))
print(json.dumps(summary, indent=2))
