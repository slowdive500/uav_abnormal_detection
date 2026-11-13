"""Tiny synthetic fixture only; NEVER a substitute for LAPSE-RATE or UAV3D."""
import numpy as np
import pandas as pd
from .data.schema import write_table

def make_fixture(path, seed=42, sequences=8, steps=64, sources=5):
    rng = np.random.RandomState(seed)
    records = []
    for sequence in range(sequences):
        for t in range(steps):
            common = np.array([np.sin(t / 12.) + .01 * sequence, np.cos(t / 17.)])
            for s in range(sources):
                value = common + rng.normal(0, .04, 2)
                records.append({"sequence_id": "synthetic_%02d" % sequence, "task_id": "fixture_only",
                                "timestamp": 1_500_000_000 + sequence * 3600 + t * 5,
                                "source_id": "uav_%d" % s, "value_0": value[0], "value_1": value[1],
                                "condition_altitude_agl": 40 + s * 5, "label": -1,
                                "data_origin": "synthetic_integration_fixture"})
    return write_table(pd.DataFrame(records), path, {"kind": "synthetic_integration_fixture", "seed": seed,
                                                   "is_real_dataset": False})
