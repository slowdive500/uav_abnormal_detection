import gzip
import json

import numpy as np
import pandas as pd
import pytest

from uav_abnormality.data.intel_berkeley import prepare_intel_berkeley, REFERENCE_SEMANTICS
from uav_abnormality.data.schema import validate_table


def _raw(tmp_path, rows, locations="1 21.5 23\n2 24.5 20\n3 19.5 19\n4 22.5 15\n"):
    (tmp_path / "mote_locs.txt").write_text(locations)
    with gzip.open(tmp_path / "data.txt.gz", "wt") as handle:
        handle.write("\n".join(rows) + "\n")
    return {"raw_dir": str(tmp_path), "time_bin_seconds": 60, "query_node_ids": [4]}


def test_real_coordinates_query_exclusion_and_causal_bins(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1 20 40 100 2.7",
        "2004-03-01 00:00:50 2 1 24 40 100 2.7",
        "2004-03-01 00:00:20 1 2 21 40 100 2.7",
        "2004-03-01 00:00:30 1 3 22 40 100 2.7",
        "2004-03-01 00:00:40 1 4 99 40 100 2.7",
    ])
    config.update(node_ids=[1, 2, 3, 4], min_sources=3, simulated_operation_height_m=2)
    table = prepare_intel_berkeley(config)
    assert len(table) == 3
    assert set(table.raw_node_id) == {1, 2, 3}
    assert table.iloc[0].value_0 == 22
    assert table.iloc[0].sample_count == 2
    assert (table.timestamp == pd.Timestamp("2004-03-01 00:01:00Z")).all()
    assert (table.sample_end < table.timestamp).all()
    assert np.allclose(table.iloc[0][["condition_x", "condition_y"]].to_numpy(float), [21.5, 23])
    assert "condition_z" not in table
    assert table.condition_simulated_operation_height_m.tolist() == [2, 2, 2]
    assert (table.label == -1).all()
    assert "truth_0" not in table
    query = table.attrs["evaluation_queries"][0]
    assert query["reference_0"] == 99
    assert query["reference_semantics"] == REFERENCE_SEMANTICS
    assert "value_0" not in query and "source_id" not in query
    assert query["task_id"] == table.task_id.iloc[0]
    json.dumps(table.attrs, allow_nan=False)
    assert len(validate_table(table)) == 3


def test_missing_times_and_sources_are_never_interpolated(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1 20 40 100 2.7",
        "2004-03-01 00:00:01 1 2 nan 40 100 2.7",
        "2004-03-01 00:02:01 5 2 21 40 100 2.7",
        "2004-03-02 00:02:01 9 1 22 40 100 2.7",
    ])
    table = prepare_intel_berkeley(config)
    assert len(table) == 3
    assert table.timestamp.nunique() == 3
    assert pd.Timestamp("2004-03-01 00:02:00Z") not in set(table.timestamp)
    assert set(table.sequence_id) == {"intel_berkeley_20040301", "intel_berkeley_20040302"}
    assert table.attrs["raw_audit"]["nonfinite_temperature_rows"] == 1
    assert table.attrs["source_overlap"] == {"1": 3}
    assert set(table.source_id) == {"virtual_uav_node_01", "virtual_uav_node_02"}


def test_official_empty_space_fields_never_shift_voltage_into_temperature(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1    2.52732",
        "2004-03-01 00:00:01 1 2 20 40  2.7",
        "2004-03-01 00:00:01 1 3 22   2.7",
        "2004-03-01 00:00:02 2",
    ])
    table = prepare_intel_berkeley(config)
    assert table.raw_node_id.tolist() == [2, 3]
    assert table.value_0.tolist() == [20, 22]
    assert table.attrs["raw_audit"]["nonfinite_temperature_rows"] == 1
    assert table.attrs["raw_audit"]["invalid_node_rows"] == 1


def test_natural_outliers_unknown_and_optional_policy_audited(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1 20 40 100 2.7",
        "2004-03-01 00:00:02 1 2 500 40 100 2.7",
        "2004-03-01 00:00:03 1 3 -100 40 100 2.7",
        "bad-date 00:00:01 1 1 20 40 100 2.7",
        "2004-03-01 00:00:01 1 55 20 40 100 2.7",
    ])
    raw = prepare_intel_berkeley(config)
    assert raw.value_0.tolist() == [20, 500, -100]
    assert raw.label.tolist() == [-1, -1, -1]
    filtered = prepare_intel_berkeley(dict(config, temperature_bounds_c=[0, 40]))
    assert filtered.value_0.tolist() == [20]
    audit = filtered.attrs["raw_audit"]
    assert audit["temperature_bounds_rejected_rows"] == 2
    assert audit["invalid_timestamp_rows"] == 1
    assert audit["invalid_node_rows"] == 1
    assert filtered.attrs["validity_policy"] == {"temperature_bounds_c": [0, 40]}


def test_duplicate_packets_and_chunk_boundaries_do_not_reweight(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1 10 40 100 2.7",
        "2004-03-01 00:00:20 2 1 30 40 100 2.7",
        "2004-03-01 00:00:01 1 1 10 40 100 2.7",
    ])
    table = prepare_intel_berkeley(dict(config, chunksize=1))
    assert table.value_0.tolist() == [20]
    assert table.sample_count.tolist() == [2]
    assert table.attrs["raw_audit"]["duplicate_packets_removed"] == 1


def test_spatial_neighbourhood_and_query_join_use_actual_geometry(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:01 1 1 20 40 100 2.7",
        "2004-03-01 00:00:01 1 2 21 40 100 2.7",
        "2004-03-01 00:00:01 1 3 22 40 100 2.7",
        "2004-03-01 00:00:01 1 4 23 40 100 2.7",
        "2004-03-01 00:01:01 2 4 24 40 100 2.7",
    ], locations="1 0 0\n2 1 0\n3 10 10\n4 0 1\n")
    config.update(spatial_regions=[{"id": "near", "center_xy": [0, 0], "radius_m": 2}], min_sources=2)
    table = prepare_intel_berkeley(config)
    assert table.raw_node_id.tolist() == [1, 2]
    assert table.task_id.tolist() == ["near", "near"]
    assert len(table.attrs["evaluation_queries"]) == 1  # No reports in second query interval.


def test_time_selection_is_inclusive_start_exclusive_end(tmp_path):
    config = _raw(tmp_path, [
        "2004-03-01 00:00:00 1 1 20 40 100 2.7",
        "2004-03-01 00:01:00 2 1 21 40 100 2.7",
        "2004-03-01 00:02:00 3 1 22 40 100 2.7",
    ])
    table = prepare_intel_berkeley(dict(config, start="2004-03-01 00:01:00", end="2004-03-01 00:02:00"))
    assert table.value_0.tolist() == [21]
    assert table.attrs["raw_audit"]["outside_time_range_rows"] == 2


@pytest.mark.parametrize("change,match", [
    ({"query_node_ids": [1, 2, 3, 4]}, "No reporting"),
    ({"node_ids": [1, 1]}, "duplicate"),
    ({"node_ids": [1.5]}, "integer"),
    ({"time_bin_seconds": float("nan")}, "finite"),
    ({"temperature_bounds_c": [40, 0]}, "lower"),
])
def test_invalid_configuration_fails_explicitly(tmp_path, change, match):
    config = _raw(tmp_path, ["2004-03-01 00:00:00 1 1 20 40 100 2.7"])
    config.update(change)
    with pytest.raises(ValueError, match=match):
        prepare_intel_berkeley(config)
