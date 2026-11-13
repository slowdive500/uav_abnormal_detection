"""Aggregate benchmark seeds into four paper-facing result files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_abnormality.benchmark_analysis import analyze_benchmarks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Root containing dataset/method/seed runs")
    parser.add_argument("--output", help="Default: INPUT/analysis")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(argv)
    summary = analyze_benchmarks(args.input, args.datasets, args.algorithms, args.seeds, args.output)
    output = Path(args.output) if args.output else Path(args.input) / "analysis"
    print(json.dumps(dict(summary["completion"], report=str(output / "REPORT.md")), ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
