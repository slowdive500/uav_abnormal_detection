"""Compare retrained no-injection experiments against fixed per-dataset attack runs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_abnormality.attack_comparison import analyze_attack_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-attack", required=True, help="No-injection benchmark root")
    parser.add_argument("--attack", action="append", required=True, metavar="DATASET=ROOT")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    roots = {}
    for mapping in args.attack:
        dataset, separator, root = mapping.partition("=")
        if not separator or not dataset or not root or dataset in roots:
            parser.error("--attack requires one unique DATASET=ROOT per dataset")
        roots[dataset] = root
    summary = analyze_attack_comparison(args.no_attack, roots, args.datasets, args.algorithms, args.seeds, args.output)
    output = Path(args.output) if args.output else Path(args.no_attack) / "comparison"
    print(json.dumps(dict(summary["completion"], report=str(output / "REPORT.md")), ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
