"""Command-line interface; paths are relative to the current working directory."""
import argparse
import json
from pathlib import Path
import sys
import yaml

def read_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    return config

def main(argv=None):
    parser = argparse.ArgumentParser(description="Condition alignment / PRBTD / TSAD-C contribution quality")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Convert verified raw observations to canonical reports")
    prepare.add_argument("--dataset", choices=["lapse-rate", "uav3d", "crowd-temperature", "taxibj", "intel-berkeley"], required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect", help="Audit physical-source and simultaneous-task coverage")
    inspect.add_argument("--reports", required=True)
    for name in ("run", "train"):
        p = sub.add_parser(name, help="Train + calibrate" + (" + held-out evaluation" if name == "run" else ""))
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--algorithm", choices=["condition_alignment", "prbtd", "tsadc"])
        p.add_argument("--seed", type=int)
    for name in ("evaluate", "infer"):
        p = sub.add_parser(name, help="Load a trusted checkpoint and score chronological reports")
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--reports", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="auto")
    fixture = sub.add_parser("fixture", help="Generate a clearly labelled synthetic integration fixture")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data.schema import write_table
        config = read_config(args.config)
        if args.dataset == "lapse-rate":
            from .data.lapse_rate import prepare_lapse_rate
            frame = prepare_lapse_rate(config)
        elif args.dataset == "uav3d":
            from .data.uav3d import prepare_uav3d
            frame = prepare_uav3d(config)
        elif args.dataset == "crowd-temperature":
            from .data.crowd_temperature import prepare_crowd_temperature
            frame = prepare_crowd_temperature(config)
        elif args.dataset == "taxibj":
            from .data.taxibj import prepare_taxibj
            frame = prepare_taxibj(config)
        else:
            from .data.intel_berkeley import prepare_intel_berkeley
            frame = prepare_intel_berkeley(config)
        audit = dict(frame.attrs)
        queries = audit.pop("evaluation_queries", None)
        if queries is not None:
            import pandas as pd
            query_path = Path(args.output).with_suffix(".queries.csv")
            query_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(queries).to_csv(query_path, index=False)
            audit["evaluation_query_file"] = str(query_path)
            audit["evaluation_query_count"] = len(queries)
        result = write_table(frame, args.output, {"dataset": args.dataset, "adapter_config": config,
                                                  "adapter_audit": audit})
    elif args.command == "inspect":
        from .data.schema import read_table, coverage_report
        result = coverage_report(read_table(args.reports))
    elif args.command in ("train", "run"):
        from .experiment import train_experiment
        config = read_config(args.config)
        if args.algorithm:
            config["algorithm"] = args.algorithm
        if "models" in config:
            # Resolve the YAML algorithm as well as an explicit CLI override,
            # identically to the multi-algorithm benchmark runner.
            config["model"] = config["models"].get(config.get("algorithm", "prbtd"), {})
        if args.seed is not None:
            config["seed"] = args.seed
        result = train_experiment(config, args.output, evaluate_test=args.command == "run")
    elif args.command in ("evaluate", "infer"):
        from .experiment import evaluate_checkpoint
        result = evaluate_checkpoint(args.checkpoint, args.reports, args.output, args.device, metrics=args.command == "evaluate")
    else:
        from .fixtures import make_fixture
        result = make_fixture(args.output, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        sys.exit(2)
