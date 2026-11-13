"""Acquire, prepare and compare the three numerical-report experiments."""
import argparse
import copy
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_abnormality.cli import main as cli_main, read_config
from uav_abnormality.data.download import download_manifest, verify_file
from uav_abnormality.experiment import train_experiment

DATASETS = {'crowd_temperature': 'crowd-temperature', 'taxibj': 'taxibj', 'intel_berkeley': 'intel-berkeley'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_status(output, report):
    report['counts'] = {state: sum(r['status'] == state for r in report['runs'])
                        for state in ('pending', 'running', 'completed', 'failed', 'unavailable')}
    report['total_runs'] = len(report['runs'])
    write_json(output / 'run_status.json', report)
    # Backward-compatible filename, now a compact index instead of duplicated metrics.
    write_json(output / 'benchmark_status.json', report)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_attack_mode(config, attack_mode):
    """Resolve the experimental arm without mutating shared dataset YAML files."""
    resolved = copy.deepcopy(config)
    resolved['attack_mode'] = attack_mode
    if attack_mode == 'none':
        resolved['corruption'] = {
            name: {'protocol': 'none', 'assume_unlabelled_normal': True}
            for name in ('train', 'validation', 'calibration', 'test')}
        if 'protocol_feasibility' in resolved:
            resolved['protocol_feasibility']['min_scored_attacks_per_split'] = 0
    return resolved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    p.add_argument('--algorithms', nargs='+', choices=['condition_alignment', 'prbtd', 'tsadc'],
                   default=['condition_alignment', 'prbtd', 'tsadc'])
    p.add_argument('--download', action='store_true', help='Download pinned public raw files before preparing')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--skip-unavailable', action='store_true', help='Record unavailable raw data explicitly and continue others')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 42, 46, 47, 1024])
    p.add_argument('--epochs', type=int, help='Override training epochs; 1 is an execution check, not a benchmark result')
    p.add_argument('--attack-mode', choices=['configured', 'none'], default='configured',
                   help='Use configured injections, or preserve original reports in all four splits')
    p.add_argument('--no-attacks', dest='attack_mode', action='store_const', const='none',
                   help='Alias for --attack-mode none; disables injections during training, validation, calibration and test')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true', help='Resume the identical saved plan; keep completed runs')
    args = p.parse_args(argv)
    for key in ('datasets', 'algorithms', 'seeds'):
        values = getattr(args, key)
        if len(set(values)) != len(values):
            p.error('Duplicate ' + key + ' would repeat the same experiment')
    if any(seed < 0 or seed + 4036 >= 2 ** 32 for seed in args.seeds):
        p.error('Seeds must support the NumPy uint32 seed range including split offsets')
    output = Path(args.output)
    plan = {'datasets': args.datasets, 'algorithms': args.algorithms, 'seeds': args.seeds,
            'epoch_override': args.epochs, 'prepare_only': args.prepare_only,
            'attack_mode': args.attack_mode,
            'config_sha256': {d: fingerprint('configs/experiments/' + d + '.yaml') for d in args.datasets}}
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        saved_plan = (json.loads((output / 'run_plan.json').read_text())
                      if (output / 'run_plan.json').is_file() else {})
        # Plans created before this switch always used configured injections.
        saved_plan.setdefault('attack_mode', 'configured')
        if saved_plan != plan:
            raise ValueError('Resume requires the identical datasets, methods, seeds, training budget, attack mode and configs')
        report = json.loads((output / 'run_status.json').read_text())
    else:
        report = {'datasets': {}, 'epoch_override': args.epochs, 'attack_mode': args.attack_mode,
                  'runs': [] if args.prepare_only else [
                      {'dataset': dataset, 'algorithm': algorithm, 'seed': seed, 'status': 'pending'}
                      for dataset in args.datasets for algorithm in args.algorithms for seed in args.seeds]}
        write_json(output / 'run_plan.json', plan)
    save_status(output, report)
    (output / 'logs').mkdir(exist_ok=True)
    for dataset in args.datasets:
        config = resolve_attack_mode(read_config('configs/experiments/' + dataset + '.yaml'), args.attack_mode)
        try:
            if args.download:
                manifest = Path('data/manifests') / (dataset + '.json')
                metadata = json.loads(manifest.read_text())
                if metadata.get('access') == 'manual_authenticated_download':
                    required = metadata.get('required_files', [])
                    if not required or not all(verify_file(Path('data/raw') / dataset / f['path'], f) for f in required):
                        raise FileNotFoundError(metadata.get('instructions', 'Dataset requires an authenticated download'))
                elif dataset == 'taxibj':
                    from uav_abnormality.data.taxibj import download_taxibj
                    download_taxibj(manifest, Path('data/raw') / dataset)
                else:
                    download_manifest(metadata, Path('data/raw') / dataset)
            if args.prepare or args.prepare_only:
                cli_main(['prepare', '--dataset', DATASETS[dataset], '--config',
                          'configs/datasets/' + dataset + '.yaml', '--output', config['data']['path']])
            if not Path(config['data']['path']).is_file():
                raise FileNotFoundError('Missing prepared reports: ' + config['data']['path'])
        except (OSError, ValueError) as exc:
            report['datasets'][dataset] = {'status': 'unavailable', 'reason': str(exc)}
            for row in report['runs']:
                if row['dataset'] == dataset:
                    row.update(status='unavailable', error=str(exc))
            save_status(output, report)
            if args.skip_unavailable:
                continue
            raise
        inputs = {'reports': fingerprint(config['data']['path'])}
        queries = config.get('evaluation', {}).get('queries_path')
        if queries:
            inputs['queries'] = fingerprint(queries)
        previous = report['datasets'].get(dataset, {}).get('input_sha256')
        if args.resume and previous is not None and previous != inputs:
            raise ValueError('Prepared inputs changed since the saved run: ' + dataset)
        report['datasets'][dataset] = {'status': 'prepared', 'reports': config['data']['path'],
                                        'input_sha256': inputs}
        save_status(output, report)
        if args.prepare_only:
            continue
        for algorithm in args.algorithms:
            for seed in args.seeds:
                row = next(r for r in report['runs'] if (r['dataset'], r['algorithm'], r['seed']) == (dataset, algorithm, seed))
                destination = output / dataset / algorithm / str(seed)
                if args.resume and row['status'] == 'completed' and (destination / 'metrics.json').is_file():
                    print('Reusing completed %s / %s / %s' % (dataset, algorithm, seed), flush=True)
                    continue
                if destination.exists() and any(destination.iterdir()):
                    # Preserve partial attempts instead of deleting their evidence.
                    archive = output / 'failed_attempts' / dataset / algorithm
                    archive.mkdir(parents=True, exist_ok=True)
                    destination.rename(archive / ('%s_%d' % (seed, time.time_ns())))
                run_config = copy.deepcopy(config)
                run_config['algorithm'], run_config['seed'] = algorithm, seed
                run_config['model'] = run_config.get('models', {}).get(algorithm, {})
                if args.epochs is not None:
                    run_config.setdefault('training', {})['epochs'] = args.epochs
                row.pop('error', None)
                row.update(status='running', log='logs/%s_%s_%s.log' % (dataset, algorithm, seed))
                save_status(output, report)
                print('Running %s / %s / seed %s (%d/%d completed)' % (
                    dataset, algorithm, seed, report['counts']['completed'], report['total_runs']), flush=True)
                start = time.perf_counter()
                with (output / row['log']).open('a', encoding='utf-8') as log:
                    try:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            train_experiment(run_config, destination)
                        row.update(status='completed', metrics=str((destination / 'metrics.json').relative_to(output)))
                    except Exception as exc:
                        row.update(status='failed', error='%s: %s' % (type(exc).__name__, exc))
                        traceback.print_exc(file=log)
                row['elapsed_seconds'] = time.perf_counter() - start
                save_status(output, report)
                print('%s %s / %s / seed %s (%.1fs)%s' % (
                    row['status'], dataset, algorithm, seed, row['elapsed_seconds'],
                    ': ' + row['error'] if 'error' in row else ''), flush=True)
        selected = [r for r in report['runs'] if r['dataset'] == dataset]
        report['datasets'][dataset]['status'] = ('completed' if all(r['status'] == 'completed' for r in selected)
                                                 else 'completed_with_failures')
        save_status(output, report)
    if not args.prepare_only:
        from uav_abnormality.benchmark_analysis import analyze_benchmarks
        analyze_benchmarks(output, expected_datasets=args.datasets, expected_algorithms=args.algorithms,
                           expected_seeds=args.seeds)
        report['analysis'] = 'analysis/REPORT.md'
        save_status(output, report)
        print('Paper tables: ' + str(output / 'analysis/REPORT.md'), flush=True)
        print('Mean / sample SD / valid seeds: ' + str(output / 'analysis/summary.csv'), flush=True)
    return 1 if any(r['status'] in ('failed', 'unavailable') for r in report['runs']) else 0


if __name__ == '__main__':
    sys.exit(main())
