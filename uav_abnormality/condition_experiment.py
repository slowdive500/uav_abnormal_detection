"""Train and deploy the proposal's condition alignment and dynamic reputation."""
import copy
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from .data.schema import read_table, coverage_report, feature_columns, KEYS
from .protocol import Standardizer, split_table, SPLITS, sha256_file
from .condition_data import ConditionScaler, ConditionWindows
from .condition_runtime import score_condition_windows, apply_reputation_and_estimate, environment_metrics
from .models.condition_alignment import ConditionAlignment
from .calibration import fit_label_trim, label_trim_pvalues, label_trim_predictions
from .attack_pipeline import (prepare_corrupted_splits, write_attack_audits,
                              restore_report_values, matching_attack_metadata)


def make_condition_windows(frame, checkpoint, allow_empty=False):
    return ConditionWindows(frame, checkpoint['sources'], Standardizer.from_dict(checkpoint['scaler']),
        ConditionScaler.from_dict(checkpoint['condition_scaler']),
        **dict(checkpoint['window'], allow_empty=allow_empty))


def build_condition_model(checkpoint, device):
    return ConditionAlignment(len(checkpoint['scaler']['columns']),
                              len(checkpoint['condition_scaler']['columns']),
                              **checkpoint['model_config']).to(device)


def _loss(model, batch, device):
    return model.loss(**{key: batch[key].to(device)
                         for key in ('x', 'mask', 'conditions', 'condition_mask')})


def plan_calibration_registry(registry, window):
    """Schedule complete temporal windows before any score/label inspection."""
    registry = registry.copy()
    registry['calibration_planned'] = False
    size, stride = int(window['window_size']), int(window.get('stride', 1))
    maximum_gap = window.get('max_gap_seconds')
    for _, group in registry.groupby(['sequence_id', 'task_id'], sort=True):
        times = np.sort(group.timestamp.unique())
        planned = []
        for end in range(size - 1, len(times), stride):
            if maximum_gap is not None and np.any(np.diff(times[end-size+1:end+1]) > maximum_gap):
                continue
            planned.append(times[end])
        registry.loc[group.index, 'calibration_planned'] = group.timestamp.isin(planned)
    return registry


def train_condition_experiment(config, output, evaluate_test=True):
    from .experiment import dump_json, seed_everything, device_for
    config = copy.deepcopy(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty run directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get('seed', 42))
    training, method = config.get('training', {}), config.get('method', {})
    seed_everything(seed, int(training.get('threads', 1)))
    device = device_for(training.get('device', 'auto'))
    raw = read_table(config['data']['path'])
    splits, manifest = split_table(raw, config.get('split', {}))
    splits, corruptions, attack_profile = prepare_corrupted_splits(splits, config, seed)
    scaler = Standardizer().fit(splits['train'])
    condition_scaler = ConditionScaler().fit(splits['train'], method.get('condition_columns', []))
    window = dict(config.get('window', {}))
    window.setdefault('window_size', 16)
    window.setdefault('min_sources', 3)
    window.setdefault('stride', 1)
    model_config = config.get('model', config.get('models', {}).get('condition_alignment', {}))
    checkpoint = {'format_version': 2, 'algorithm': 'condition_alignment', 'model_config': model_config,
                  'sources': sorted(splits['train'].source_id.unique().tolist()),
                  'scaler': scaler.to_dict(), 'condition_scaler': condition_scaler.to_dict(),
                  'window': window, 'seed': seed, 'input_sha256': sha256_file(config['data']['path']),
                  'config': config}
    needed = SPLITS if evaluate_test else SPLITS[:-1]
    datasets = {name: make_condition_windows(splits[name], checkpoint,
                allow_empty=name in ('calibration', 'test')) for name in needed}
    manifest.update(windows={key: len(value) for key, value in datasets.items()},
                    source_vocabulary=checkpoint['sources'],
                    coverage={key: coverage_report(value) for key, value in splits.items()},
                    corruptions=corruptions, condition_scaler=condition_scaler.to_dict())
    if attack_profile is not None:
        manifest['attack_audits'] = write_attack_audits(splits, config, output, attack_profile, corruptions)
    dump_json(output / 'split_manifest.json', manifest)
    (output / 'splits').mkdir()
    for name, frame in splits.items():
        frame.to_csv(output / 'splits' / (name + '.csv'), index=False)
    if attack_profile is not None:
        checkpoint['attack_test_input'] = {'sha256': sha256_file(output / 'splits/test.csv'),
                                           'metadata': corruptions['test']}
    (output / 'config.resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    model = build_condition_model(checkpoint, device)
    batch_size = int(training.get('batch_size', 64))
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(datasets['validation'], batch_size=batch_size)
    # Reference eligibility is determined by conditions and observation masks.
    # A zero loss with no evaluable targets must not pass as a fitted model.
    model.eval()
    eligibility = {}
    with torch.no_grad():
        for name in ('train', 'validation'):
            count = 0
            for batch in DataLoader(datasets[name], batch_size=batch_size):
                diagnostic = model.diagnostics(**{key: batch[key].to(device)
                    for key in ('x', 'mask', 'conditions', 'condition_mask')})
                count += int(diagnostic['evaluated'].sum())
            if count == 0:
                raise ValueError('No evaluable condition-alignment reports in ' + name +
                                 '; inspect condition availability, reference radius and min_references')
            eligibility[name] = count
    manifest['condition_evaluable_reports'] = eligibility
    dump_json(output / 'split_manifest.json', manifest)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get('learning_rate', .001)),
                                 weight_decay=float(training.get('weight_decay', 0)))
    epochs, patience = int(training.get('epochs', 20)), int(training.get('patience', 5))
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    best, bad, history, start = float('inf'), 0, [], time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite leave-one-source-out Huber loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get('gradient_clip', 1)))
            optimizer.step()
            total += float(loss.detach()) * len(batch['x'])
            count += len(batch['x'])
        model.eval()
        with torch.no_grad():
            val_sum = sum(float(_loss(model, batch, device)) * len(batch['x']) for batch in validation)
        val = val_sum / len(datasets['validation'])
        if not np.isfinite(val):
            raise FloatingPointError('Nonfinite validation loss')
        record = {'epoch': epoch + 1, 'train_loss': total / count, 'validation_loss': val}
        history.append(record)
        print(record, flush=True)
        if val < best:
            best, bad = val, 0
            checkpoint['state_dict'] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint['optimizer_state_dict'] = copy.deepcopy(optimizer.state_dict())
            checkpoint['best_epoch'] = epoch + 1
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(checkpoint['state_dict'])
    elapsed = time.perf_counter() - start
    scores = score_condition_windows(model, datasets['calibration'], device, batch_size)
    diagnostics = list(KEYS) + ['error', 'quality', 'dispersion', 'score', 'evaluated', 'reference_count']
    registry = splits['calibration'].merge(scores[diagnostics], on=list(KEYS), how='left', validate='one_to_one')
    registry = plan_calibration_registry(registry, window)
    calibration_config = dict(method.get('calibration', {}))
    calibration_config.setdefault('alpha', config.get('calibration', {}).get('alpha', .05))
    calibration, audit = fit_label_trim(registry[registry.calibration_planned], calibration_config)
    calibration['window_schedule'] = window
    checkpoint['calibration'] = calibration
    checkpoint['threshold'] = calibration['threshold']
    torch.save(checkpoint, output / 'checkpoint.pt')
    registry.to_csv(output / 'calibration_scores.csv', index=False)
    audit.to_csv(output / 'calibration_audit.csv', index=False)
    dump_json(output / 'calibration.json', calibration)
    policy = config.get('protocol_feasibility', {})
    if policy.get('enabled', False) and (
            calibration['retained'] < int(policy.get('min_retained_calibration_units', 19))
            or not calibration['alarms_attainable']):
        raise ValueError('Fitted calibration failed protocol feasibility: retained=%d, alarms_attainable=%s. '
                         'Inspect calibration_scores.csv before treating this as a usable benchmark.' %
                         (calibration['retained'], calibration['alarms_attainable']))
    dump_json(output / 'history.json', history)
    dump_json(output / 'runtime.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'device': str(device), 'train_seconds': elapsed, 'seed': seed,
        'parameter_count': sum(p.numel() for p in model.parameters())})
    if evaluate_test:
        from .experiment import evaluate_checkpoint
        return evaluate_checkpoint(output / 'checkpoint.pt', output / 'splits/test.csv', output, str(device))
    return {'checkpoint': str(output / 'checkpoint.pt'), 'threshold': calibration['threshold'],
            'best_epoch': checkpoint['best_epoch']}


def evaluate_condition_checkpoint(checkpoint, model, reports, output, device, metrics=True):
    from .experiment import dump_json, seed_everything
    from .metrics import evaluate_reports
    from .evaluation import evaluate_references, evaluate_queries, _rank
    seed_everything(checkpoint['seed'], int(checkpoint['config'].get('training', {}).get('threads', 1)))
    raw = read_table(reports)
    if feature_columns(raw) != checkpoint['scaler']['columns']:
        raise ValueError('Inference feature schema differs from training')
    start = time.perf_counter()
    dataset = make_condition_windows(raw, checkpoint, allow_empty=True)
    predictions = score_condition_windows(model, dataset, device,
        int(checkpoint['config'].get('training', {}).get('batch_size', 64)))
    predictions['p_value'] = label_trim_pvalues(predictions.score, checkpoint['calibration'])
    predictions['prediction'] = label_trim_predictions(predictions.score, checkpoint['calibration']).astype(int)
    scaler = Standardizer.from_dict(checkpoint['scaler'])
    condition_scaler = ConditionScaler.from_dict(checkpoint['condition_scaler'])
    config = checkpoint['config']
    predictions, estimates = apply_reputation_and_estimate(model, predictions, raw, scaler, condition_scaler,
        device, config.get('method', {}), config.get('evaluation', {}).get('queries_path'))
    elapsed = time.perf_counter() - start
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    estimates.to_csv(output / 'native_environment.csv', index=False)
    predictions[list(KEYS) + ['error', 'quality', 'dispersion', 'score', 'p_value', 'prediction',
                             'reputation_prior', 'reputation']].to_csv(output / 'reputation.csv', index=False)
    availability = raw.merge(predictions[list(KEYS)], on=list(KEYS), how='left', indicator=True, validate='one_to_one')
    unassessed = availability[availability['_merge'] == 'left_only'].drop(columns='_merge')
    unassessed.to_csv(output / 'unassessed_reports.csv', index=False)
    result = {'algorithm': 'condition_alignment', 'input_reports': len(raw), 'scored_reports': len(predictions),
              'warmup_or_ineligible_reports': len(unassessed), 'scoring_coverage': len(predictions) / len(raw),
              'scoring_seconds': elapsed, 'input_sha256': sha256_file(reports),
              'calibration_summary': {key: checkpoint['calibration'][key]
                  for key in ('n', 'm', 'removed', 'retained', 'min_p', 'alarms_attainable')}}
    if metrics:
        result.update(evaluate_reports(predictions, checkpoint['threshold']))
        result['reference_evaluation'] = evaluate_references(predictions, checkpoint['threshold'], output)
        evaluation = config.get('evaluation', {})
        if evaluation.get('queries_path'):
            result['query_evaluation'] = evaluate_queries(predictions, checkpoint['threshold'],
                evaluation['queries_path'], output, evaluation)
        result['native_environment'] = environment_metrics(estimates, len(scaler.columns))
        truth = ['truth_' + name[len('value_'):] for name in scaler.columns]
        if all(name in predictions for name in truth):
            actual_error = np.abs(predictions[scaler.columns].to_numpy(float) - predictions[truth].to_numpy(float)).mean(-1)
            result['quality_error_spearman'] = _rank(-predictions.quality, actual_error)
        if 'attack_id' in raw:
            from .attack_evaluation import evaluate_attacks
            restored = restore_report_values(raw)
            seed_everything(checkpoint['seed'], int(config.get('training', {}).get('threads', 1)))
            baseline_predictions = score_condition_windows(model,
                make_condition_windows(restored, checkpoint, allow_empty=True), device,
                int(config.get('training', {}).get('batch_size', 64)))
            baseline_predictions['p_value'] = label_trim_pvalues(baseline_predictions.score, checkpoint['calibration'])
            baseline_predictions['prediction'] = label_trim_predictions(
                baseline_predictions.score, checkpoint['calibration']).astype(int)
            baseline_predictions, baseline_estimates = apply_reputation_and_estimate(model,
                baseline_predictions, restored, scaler, condition_scaler, device, config.get('method', {}),
                config.get('evaluation', {}).get('queries_path'))
            baseline_predictions.to_csv(output / 'baseline_predictions.csv', index=False)
            baseline_estimates.to_csv(output / 'baseline_native_environment.csv', index=False)
            result['attack_evaluation'] = evaluate_attacks(raw, predictions, checkpoint['threshold'],
                output, config, baseline_predictions=baseline_predictions, native_estimates=estimates,
                baseline_native_estimates=baseline_estimates,
                attempt_metadata=matching_attack_metadata(checkpoint, reports))
        dump_json(output / 'metrics.json', result)
    else:
        dump_json(output / 'inference.json', result)
    return result
