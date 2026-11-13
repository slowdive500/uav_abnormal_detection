"""Shared all-split attack preparation and clean-report counterfactuals."""
import json
from pathlib import Path
import numpy as np
from .data.schema import feature_columns
from .protocol import SPLITS, Standardizer, sha256_file
from .corruption import inject_faults


def annotate_attack_windows(frame, window):
    """Mark source-local attack history on the model's task clock, without labels."""
    if 'injection_label' not in frame:
        return frame
    out = frame.copy()
    out['window_contains_attack'] = False
    size = int(window.get('window_size', 16))
    gap = window.get('max_gap_seconds')
    for _, group in out.groupby(['sequence_id', 'task_id'], sort=True):
        times = np.sort(group.timestamp.unique())
        breaks = np.where(np.diff(times) > float(gap))[0] + 1 if gap is not None else []
        for segment in np.split(times, breaks):
            current = group[group.timestamp.isin(segment)]
            for _, source in current.groupby('source_id', sort=True):
                attacked = source.set_index('timestamp').injection_label.reindex(segment, fill_value=0)
                history = attacked.rolling(size, min_periods=1).max().astype(bool)
                out.loc[source.index, 'window_contains_attack'] = source.timestamp.map(history).to_numpy()
    out['evaluation_phase'] = np.where(out.injection_label.eq(1), 'active',
        np.where(out.window_contains_attack, 'recovery', 'background'))
    return out


def prepare_corrupted_splits(splits, config, seed):
    """Freeze the attack specification on training, then corrupt every split."""
    corruption = config.get('corruption', {})
    attack_mode = config.get('attack_mode', 'configured')
    if attack_mode not in ('configured', 'none'):
        raise ValueError('attack_mode must be configured or none')
    if attack_mode == 'none':
        result, metadata = {}, {}
        for name in SPLITS:
            original = splits[name]
            if 'injection_label' in original and original.injection_label.eq(1).any():
                raise ValueError('No-attack runs require original reports without prior injected attacks')
            frame = original.copy()
            frame['original_label'] = original.label
            for column in feature_columns(original):
                frame['baseline_' + column[len('value_'):]] = original[column]
            # Match the attacked benchmark's background-label convention without
            # removing, denoising or certifying any naturally occurring faults.
            frame.loc[frame.label.eq(-1), 'label'] = 0
            frame['injection_label'] = 0
            frame['fault_type'], frame['fault_event_id'] = 'none', ''
            result[name] = annotate_attack_windows(frame, config.get('window', {}))
            metadata[name] = {
                'protocol': 'none', 'attack_mode': 'none', 'injected_reports': 0,
                'affected_sources': [], 'events': [], 'report_values_unchanged': True,
                'baseline_assumption': 'Unlabelled baseline assumed normal only for injected-fault '
                    'evaluation; natural faults remain unverified and unchanged',
            }
        if config.get('protocol_feasibility', {}).get('enabled', False):
            from .preflight import validate_experiment_feasibility
            feasibility = validate_experiment_feasibility(result, config)
            for name in SPLITS:
                metadata[name]['preflight'] = feasibility[name]
        return result, metadata, None
    targeted = [name for name in SPLITS
                if corruption.get(name, {}).get('protocol') == 'targeted_report_attack']
    profile = None
    if targeted:
        if set(targeted) != set(SPLITS):
            raise ValueError('targeted_report_attack must be configured in all four splits')
        common = ('constraints', 'source_fraction', 'source_selection_seed',
                  'coalition_sizes', 'max_local_fraction', 'kinds')
        specification = corruption['train']
        if not specification.get('kinds'):
            raise ValueError('The all-split targeted protocol requires explicit attack kinds')
        for name in SPLITS[1:]:
            if any(corruption[name].get(key) != specification.get(key) for key in common):
                raise ValueError('Attack constraints, kinds and compromised sources must be shared across splits')
        from .attacks import fit_attack_profile
        profile = fit_attack_profile(splits['train'], specification)
    initial = Standardizer().fit(splits['train'])
    result, metadata = dict(splits), {}
    for i, name in enumerate(SPLITS):
        if corruption.get(name) is not None:
            result[name], metadata[name] = inject_faults(result[name], corruption[name], initial.scale,
                seed + 1009 * (i + 1), attack_profile=profile)
            result[name] = annotate_attack_windows(result[name], config.get('window', {}))
    if profile is not None and not result['train'].injection_label.eq(1).any():
        raise ValueError('No realized training attacks under the targeted protocol; inspect temporal coverage, '
                         'compromised-source participation and feasibility instead of silently training on clean reports')
    if config.get('protocol_feasibility', {}).get('enabled', False):
        from .preflight import validate_experiment_feasibility
        feasibility = validate_experiment_feasibility(result, config)
        for name in SPLITS:
            metadata.setdefault(name, {})['preflight'] = feasibility[name]
    return result, metadata, profile


def write_attack_audits(splits, config, output, profile, metadata):
    if profile is None:
        return None
    from .attack_evaluation import evaluate_attacks
    output = Path(output)
    (output / 'attack_profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False))
    audits = {}
    for name in SPLITS:
        folder = output / 'attack_audits' / name
        folder.mkdir(parents=True, exist_ok=True)
        audits[name] = evaluate_attacks(splits[name], config=config, output=folder,
                                        attempt_metadata=metadata[name])
        (folder / 'metrics.json').write_text(json.dumps(audits[name], indent=2, allow_nan=False))
    return audits


def matching_attack_metadata(checkpoint, reports):
    """Use zero-report attempt records only for the exact saved test input."""
    audit = checkpoint.get('attack_test_input', {})
    return audit.get('metadata') if audit.get('sha256') == sha256_file(reports) else None


def restore_report_values(frame):
    """Remove only the generated report tampering; retain natural baseline faults."""
    restored = frame.copy()
    for name in feature_columns(frame):
        baseline = 'baseline_' + name[len('value_'):]
        if baseline not in frame:
            raise ValueError('Missing counterfactual baseline column: ' + baseline)
        restored[name] = frame[baseline]
    if 'original_label' in frame:
        restored['label'] = frame.original_label
    restored['injection_label'] = 0
    restored['fault_type'], restored['fault_event_id'] = 'none', ''
    restored['window_contains_attack'] = False
    restored['evaluation_phase'] = 'counterfactual'
    return restored
