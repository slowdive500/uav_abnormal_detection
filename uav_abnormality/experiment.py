"""Unified unsupervised train/calibrate/test and causal inference runner."""
import copy
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from .data.schema import read_table, coverage_report, feature_columns
from .protocol import split_table, Standardizer, Windows, sha256_file, SPLITS
from .attack_pipeline import (prepare_corrupted_splits, write_attack_audits,
                              restore_report_values, matching_attack_metadata)
from .metrics import calibrate_threshold, evaluate_reports
from .models import build_model

def dump_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device: cpu or fix NVIDIA driver")
    return torch.device(name)

def make_windows(frame, checkpoint):
    return Windows(frame, checkpoint["sources"], Standardizer.from_dict(checkpoint["scaler"]), **checkpoint["window"])


def has_observation_gap(dataset, gid, previous_time, current_time):
    """Stride skips scored endpoints, not observations; inspect the raw clock."""
    limit = dataset.max_gap_seconds
    if limit is None or previous_time is None:
        return False
    times = dataset.groups[gid]["times"]
    left = int(np.searchsorted(times, previous_time, side="left"))
    right = int(np.searchsorted(times, current_time, side="right"))
    return bool(np.any(np.diff(times[left:right]) > limit))

@torch.no_grad()
def score_windows(model, dataset, device, batch_size=64, reputation_scope="task"):
    if reputation_scope not in ("task", "source"):
        raise ValueError("reputation_scope must be task or source")
    if reputation_scope == "source" and hasattr(model, "_new_state"):
        return score_source_history(model, dataset, device)
    model.eval()
    results, current_group = [], None
    # No batch may cross a sequence/task boundary for stateful PRBTD.
    indices_by_group = {}
    for index, (gid, _) in enumerate(dataset.index):
        indices_by_group.setdefault(gid, []).append(index)
    for gid, indices in indices_by_group.items():
        # Separate disjoint time segments before batching a stateful model.
        segments, segment = [], []
        max_gap = getattr(dataset, "max_gap_seconds", None)
        previous_time = None
        for index in indices:
            stamp = dataset.metadata(index)[1]
            if has_observation_gap(dataset, gid, previous_time, stamp):
                segments.append(segment)
                segment = []
            segment.append(index)
            previous_time = stamp
        if segment:
            segments.append(segment)
        chunks = []
        for segment in segments:
            for start in range(0, len(segment), batch_size):
                chunks.append((start == 0, segment[start:start + batch_size]))
        for reset, selected in chunks:
            if reset and hasattr(model, "reset_state"):
                model.reset_state()
            batch = [dataset[i] for i in selected]
            x = torch.stack([b["x"] for b in batch]).to(device)
            mask = torch.stack([b["mask"] for b in batch]).to(device)
            method = getattr(model, "stream_score", model.score)
            scores = method(x, mask).detach().cpu().numpy()
            diagnostics = {key: value.detach().cpu().numpy() for key, value in
                           (getattr(model, "stream_diagnostics", None) or {}).items()}
            if scores.shape != (len(selected), len(dataset.sources)):
                raise ValueError("Model must return score[B,S]")
            observed = mask[:, -1].any(-1).cpu().numpy()
            if not np.isfinite(scores[observed]).all():
                raise FloatingPointError("Non-finite anomaly scores")
            for b, index in enumerate(selected):
                _, _, row_ids = dataset.metadata(index)
                for source, row_id in enumerate(row_ids):
                    if row_id < 0 or not observed[b, source]:
                        continue
                    row = dataset.frame.iloc[row_id].to_dict()
                    row["score"] = float(scores[b, source])
                    for key, values in diagnostics.items():
                        row[key] = float(values[b, source])
                    results.append(row)
    return pd.DataFrame(results)


@torch.no_grad()
def score_source_history(model, dataset, device):
    """Persistent PRBTD source reputation with task-local implication buffers.

    All tasks at a timestamp read the same previous reputation. If a source
    contributes to several tasks, average its proposed updates once per round.
    Missing/abstained reports never update reputation. Independent sequences
    and train/calibration/test calls start from the prior again.
    """
    model.eval()
    model.reset_state()
    rounds = {}
    for index in range(len(dataset)):
        (sequence, task), stamp, _ = dataset.metadata(index)
        rounds.setdefault((sequence, stamp), []).append((task, index))
    sequence = None
    global_reputation, states, last_time, results = None, {}, {}, []
    for (seq, stamp), entries in sorted(rounds.items()):
        if sequence != seq:
            sequence, states, last_time = seq, {}, {}
            global_reputation = torch.full((1, len(dataset.sources)), .5, device=device)
        proposals, active_masks, round_rows = [], [], []
        for task, index in sorted(entries):
            item = dataset[index]
            x = item["x"].unsqueeze(0).to(device)
            mask = item["mask"].unsqueeze(0).to(device)
            gid = dataset.index[index][0]
            if task not in states or has_observation_gap(dataset, gid, last_time.get(task), stamp):
                states[task] = model._new_state(x, 1)
            state = states[task]
            state["reputation"] = global_reputation.clone()
            model._stream_state = state
            scores = model.stream_score(x, mask)[0].cpu().numpy()
            active = mask[:, -1].any(-1)
            if not np.isfinite(scores[active[0].cpu().numpy()]).all():
                raise FloatingPointError("Non-finite anomaly scores")
            proposals.append(state["reputation"].clone())
            active_masks.append(active)
            _, _, row_ids = dataset.metadata(index)
            for source, row_id in enumerate(row_ids):
                if row_id < 0 or not active[0, source]:
                    continue
                row = dataset.frame.iloc[row_id].to_dict()
                row.update(score=float(scores[source]), quality_native=float(1 - scores[source]),
                           reputation_prior=float(global_reputation[0, source]), reputation_scope="source")
                round_rows.append((source, row))
            last_time[task] = stamp
        masks = torch.stack(active_masks)
        counts = masks.sum(0)
        mean = (torch.stack(proposals) * masks).sum(0) / counts.clamp_min(1)
        global_reputation = torch.where(counts > 0, mean, global_reputation)
        for source, row in round_rows:
            row["reputation"] = float(global_reputation[0, source])
            results.append(row)
    if model._stream_state is not None:
        model._stream_state["reputation"] = global_reputation
    return pd.DataFrame(results)

@torch.no_grad()
def validation_loss(model, loader, device, seed):
    model.eval()
    # Stable validation noise for the diffusion objective, without perturbing
    # training RNG state or using any anomaly label.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        total, count = 0., 0
        for batch in loader:
            loss = model.loss(batch["x"].to(device), batch["mask"].to(device))
            total += float(loss) * len(batch["x"])
            count += len(batch["x"])
    return total / count

def train_experiment(config, output, evaluate_test=True):
    if config.get("algorithm") == "condition_alignment":
        from .condition_experiment import train_condition_experiment
        return train_condition_experiment(config, output, evaluate_test)
    config = copy.deepcopy(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is nonempty; choose a new run directory: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    training = config.get("training", {})
    seed_everything(seed, int(training.get("threads", 1)))
    device = device_for(training.get("device", "auto"))
    raw = read_table(config["data"]["path"])
    splits, manifest = split_table(raw, config.get("split", {}))
    splits, corruptions, attack_profile = prepare_corrupted_splits(splits, config, seed)
    scaler = Standardizer().fit(splits["train"])
    sources = sorted(splits["train"].source_id.unique().tolist())
    window = dict(config.get("window", {}))
    window.setdefault("window_size", 16)
    window.setdefault("min_sources", 2)
    window.setdefault("stride", 1)
    checkpoint = {"format_version": 1, "algorithm": config["algorithm"], "model_config": config.get("model", {}),
                  "sources": sources, "scaler": scaler.to_dict(), "window": window, "seed": seed,
                  "input_sha256": sha256_file(config["data"]["path"]), "config": config}
    needed_splits = SPLITS if evaluate_test else SPLITS[:-1]
    datasets = {name: make_windows(splits[name], checkpoint) for name in needed_splits}
    manifest["windows"] = {k: len(v) for k, v in datasets.items()}
    manifest["source_vocabulary"] = sources
    manifest["coverage"] = {k: coverage_report(v) for k, v in splits.items()}
    manifest["corruptions"] = corruptions
    if attack_profile is not None:
        manifest['attack_audits'] = write_attack_audits(splits, config, output, attack_profile, corruptions)
    dump_json(output / "split_manifest.json", manifest)
    (output / "splits").mkdir()
    for name, frame in splits.items():
        frame.to_csv(output / "splits" / (name + ".csv"), index=False)
    if attack_profile is not None:
        checkpoint['attack_test_input'] = {'sha256': sha256_file(output / 'splits/test.csv'),
                                           'metadata': corruptions['test']}
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = build_model(config["algorithm"], len(sources), len(scaler.columns), window["window_size"], config.get("model")).to(device)
    batch_size = int(training.get("batch_size", 32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False)
    # Original S4 kernels mark sensitive state parameters with _optim settings.
    groups = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        options = dict(getattr(parameter, "_optim", {}))
        key = tuple(sorted(options.items()))
        groups.setdefault(key, {"params": [], **options})["params"].append(parameter)
    optimizer = torch.optim.Adam(list(groups.values()), lr=float(training.get("learning_rate", 1e-3)),
                                 weight_decay=float(training.get("weight_decay", 0.)))
    epochs, patience = int(training.get("epochs", 30)), int(training.get("patience", 5))
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    history, best, bad, start = [], float("inf"), 0, time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total, count = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch["x"].to(device), batch["mask"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.)))
            optimizer.step()
            total += float(loss.detach()) * len(batch["x"])
            count += len(batch["x"])
        val = validation_loss(model, val_loader, device, seed + 17)
        if not np.isfinite(val):
            raise FloatingPointError("Non-finite validation loss")
        record = {"epoch": epoch + 1, "train_loss": total / count, "validation_loss": val}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val < best:
            best, bad = val, 0
            checkpoint["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint["best_epoch"] = epoch + 1
            # Optimizer retained for research resumption through Python API.
            checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict())
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(checkpoint["state_dict"])
    train_seconds = time.perf_counter() - start
    scoring = config.get("scoring", {})
    calibration = score_windows(model, datasets["calibration"], device, batch_size,
                                scoring.get("reputation_scope", "task"))
    alpha = float(config.get("calibration", {}).get("alpha", .05))
    threshold = calibrate_threshold(calibration.score, alpha)
    checkpoint["threshold"] = threshold
    checkpoint["calibration"] = {"method": "unlabelled_empirical_quantile", "alpha": alpha,
                                 "n_scores": len(calibration), "guarantee": "none under contamination or dependent windows"}
    torch.save(checkpoint, output / "checkpoint.pt")
    calibration.to_csv(output / "calibration_scores.csv", index=False)
    dump_json(output / "calibration.json", dict(checkpoint["calibration"], threshold=threshold))
    dump_json(output / "history.json", history)
    dump_json(output / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
              "device": str(device), "cuda_available": torch.cuda.is_available(), "train_seconds": train_seconds,
              "parameter_count": sum(p.numel() for p in model.parameters()), "seed": seed})
    if evaluate_test:
        return evaluate_checkpoint(output / "checkpoint.pt", output / "splits/test.csv", output, str(device))
    return {"checkpoint": str(output / "checkpoint.pt"), "threshold": threshold, "best_epoch": checkpoint["best_epoch"]}

def load_checkpoint(path, device):
    # Only locally produced/trusted checkpoints: torch serialization uses pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("algorithm") == "condition_alignment":
        if checkpoint.get("format_version") != 2:
            raise ValueError("Unsupported condition-alignment checkpoint format")
        from .condition_experiment import build_condition_model
        model = build_condition_model(checkpoint, device)
        model.load_state_dict(checkpoint["state_dict"])
        return checkpoint, model
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = build_model(checkpoint["algorithm"], len(checkpoint["sources"]), len(checkpoint["scaler"]["columns"]),
                        checkpoint["window"]["window_size"], checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model

def evaluate_checkpoint(checkpoint_path, reports, output, device="auto", metrics=True):
    device = device_for(device)
    checkpoint, model = load_checkpoint(checkpoint_path, device)
    if checkpoint["algorithm"] == "condition_alignment":
        from .condition_experiment import evaluate_condition_checkpoint
        return evaluate_condition_checkpoint(checkpoint, model, reports, output, device, metrics)
    seed_everything(checkpoint["seed"], int(checkpoint["config"].get("training", {}).get("threads", 1)))
    frame = read_table(reports)
    if feature_columns(frame) != checkpoint["scaler"]["columns"]:
        raise ValueError("Inference feature schema differs from trained schema")
    dataset = make_windows(frame, checkpoint)
    start = time.perf_counter()
    predictions = score_windows(model, dataset, device, int(checkpoint["config"].get("training", {}).get("batch_size", 32)),
                                checkpoint["config"].get("scoring", {}).get("reputation_scope", "task"))
    elapsed = time.perf_counter() - start
    predictions["prediction"] = (predictions.score > checkpoint["threshold"]).astype(int)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    result = {"input_reports": len(frame), "scored_reports": len(predictions), "warmup_or_ineligible_reports": len(frame) - len(predictions),
              "scoring_seconds": elapsed, "algorithm": checkpoint["algorithm"], "input_sha256": sha256_file(reports)}
    if metrics:
        result.update(evaluate_reports(predictions, checkpoint["threshold"]))
        from .evaluation import evaluate_references, evaluate_queries
        result["reference_evaluation"] = evaluate_references(predictions, checkpoint["threshold"], output)
        evaluation = checkpoint["config"].get("evaluation", {})
        if evaluation.get("queries_path"):
            result["query_evaluation"] = evaluate_queries(predictions, checkpoint["threshold"],
                                                          evaluation["queries_path"], output, evaluation)
        if "reputation" in predictions:
            columns = ["sequence_id", "task_id", "timestamp", "source_id", "quality_native",
                       "reputation_prior", "reputation"]
            predictions[columns].to_csv(output / "reputation.csv", index=False)
        result["scoring_coverage"] = len(predictions) / len(frame)
        if 'attack_id' in frame:
            from .attack_evaluation import evaluate_attacks
            restored = restore_report_values(frame)
            seed_everything(checkpoint['seed'], int(checkpoint['config'].get('training', {}).get('threads', 1)))
            baseline_predictions = score_windows(model, make_windows(restored, checkpoint), device,
                int(checkpoint['config'].get('training', {}).get('batch_size', 32)),
                checkpoint['config'].get('scoring', {}).get('reputation_scope', 'task'))
            baseline_predictions['prediction'] = (baseline_predictions.score > checkpoint['threshold']).astype(int)
            baseline_predictions.to_csv(output / 'baseline_predictions.csv', index=False)
            result['attack_evaluation'] = evaluate_attacks(frame, predictions, checkpoint['threshold'],
                output, checkpoint['config'], baseline_predictions=baseline_predictions,
                attempt_metadata=matching_attack_metadata(checkpoint, reports))
        dump_json(output / "metrics.json", result)
    else:
        dump_json(output / "inference.json", result)
    return result
