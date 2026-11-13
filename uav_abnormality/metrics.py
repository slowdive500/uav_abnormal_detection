"""Raw report-level metrics. Never use test-tuned thresholds or point adjustment."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def binary_metrics(labels, scores, threshold, decisions=None):
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    known = np.isin(labels, [0, 1]) & np.isfinite(scores)
    y, score = labels[known].astype(int), scores[known]
    result = {"scored_reports": int(len(labels)), "labelled_reports": int(len(y)),
              "unknown_or_invalid_reports": int((~known).sum()), "threshold": float(threshold) if threshold is not None else None,
              "point_adjustment": False}
    if not len(y):
        result["status"] = "no_ground_truth_labels"
        return result
    pred = np.asarray(decisions)[known].astype(bool) if decisions is not None else (
        score > threshold if threshold is not None else np.zeros(len(score), dtype=bool))
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    both = len(np.unique(y)) == 2
    has_positives = bool((y == 1).any())
    result.update({"status": "ok" if both else "single_label_class", "precision": float(precision),
                   "recall": float(recall) if has_positives else None,
                   "f1": float(f1) if has_positives else None,
                   "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                   "auroc": float(roc_auc_score(y, score)) if both else None,
                   "average_precision": float(average_precision_score(y, score)) if both else None,
                   "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return result

def evaluate_reports(frame, threshold):
    def measure(group, label="label"):
        return binary_metrics(group[label], group.score, threshold,
                              group.prediction if "prediction" in group else None)
    result = measure(frame)
    per_source = {str(source): measure(group)
                  for source, group in frame.groupby("source_id", sort=True)}
    result["per_source"] = per_source
    valid = [item["f1"] for item in per_source.values() if item.get("f1") is not None]
    result["macro_source_f1"] = float(np.mean(valid)) if valid else None
    if "injection_label" in frame:
        result["injected_fault_detection"] = measure(frame, "injection_label")
        result["injected_fault_detection"]["label_semantics"] = "additional_injected_fault_only; original natural faults unverified"
    if "fault_type" in frame:
        # Same clean negatives for every fault family, with the fixed threshold.
        result["per_fault"] = {str(kind): measure(
            frame.loc[(frame.fault_type == kind) | ((frame.fault_type == "none") & (frame.label == 0))])
            for kind in sorted(set(frame.fault_type) - {"none"})}
    return result

def calibrate_threshold(scores, alpha):
    values = np.asarray(scores, float)
    values = values[np.isfinite(values)]
    if not len(values) or not 0 < alpha < 1:
        raise ValueError("Calibration requires finite scores and alpha in (0,1)")
    # Empirical validation quantile, not a claimed finite-sample guarantee.
    return float(np.quantile(values, 1 - alpha, method="higher"))
