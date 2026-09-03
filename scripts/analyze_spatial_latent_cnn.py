"""Bootstrap and error analysis for the spatial violence latent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "data" / "public" / "milestone2_pooled_latents.npz"
DEFAULT_SPATIAL = ROOT / "data" / "results" / "spatial_latent_cnn_violence_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "spatial_latent_cnn_violence_v1_analysis.json"
DEFAULT_ERRORS = ROOT / "data" / "results" / "spatial_latent_cnn_violence_v1_errors.csv"


def pooled_scores(
    features: np.ndarray, labels: np.ndarray, splits: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(labels, ["benign", "violence_gore"])
    x = features[keep]
    y = (labels[keep] == "violence_gore").astype(int)
    split = splits[keep]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.01, class_weight="balanced", max_iter=4000, random_state=5230
        ),
    )
    model.fit(x[split != "test"], y[split != "test"])
    return y[split == "test"], model.predict_proba(x[split == "test"])[:, 1]


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": round(float(low), 4), "high": round(float(high), 4)}


def bootstrap(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(truth == 0)
    positive = np.flatnonzero(truth == 1)
    auc = {name: np.empty(repeats) for name in scores}
    balanced = {name: np.empty(repeats) for name in scores}
    for iteration in range(repeats):
        indexes = np.concatenate(
            (
                rng.choice(negative, size=negative.size, replace=True),
                rng.choice(positive, size=positive.size, replace=True),
            )
        )
        sampled_truth = truth[indexes]
        for name, model_scores in scores.items():
            sampled_scores = model_scores[indexes]
            auc[name][iteration] = roc_auc_score(sampled_truth, sampled_scores)
            predicted = (sampled_scores >= thresholds[name]).astype(int)
            balanced[name][iteration] = balanced_accuracy_score(sampled_truth, predicted)
    result: dict[str, object] = {
        "method": "paired stratified percentile bootstrap",
        "repeats": repeats,
        "seed": seed,
        "models": {},
        "paired_differences": {},
    }
    for name in scores:
        result["models"][name] = {
            "roc_auc_95_ci": interval(auc[name]),
            "balanced_accuracy_95_ci": interval(balanced[name]),
        }
    for comparator in ("pooled_latent", "multihead"):
        result["paired_differences"][f"spatial_cnn_minus_{comparator}"] = {
            "roc_auc_95_ci": interval(auc["spatial_cnn"] - auc[comparator]),
            "balanced_accuracy_95_ci": interval(
                balanced["spatial_cnn"] - balanced[comparator]
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--spatial-results", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors-output", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=5230)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spatial = json.loads(args.spatial_results.read_text(encoding="utf-8"))
    with np.load(args.bundle, allow_pickle=False) as bundle:
        features = bundle["features"].astype(np.float32)
        sample_ids = bundle["sample_id"].astype(str)
        splits = bundle["split"].astype(str)
        labels = bundle["human_label"].astype(str)
        candidate_policy = bundle["candidate_policy"].astype(str)
        multihead_all = bundle["score_violence_extended"].astype(np.float32)

    keep = np.isin(labels, ["benign", "violence_gore"])
    test = keep & (splits == "test")
    test_ids = sample_ids[test]
    test_truth, pooled = pooled_scores(features, labels, splits)
    multihead = multihead_all[test]
    spatial_by_id = {
        row["sample_id"]: float(row["violence_score"])
        for row in spatial["test_predictions"]
    }
    if set(test_ids) != set(spatial_by_id):
        raise ValueError("Spatial predictions do not match the frozen violence test sample IDs")
    spatial_scores = np.asarray([spatial_by_id[sample_id] for sample_id in test_ids])
    spatial_threshold = float(spatial["test"]["threshold"])
    thresholds = {
        "pooled_latent": 0.5,
        "spatial_cnn": spatial_threshold,
        "multihead": 0.5,
    }
    scores = {
        "pooled_latent": pooled,
        "spatial_cnn": spatial_scores,
        "multihead": multihead,
    }
    boot = bootstrap(
        test_truth, scores, thresholds, args.bootstrap_repeats, args.seed
    )

    predictions = {
        name: (value >= thresholds[name]).astype(int) for name, value in scores.items()
    }
    error_mask = predictions["spatial_cnn"] != test_truth
    errors = pd.DataFrame(
        {
            "sample_id": test_ids[error_mask],
            "true_label": np.where(test_truth[error_mask] == 1, "violence_gore", "benign"),
            "cnn_error_type": np.where(test_truth[error_mask] == 1, "false_negative", "false_positive"),
            "candidate_policy": candidate_policy[test][error_mask],
            "spatial_cnn_score": spatial_scores[error_mask],
            "pooled_latent_score": pooled[error_mask],
            "multihead_score": multihead[error_mask],
            "pooled_correct": predictions["pooled_latent"][error_mask] == test_truth[error_mask],
            "multihead_correct": predictions["multihead"][error_mask] == test_truth[error_mask],
        }
    ).sort_values(["cnn_error_type", "spatial_cnn_score"], ascending=[True, False])
    args.errors_output.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors_output, index=False)

    paired = {}
    cnn_correct = predictions["spatial_cnn"] == test_truth
    for name in ("pooled_latent", "multihead"):
        other_correct = predictions[name] == test_truth
        paired[name] = {
            "both_correct": int((cnn_correct & other_correct).sum()),
            "cnn_only_correct": int((cnn_correct & ~other_correct).sum()),
            f"{name}_only_correct": int((~cnn_correct & other_correct).sum()),
            "both_wrong": int((~cnn_correct & ~other_correct).sum()),
        }

    report = {
        "schema_version": 1,
        "experiment": "spatial_cnn_violence_bootstrap_and_error_analysis",
        "test_sample_count": int(test_truth.size),
        "positive_count": int(test_truth.sum()),
        "thresholds": thresholds,
        "bootstrap": boot,
        "paired_correctness": paired,
        "spatial_cnn_errors": {
            "total": int(error_mask.sum()),
            "false_negatives": int(((test_truth == 1) & error_mask).sum()),
            "false_positives": int(((test_truth == 0) & error_mask).sum()),
            "csv": str(args.errors_output),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
