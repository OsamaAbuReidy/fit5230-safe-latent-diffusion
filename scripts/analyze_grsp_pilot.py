"""Exploratory, group-aware analysis of GRSP pilot residual features."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# Small pilot matrices do not benefit from a large BLAS thread pool, and some
# Windows Python builds can stall while initialising one.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = REPOSITORY_ROOT / "data" / "raw" / "grsp_pilot" / "features" / "grsp_pilot_02"
DEFAULT_ANNOTATIONS = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "grsp_pilot"
    / "manifests"
    / "grsp_pilot_02_annotations.csv"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "outputs" / "reports" / "grsp_pilot_02_analysis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--early-steps", type=int, default=6)
    return parser.parse_args()


def feature_file(root: Path, prompt_id: str, seed: int) -> Path:
    matches = sorted((root / prompt_id).glob(f"seed_{seed}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one feature file for {prompt_id}, seed {seed}; found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    args = parse_args()
    with args.annotations.open("r", encoding="utf-8", newline="") as handle:
        annotations = list(csv.DictReader(handle))

    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    prompt_rule: list[int] = []
    records: list[dict[str, object]] = []

    for annotation in annotations:
        prompt_id = annotation["prompt_id"]
        seed = int(annotation["seed"])
        archive = np.load(feature_file(args.feature_root, prompt_id, seed), allow_pickle=False)
        statistics = archive["statistics"][:, 0, :].astype(np.float32)
        if statistics.shape[0] < args.early_steps:
            raise RuntimeError(f"{prompt_id}, seed {seed} has too few sampling steps")

        features.append(statistics[: args.early_steps].reshape(-1))
        labels.append(1 if annotation["output_label"] == "unsafe" else 0)
        groups.append(prompt_id.rsplit("_", 1)[1])
        prompt_rule.append(1 if prompt_id.startswith("direct_") else 0)
        records.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "output_label": annotation["output_label"],
                "early_abs_mean": float(statistics[: args.early_steps, 2].mean()),
                "early_rms": float(statistics[: args.early_steps, 3].mean()),
                "step_0_rms": float(statistics[0, 3]),
                "step_5_rms": float(statistics[min(5, statistics.shape[0] - 1), 3]),
            }
        )

    x = np.asarray(features)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    predictions = np.zeros_like(y)
    probabilities = np.zeros(len(y), dtype=np.float64)
    folds: list[dict[str, object]] = []

    for group in sorted(set(groups)):
        test = group_array == group
        train = ~test
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        folds.append(
            {
                "held_out_pair": f"pair_{group}",
                "accuracy": float(accuracy_score(y[test], predictions[test])),
                "truth": y[test].tolist(),
                "predictions": predictions[test].tolist(),
            }
        )

    prompt_predictions = np.asarray(prompt_rule)
    report = {
        "warning": (
            "Exploratory pilot only. Labels are strongly correlated with prompt class, "
            "and the only within-prompt output variation contains four samples."
        ),
        "sample_count": len(y),
        "unsafe_count": int(y.sum()),
        "safe_count": int((1 - y).sum()),
        "feature_definition": f"first {args.early_steps} steps x 7 residual statistics",
        "leave_one_pair_out": {
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "confusion_matrix_tn_fp_fn_tp": [
                int(value) for value in confusion_matrix(y, predictions).ravel()
            ],
            "folds": folds,
        },
        "prompt_class_rule": {
            "definition": "predict unsafe for every direct prompt and safe for every benign prompt",
            "accuracy": float(accuracy_score(y, prompt_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prompt_predictions)),
            "confusion_matrix_tn_fp_fn_tp": [
                int(value) for value in confusion_matrix(y, prompt_predictions).ravel()
            ],
        },
        "direct_006": [record for record in records if record["prompt_id"] == "direct_006"],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
