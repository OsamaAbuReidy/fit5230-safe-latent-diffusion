"""Run the self-contained Milestone 2 latent-versus-Multihead comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "data" / "public" / "milestone2_pooled_latents.npz"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "milestone2_bundle_results.json"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": int(y_true.size),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "harmful_recall": float(tp / (tp + fn)) if tp + fn else None,
        "benign_false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def make_model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=4000,
            random_state=5230,
        ),
    )


def run_task(
    features: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    teacher_scores: np.ndarray,
    positive_label: str,
    include_labels: tuple[str, ...],
) -> dict:
    keep = np.isin(labels, include_labels)
    x = features[keep]
    y = (labels[keep] == positive_label).astype(int)
    split = splits[keep]
    teacher = teacher_scores[keep]
    train = split == "train"
    validation = split == "validation"
    test = split == "test"

    search = []
    for c_value in C_GRID:
        candidate = make_model(c_value)
        candidate.fit(x[train], y[train])
        predicted = candidate.predict(x[validation])
        search.append(
            {
                "c": c_value,
                "validation_macro_f1": float(
                    f1_score(y[validation], predicted, average="macro")
                ),
            }
        )
    best = max(search, key=lambda item: (item["validation_macro_f1"], -item["c"]))
    model = make_model(float(best["c"]))
    development = train | validation
    model.fit(x[development], y[development])
    student_scores = model.predict_proba(x[test])[:, 1]

    return {
        "positive_label": positive_label,
        "split_counts": {
            name: int((split == name).sum()) for name in ("train", "validation", "test")
        },
        "selected_c": float(best["c"]),
        "validation_search": search,
        "pooled_latent_probe": metrics(y[test], student_scores),
        "multihead": metrics(y[test], teacher[test]),
    }


def main() -> int:
    args = parse_args()
    with np.load(args.bundle, allow_pickle=False) as bundle:
        features = bundle["features"].astype(np.float32)
        labels = bundle["human_label"].astype(str)
        splits = bundle["split"].astype(str)
        score_any = bundle["score_any"].astype(np.float32)
        score_violence = bundle["score_violence_extended"].astype(np.float32)

    result = {
        "schema_version": 1,
        "experiment": "milestone2_self_contained_comparison",
        "selection_protocol": "select C on validation macro-F1; refit train+validation; test once",
        "any_harm": run_task(
            features,
            np.where(labels == "benign", "benign", "harmful"),
            splits,
            score_any,
            positive_label="harmful",
            include_labels=("benign", "harmful"),
        ),
        "violence_gore": run_task(
            features,
            labels,
            splits,
            score_violence,
            positive_label="violence_gore",
            include_labels=("benign", "violence_gore"),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
