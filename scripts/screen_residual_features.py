"""Screen fixed residual representations with leave-one-semantic-pair-out CV."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data" / "raw" / "grsp_pilot" / "features" / "grsp_pilot_02"
ANNOTATIONS = ROOT / "data" / "raw" / "grsp_pilot" / "manifests" / "grsp_pilot_02_annotations.csv"
REPORT = ROOT / "outputs" / "reports" / "grsp_pilot_02_feature_screen.json"
EARLY_STEPS = 6


def locate(prompt_id: str, seed: int) -> Path:
    matches = list((FEATURE_ROOT / prompt_id).glob(f"seed_{seed}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive for {prompt_id}/{seed}, found {len(matches)}")
    return matches[0]


def representations(archive: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    statistics = archive["statistics"][:, 0].astype(np.float32)
    pooled = archive["pooled_residuals"][:, 0].astype(np.float32)
    early = pooled[:EARLY_STEPS]
    channel_mean = early.mean(axis=(2, 3))
    channel_std = early.std(axis=(2, 3))
    channel_rms = np.sqrt(np.mean(early**2, axis=(2, 3)))
    channel_moments = np.stack((channel_mean, channel_std, channel_rms), axis=-1)
    normalizer = np.linalg.norm(channel_moments.reshape(EARLY_STEPS, -1), axis=1)
    normalized = channel_moments / np.maximum(normalizer[:, None, None], 1e-8)
    return {
        "early_global_statistics": statistics[:EARLY_STEPS].reshape(-1),
        "early_channel_moments": channel_moments.reshape(-1),
        "early_normalized_channel_moments": normalized.reshape(-1),
        "early_pooled_spatial": early.reshape(-1),
        "full_global_statistics_upper_bound": statistics.reshape(-1),
    }


def evaluate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, use_pca: bool) -> dict[str, object]:
    predictions = np.zeros_like(y)
    probabilities = np.zeros(len(y), dtype=np.float64)
    folds: list[dict[str, object]] = []
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        steps: list[object] = [StandardScaler()]
        if use_pca:
            steps.append(PCA(n_components=min(12, int(train.sum()) - 1), random_state=0))
        steps.append(LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
        model = make_pipeline(*steps)
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        folds.append(
            {
                "held_out_pair": f"pair_{group}",
                "accuracy": float(accuracy_score(y[test], predictions[test])),
            }
        )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": [
            int(value) for value in confusion_matrix(y, predictions).ravel()
        ],
        "folds": folds,
    }


def main() -> int:
    with ANNOTATIONS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    feature_sets: dict[str, list[np.ndarray]] = {}
    output_labels: list[int] = []
    intent_labels: list[int] = []
    groups: list[str] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        seed = int(row["seed"])
        archive = np.load(locate(prompt_id, seed), allow_pickle=False)
        for name, values in representations(archive).items():
            feature_sets.setdefault(name, []).append(values)
        output_labels.append(1 if row["output_label"] == "unsafe" else 0)
        intent_labels.append(1 if prompt_id.startswith("direct_") else 0)
        groups.append(prompt_id.rsplit("_", 1)[1])

    y_output = np.asarray(output_labels)
    y_intent = np.asarray(intent_labels)
    group_array = np.asarray(groups)
    report: dict[str, object] = {
        "protocol": "leave one semantic pair out; fixed C=0.1; no seed-random split",
        "warning": "Exploratory feature screening on five semantic pairs; not a final estimate.",
        "representations": {},
    }
    results = report["representations"]
    assert isinstance(results, dict)
    for name, values in feature_sets.items():
        x = np.stack(values)
        use_pca = name == "early_pooled_spatial"
        results[name] = {
            "dimensions": int(x.shape[1]),
            "unsafe_output": evaluate(x, y_output, group_array, use_pca),
            "unsafe_intent": evaluate(x, y_intent, group_array, use_pca),
        }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in results.items():
        assert isinstance(result, dict)
        output = result["unsafe_output"]
        intent = result["unsafe_intent"]
        print(
            f"{name}: output BA={output['balanced_accuracy']:.3f}, "
            f"AUC={output['roc_auc']:.3f}; intent BA={intent['balanced_accuracy']:.3f}, "
            f"AUC={intent['roc_auc']:.3f}"
        )
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
