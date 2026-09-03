"""Build the small, prompt-free feature bundle used by the Milestone 2 Colab.

The bundle contains only pooled final-latent features, frozen split assignments,
human output labels, and predictions from the released Multihead Detector. Raw
prompts, images, and full latent tensors are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_latent_guard_classifier import CLASSES, pooled_latent_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "annotations" / "latent_guard_pilot_v1_human_labels_v2.csv"
DEFAULT_RECORDS = ROOT / "data" / "raw" / "latent_guard" / "latent_guard_pilot_v1" / "records"
DEFAULT_MULTIHEAD = ROOT / "data" / "results" / "multihead_phase1_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "public" / "milestone2_pooled_latents.npz"
DEFAULT_METADATA = ROOT / "data" / "public" / "milestone2_pooled_latents.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--multihead", type=Path, default=DEFAULT_MULTIHEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = pd.read_csv(args.labels, dtype=str).fillna("")
    labels = labels[
        (labels["quality_label"] == "usable") & labels["review_label"].isin(CLASSES)
    ].copy()
    # Preserve the canonical review-file order. Iterative solvers can exhibit
    # small floating-point differences when otherwise identical rows are
    # reordered, which is enough to alter a validation tie near the threshold.
    labels = labels.reset_index(drop=True)

    multihead = pd.read_csv(args.multihead)
    multihead = multihead[multihead["status"] == "ok"].copy()
    if multihead["sample_id"].duplicated().any():
        raise ValueError("Multihead predictions contain duplicate sample IDs")

    score_columns = [
        "score_sexual",
        "score_violent",
        "score_disturbing",
        "score_hateful",
        "score_political",
        "score_any",
        "score_violence_extended",
    ]
    joined = labels.merge(
        multihead[["sample_id", *score_columns]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if joined[score_columns].isna().any().any():
        missing = joined.loc[joined["score_any"].isna(), "sample_id"].tolist()
        raise ValueError(f"Missing Multihead scores for {len(missing)} samples: {missing[:5]}")

    features: list[np.ndarray] = []
    for sample_id in joined["sample_id"]:
        record_path = args.records / f"{sample_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        with np.load(record["latent_file"], allow_pickle=False) as archive:
            features.append(pooled_latent_features(archive["latent"]))

    x = np.stack(features).astype(np.float32)
    arrays = {
        "features": x,
        "sample_id": joined["sample_id"].to_numpy(dtype=str),
        "split": joined["split"].to_numpy(dtype=str),
        "human_label": joined["review_label"].to_numpy(dtype=str),
        "candidate_policy": joined["candidate_policy"].to_numpy(dtype=str),
    }
    arrays.update(
        {column: joined[column].to_numpy(dtype=np.float32) for column in score_columns}
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    metadata = {
        "schema_version": 1,
        "description": "Prompt-free pooled SD3.5 final-latent features for FIT5230 Milestone 2",
        "sample_count": int(len(joined)),
        "feature_count": int(x.shape[1]),
        "feature_definition": "8x8 spatial means plus per-channel mean/std/min/max",
        "source_latent_shape": [1, 16, 128, 128],
        "human_label_counts": joined["review_label"].value_counts().sort_index().to_dict(),
        "split_counts": joined["split"].value_counts().sort_index().to_dict(),
        "contains_prompts": False,
        "contains_images": False,
        "contains_full_latents": False,
        "multihead_threshold": 0.5,
        "integrity": {
            "unique_sample_ids": bool(joined["sample_id"].is_unique),
            "all_features_finite": bool(np.isfinite(x).all()),
            "all_teacher_scores_finite": bool(
                np.isfinite(joined[score_columns].to_numpy(dtype=np.float32)).all()
            ),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")
    print(f"Wrote {args.metadata}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
