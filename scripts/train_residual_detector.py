"""Train the frozen pilot residual detector on direct and benign prompts."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data" / "raw" / "grsp_pilot" / "features" / "grsp_pilot_02"
ANNOTATIONS = ROOT / "data" / "raw" / "grsp_pilot" / "manifests" / "grsp_pilot_02_annotations.csv"
MODEL_DIR = ROOT / "outputs" / "models" / "grsp_pilot_02"
EARLY_STEPS = 6
POOL_SIZE = 8
PCA_COMPONENTS = 12


def locate(prompt_id: str, seed: int) -> Path:
    matches = list((FEATURE_ROOT / prompt_id).glob(f"seed_{seed}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive for {prompt_id}/{seed}, found {len(matches)}")
    return matches[0]


def extract(archive_path: Path) -> np.ndarray:
    archive = np.load(archive_path, allow_pickle=False)
    pooled = archive["pooled_residuals"][:, 0].astype(np.float32)
    if pooled.shape[0] < EARLY_STEPS or pooled.shape[-2:] != (POOL_SIZE, POOL_SIZE):
        raise RuntimeError(f"Unexpected residual shape in {archive_path}: {pooled.shape}")
    return pooled[:EARLY_STEPS].reshape(-1)


def main() -> int:
    with ANNOTATIONS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    x: list[np.ndarray] = []
    y: list[int] = []
    samples: list[dict[str, object]] = []
    for row in rows:
        prompt_id = row["prompt_id"]
        seed = int(row["seed"])
        archive_path = locate(prompt_id, seed)
        x.append(extract(archive_path))
        # Freeze the training target as unsafe intent. Attack-set images will be
        # manually validated and unsuccessful attacks excluded from ASR tests.
        y.append(1 if prompt_id.startswith("direct_") else 0)
        samples.append({"prompt_id": prompt_id, "seed": seed, "feature_file": str(archive_path)})

    matrix = np.stack(x)
    labels = np.asarray(y)
    detector = make_pipeline(
        StandardScaler(),
        PCA(n_components=PCA_COMPONENTS, random_state=0),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
    )
    detector.fit(matrix, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "early_spatial_detector.joblib"
    metadata_path = MODEL_DIR / "early_spatial_detector.json"
    joblib.dump(detector, model_path)
    metadata = {
        "status": "frozen_pilot_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": "grsp_pilot_02",
        "training_target": "unsafe_intent",
        "sample_count": len(labels),
        "safe_count": int((labels == 0).sum()),
        "unsafe_count": int((labels == 1).sum()),
        "feature": {
            "name": "early_pooled_spatial",
            "definition": "flattened pooled CFG residual for sampling steps 0 through 5",
            "residual_definition": "cond_denoised - uncond_denoised",
            "early_steps": EARLY_STEPS,
            "pool_size": POOL_SIZE,
            "dimensions": int(matrix.shape[1]),
        },
        "model": {
            "scaler": "StandardScaler",
            "projection": f"PCA(n_components={PCA_COMPONENTS}, random_state=0)",
            "classifier": "LogisticRegression(C=0.1, class_weight='balanced')",
        },
        "selection_warning": (
            "Representation selected after exploratory screening on five semantic pairs. "
            "DACA/PGJ attack results must remain untouched test data."
        ),
        "samples": samples,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Training samples: {len(labels)} ({int(labels.sum())} unsafe, {int((1-labels).sum())} safe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
