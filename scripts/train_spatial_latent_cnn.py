"""Train a compact spatial CNN on final SD3.5 latents for violence detection.

The script preserves the frozen semantic split. Model selection and threshold
selection use only the validation set; the test set is evaluated once after the
best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "annotations" / "latent_guard_pilot_v1_human_labels_v2.csv"
DEFAULT_RECORDS = ROOT / "data" / "raw" / "latent_guard" / "latent_guard_pilot_v1" / "records"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "spatial_latent_cnn_violence_v1.json"
DEFAULT_MODEL = ROOT / "models" / "latent_guard" / "spatial_latent_cnn_violence_v1.pt"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 5230
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 80
    patience: int = 12
    min_delta: float = 1e-4
    dropout: float = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_index(labels_path: Path, records_root: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    labels = labels[
        (labels["quality_label"] == "usable")
        & labels["review_label"].isin(["benign", "violence_gore"])
    ].copy()
    latent_paths: list[str] = []
    for sample_id in labels["sample_id"]:
        record_path = records_root / f"{sample_id}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        latent_path = Path(record["latent_file"])
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing latent for {sample_id}: {latent_path}")
        latent_paths.append(str(latent_path))
    labels["latent_path"] = latent_paths
    labels["target"] = (labels["review_label"] == "violence_gore").astype(np.float32)
    return labels.reset_index(drop=True)


def load_expansion_index(labels_path: Path, records_root: Path) -> pd.DataFrame:
    """Load decisive binary labels produced by the blind expansion reviewer."""

    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    labels = labels[
        (labels["quality_label"] == "usable")
        & labels["violence_label"].isin(["violence", "non_violent"])
    ].copy()
    latent_paths: list[str] = []
    for sample_id in labels["sample_id"]:
        record = json.loads(
            (records_root / f"{sample_id}.json").read_text(encoding="utf-8")
        )
        latent_path = Path(record["latent_file"])
        if not latent_path.exists():
            raise FileNotFoundError(f"Missing expansion latent for {sample_id}: {latent_path}")
        latent_paths.append(str(latent_path))
    labels["latent_path"] = latent_paths
    labels["target"] = (labels["violence_label"] == "violence").astype(np.float32)
    labels["review_label"] = np.where(
        labels["target"] == 1, "violence_gore", "benign"
    )
    return labels.reset_index(drop=True)


def load_latent(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        latent = archive["latent"]
    if latent.shape == (1, 16, 128, 128):
        latent = latent[0]
    if latent.shape != (16, 128, 128):
        raise ValueError(f"Expected [16,128,128] latent, received {latent.shape} from {path}")
    return latent.astype(np.float32, copy=False)


def channel_normalization(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(16, dtype=np.float64)
    channel_sq_sum = np.zeros(16, dtype=np.float64)
    pixels = 0
    for path in rows["latent_path"]:
        latent = load_latent(path)
        channel_sum += latent.sum(axis=(1, 2), dtype=np.float64)
        channel_sq_sum += np.square(latent, dtype=np.float64).sum(axis=(1, 2))
        pixels += latent.shape[1] * latent.shape[2]
    mean = channel_sum / pixels
    variance = np.maximum(channel_sq_sum / pixels - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class LatentDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        latent = (load_latent(row["latent_path"]) - self.mean) / self.std
        value = torch.from_numpy(np.ascontiguousarray(latent))
        if self.augment and torch.rand(()) < 0.5:
            value = torch.flip(value, dims=(2,))
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return value, target


class SpatialLatentCNN(nn.Module):
    """Small convolutional detector retaining a 4x4 coarse spatial layout."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(value)).squeeze(1)


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    truth: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for values, targets in loader:
        values = values.to(device, non_blocking=True)
        logits = model(values)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        truth.append(targets.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.concatenate(truth).astype(int), np.concatenate(scores), elapsed


def select_threshold(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        score = balanced_accuracy_score(truth, predicted)
        if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def metrics(truth: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "sample_count": int(truth.size),
        "positive_count": int(truth.sum()),
        "threshold": round(float(threshold), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 4),
        "harmful_recall": round(float(tp / (tp + fn)), 4),
        "benign_false_positive_rate": round(float(fp / (fp + tn)), 4),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(truth, scores)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.inference_mode()
def benchmark_model_only(
    model: nn.Module, device: torch.device, repeats: int = 200
) -> float:
    """Measure batch-one classifier latency with an already-resident latent."""
    model.eval()
    example = torch.zeros((1, 16, 128, 128), dtype=torch.float32, device=device)
    for _ in range(20):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / repeats


def train(args: argparse.Namespace) -> dict[str, object]:
    config = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    set_deterministic(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = load_index(args.labels.resolve(), args.records.resolve())
    label_sources = [
        {
            "labels": str(args.labels.resolve()),
            "records": str(args.records.resolve()),
            "sha256": sha256_file(args.labels.resolve()),
            "decisive_rows": int(len(rows)),
        }
    ]
    if args.extra_labels is not None:
        if args.extra_records is None:
            raise ValueError("--extra-records is required with --extra-labels")
        extra = load_expansion_index(
            args.extra_labels.resolve(), args.extra_records.resolve()
        )
        if set(rows["sample_id"]) & set(extra["sample_id"]):
            raise ValueError("Pilot and expansion sample IDs overlap")
        label_sources.append(
            {
                "labels": str(args.extra_labels.resolve()),
                "records": str(args.extra_records.resolve()),
                "sha256": sha256_file(args.extra_labels.resolve()),
                "decisive_rows": int(len(extra)),
            }
        )
        rows = pd.concat([rows, extra], ignore_index=True, sort=False)
    split_rows = {name: rows[rows["split"] == name].copy() for name in ("train", "validation", "test")}
    mean, std = channel_normalization(split_rows["train"])
    loaders = {
        name: DataLoader(
            LatentDataset(part, mean, std, augment=name == "train"),
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )
        for name, part in split_rows.items()
    }
    model = SpatialLatentCNN(config.dropout).to(device)
    positives = float(split_rows["train"]["target"].sum())
    negatives = float(len(split_rows["train"]) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_auc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale_epochs = 0
    training_started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for values, targets in loaders["train"]:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_truth, val_scores, _ = predict(model, loaders["validation"], device)
        val_auc = float(roc_auc_score(val_truth, val_scores))
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6), "validation_roc_auc": round(val_auc, 6)})
        if val_auc > best_auc + config.min_delta:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)

    val_truth, val_scores, _ = predict(model, loaders["validation"], device)
    threshold, _ = select_threshold(val_truth, val_scores)
    test_truth, test_scores, test_seconds = predict(model, loaders["test"], device)
    model_only_ms = benchmark_model_only(model, device)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "state_dict": best_state,
            "channel_mean": mean,
            "channel_std": std,
            "threshold": threshold,
            "config": asdict(config),
        },
        args.model_output,
    )
    return {
        "schema_version": 1,
        "experiment": "spatial_final_latent_cnn_violence_vs_benign",
        "model": "SD3.5 Medium final latent [16,128,128]",
        "labels_sha256": sha256_file(args.labels.resolve()),
        "label_sources": label_sources,
        "selection_protocol": "early stop on validation ROC AUC; select threshold on validation balanced accuracy; evaluate test once",
        "device": str(device),
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "split_counts": {
            name: {
                "total": int(len(part)),
                "benign": int((part["target"] == 0).sum()),
                "violence_gore": int((part["target"] == 1).sum()),
            }
            for name, part in split_rows.items()
        },
        "best_epoch": best_epoch,
        "best_validation_roc_auc": round(best_auc, 4),
        "validation": metrics(val_truth, val_scores, threshold),
        "test": metrics(test_truth, test_scores, threshold),
        "training_seconds": round(training_seconds, 3),
        "test_inference_seconds": round(test_seconds, 6),
        "test_pipeline_milliseconds_per_sample": round(test_seconds * 1000 / len(test_truth), 4),
        "model_only_milliseconds_per_sample": round(model_only_ms, 4),
        "test_predictions": [
            {
                "sample_id": sample_id,
                "true_label": "violence_gore" if truth else "benign",
                "violence_score": round(float(score), 8),
                "predicted_label": "violence_gore" if score >= threshold else "benign",
            }
            for sample_id, truth, score in zip(
                split_rows["test"]["sample_id"], test_truth, test_scores
            )
        ],
        "history": history,
        "limitations": [
            "Labels come from one human reviewer.",
            "This development experiment uses the original violence/benign pilot rather than held-out DACA/PGJ generations.",
            "The test set is small; confidence intervals and independent binary relabelling remain required.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--extra-labels", type=Path)
    parser.add_argument("--extra-records", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=5230)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("device", "parameter_count", "best_epoch", "validation", "test", "training_seconds", "test_pipeline_milliseconds_per_sample", "model_only_milliseconds_per_sample")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
