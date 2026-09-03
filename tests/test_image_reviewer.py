import csv
import json
from pathlib import Path

from scripts.review_latent_guard_images import (
    ReviewItem,
    append_label,
    completed_sample_ids,
    load_review_items,
)


def sample_record(image: Path) -> dict:
    return {
        "status": "success",
        "run_id": "pilot",
        "comfy": {"image_files": [str(image)]},
        "sample": {
            "sample_id": "sample_001",
            "record_id": "source:1",
            "scenario_id": "scenario:1",
            "prompt_sha256": "abc123",
            "candidate_policy": "sexual",
            "split": "train",
            "source": "fixture",
            "seed": "42",
        },
    }


def test_load_and_resume_review(tmp_path: Path) -> None:
    record_root = tmp_path / "pilot" / "records"
    record_root.mkdir(parents=True)
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    record = sample_record(image)
    (record_root / "sample_001.json").write_text(json.dumps(record), encoding="utf-8")

    items = load_review_items(tmp_path, "pilot")
    assert len(items) == 1
    labels = tmp_path / "labels.csv"
    append_label(labels, ReviewItem("sample_001", image, record), "sexual", "reviewer")
    assert completed_sample_ids(labels) == {"sample_001"}
    with labels.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["review_label"] == "sexual"
    assert row["candidate_policy"] == "sexual"


def test_missing_images_are_skipped(tmp_path: Path) -> None:
    record_root = tmp_path / "pilot" / "records"
    record_root.mkdir(parents=True)
    record = sample_record(tmp_path / "missing.png")
    (record_root / "missing.json").write_text(json.dumps(record), encoding="utf-8")
    assert load_review_items(tmp_path, "pilot") == []


def test_invalid_is_a_resumable_label(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    record = sample_record(image)
    labels = tmp_path / "labels.csv"
    append_label(labels, ReviewItem("sample_001", image, record), "invalid", "reviewer")
    assert completed_sample_ids(labels) == {"sample_001"}


def test_all_policy_labels_are_resumable(tmp_path: Path) -> None:
    for index, label in enumerate(("sexual", "violence_gore", "dangerous"), start=1):
        image = tmp_path / f"image_{index}.png"
        image.write_bytes(b"fixture")
        record = sample_record(image)
        record["sample"]["sample_id"] = f"sample_{index:03d}"
        labels = tmp_path / f"labels_{index}.csv"
        item = ReviewItem(record["sample"]["sample_id"], image, record)
        append_label(labels, item, label, "reviewer")
        assert completed_sample_ids(labels) == {item.sample_id}
