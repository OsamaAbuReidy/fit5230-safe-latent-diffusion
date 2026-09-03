"""Audit the currently completed subset of the violence expansion run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "latent_guard_violence_expansion_240.csv"
DEFAULT_RUN_ROOT = ROOT / "data" / "raw" / "latent_guard" / "latent_guard_violence_expansion_v1"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "latent_guard_violence_expansion_v1_snapshot_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    errors: list[str] = []
    records: list[dict] = []
    record_paths = sorted((args.run_root / "records").glob("*.json"))
    for path in record_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"unreadable record {path.name}: {error}")
            continue
        records.append(record)

    completed_ids: list[str] = []
    image_paths: list[str] = []
    latent_paths: list[str] = []
    image_sizes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    latent_shapes: Counter[str] = Counter()
    latent_dtypes: Counter[str] = Counter()
    setting_variants: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    elapsed: list[float] = []

    for record in records:
        sample = record.get("sample", {})
        sample_id = sample.get("sample_id", "")
        expected = manifest_by_id.get(sample_id)
        if expected is None:
            errors.append(f"record absent from manifest: {sample_id}")
            continue
        if record.get("status") != "success":
            errors.append(f"non-success completion record: {sample_id}")
            continue
        completed_ids.append(sample_id)
        policy_counts[expected["candidate_policy"]] += 1
        split_counts[expected["split"]] += 1
        elapsed.append(float(record.get("elapsed_seconds", 0)))
        for field in ("record_id", "prompt_sha256", "seed", "split", "candidate_policy"):
            if str(sample.get(field)) != str(expected.get(field)):
                errors.append(f"manifest mismatch {sample_id}: {field}")

        images = record.get("comfy", {}).get("image_files", [])
        if len(images) != 1:
            errors.append(f"expected one image for {sample_id}, found {len(images)}")
        else:
            image_path = Path(images[0])
            image_paths.append(str(image_path))
            if not image_path.is_file():
                errors.append(f"missing image: {sample_id}")
            else:
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        image_sizes[str(list(image.size))] += 1
                        image_modes[image.mode] += 1
                        if image.size != (1024, 1024):
                            errors.append(f"wrong image size {sample_id}: {image.size}")
                except (OSError, ValueError) as error:
                    errors.append(f"unreadable image {sample_id}: {error}")

        latent_path = Path(record.get("latent_file", ""))
        latent_paths.append(str(latent_path))
        if not latent_path.is_file():
            errors.append(f"missing latent: {sample_id}")
        else:
            try:
                with np.load(latent_path, allow_pickle=False) as archive:
                    latent = archive["latent"]
                    metadata = json.loads(str(archive["metadata_json"].item()))
                latent_shapes[str(list(latent.shape))] += 1
                latent_dtypes[str(latent.dtype)] += 1
                if latent.shape != (1, 16, 128, 128):
                    errors.append(f"wrong latent shape {sample_id}: {latent.shape}")
                if latent.dtype != np.float16:
                    errors.append(f"wrong latent dtype {sample_id}: {latent.dtype}")
                if not np.isfinite(latent).all():
                    errors.append(f"non-finite latent: {sample_id}")
                for field in ("sample_id", "prompt_sha256", "seed"):
                    if str(metadata.get(field)) != str(expected.get(field)):
                        errors.append(f"latent metadata mismatch {sample_id}: {field}")
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"unreadable latent {sample_id}: {error}")

        generation = record.get("generation", {})
        setting_variants[json.dumps(generation, sort_keys=True)] += 1

    if len(completed_ids) != len(set(completed_ids)):
        errors.append("duplicate completed sample IDs")
    if len(image_paths) != len(set(image_paths)):
        errors.append("duplicate image paths")
    if len(latent_paths) != len(set(latent_paths)):
        errors.append("duplicate latent paths")
    if len(setting_variants) > 1:
        errors.append(f"generation settings have {len(setting_variants)} variants")

    failed_attempts = 0
    attempt_statuses: Counter[str] = Counter()
    attempts_path = args.run_root / "attempts.jsonl"
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            attempt_statuses[item.get("status", "unknown")] += 1
        failed_attempts = sum(
            count for status, count in attempt_statuses.items() if status != "success"
        )

    report = {
        "schema_version": 1,
        "run_id": "latent_guard_violence_expansion_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "snapshot": {
            "planned": len(manifest),
            "completed": len(completed_ids),
            "remaining": len(manifest) - len(completed_ids),
            "policy_counts": dict(policy_counts),
            "split_counts": dict(split_counts),
            "mean_completed_seconds": round(float(np.mean(elapsed)), 2) if elapsed else None,
        },
        "artifacts": {
            "readable_images": sum(image_sizes.values()),
            "image_sizes": dict(image_sizes),
            "image_modes": dict(image_modes),
            "valid_latents": sum(latent_shapes.values()),
            "latent_shapes": dict(latent_shapes),
            "latent_dtypes": dict(latent_dtypes),
            "generation_setting_variants": len(setting_variants),
        },
        "attempt_log": {
            "statuses": dict(attempt_statuses),
            "non_success_attempts": failed_attempts,
            "note": "Incomplete startup attempts are retained in the audit log rather than silently discarded.",
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
