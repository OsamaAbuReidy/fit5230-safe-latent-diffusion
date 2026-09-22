"""Build a frozen 200-sample supplement for the sexual/nudity head.

The supplement is selected only from development records unused by both the
original 400-sample pilot and the frozen sexual expansion. It adds 100 direct
sexual/nudity candidates and 100 benign human-subject controls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_sexual_expansion_manifest import (
    HARD_BENIGN_TERMS,
    HUMAN_TERMS,
    MINOR_TERMS,
    SALT,
    choose,
    contains_term,
    manifest_row,
    read_csv,
    stable_rank,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED = ROOT / "data" / "processed" / "prompt_sources_normalized.csv"
DEFAULT_PILOT = ROOT / "data" / "manifests" / "latent_guard_pilot_400.csv"
DEFAULT_EXPANSION = ROOT / "data" / "manifests" / "latent_guard_sexual_expansion_400.csv"
DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "latent_guard_sexual_supplement_200.csv"
DEFAULT_SUMMARY = ROOT / "data" / "manifests" / "latent_guard_sexual_supplement_200_summary.json"


def assign_group(
    selected: list[dict[str, str]], policy: str, prefix: str
) -> list[dict[str, object]]:
    ordered = sorted(selected, key=lambda row: stable_rank(f"supplement-split|{row['scenario_id']}"))
    output = []
    for index, row in enumerate(ordered, start=1):
        split = "train" if index <= 80 else "validation"
        output.append(manifest_row(row, f"lgss_{prefix}_{index:03d}", policy, split))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--expansion", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.normalized)
    pilot = read_csv(args.pilot)
    expansion = read_csv(args.expansion)
    excluded = {row["record_id"] for row in [*pilot, *expansion]}

    sexual = [
        *choose(rows, source="i2p", policy="sexual", count=50, excluded=excluded),
        *choose(
            rows,
            source="t2i_risky_prompt",
            policy="sexual",
            count=50,
            excluded=excluded,
        ),
    ]
    hard_benign = choose(
        rows,
        source="coco30k",
        policy="benign",
        count=100,
        excluded=excluded,
        hard_benign=True,
    )
    manifest = [
        *assign_group(sexual, "sexual", "sexual"),
        *assign_group(hard_benign, "hard_benign", "hard_benign"),
    ]

    if len(manifest) != 200:
        raise RuntimeError(f"Expected 200 rows, found {len(manifest)}")
    if len({row["record_id"] for row in manifest}) != 200:
        raise RuntimeError("Duplicate source records in supplement")
    if len({row["prompt_sha256"] for row in manifest}) != 200:
        raise RuntimeError("Duplicate prompt text in supplement")
    if {row["record_id"] for row in manifest} & excluded:
        raise RuntimeError("Previously used source record leaked into supplement")
    if any(row["split"] == "test" for row in manifest):
        raise RuntimeError("Supplement must not create a test split")

    source_by_id = {row["record_id"]: row for row in rows}
    if any(source_by_id[row["record_id"]]["eligibility"] != "development" for row in manifest):
        raise RuntimeError("Held-out adversarial source leaked into supplement")
    if any(contains_term(source_by_id[row["record_id"]]["prompt"], MINOR_TERMS) for row in manifest):
        raise RuntimeError("Prompt containing an excluded minor term entered the supplement")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "schema_version": 1,
        "manifest": str(args.output.resolve()),
        "sample_count": len(manifest),
        "policy_counts": Counter(row["candidate_policy"] for row in manifest),
        "split_counts": Counter(row["split"] for row in manifest),
        "source_counts": Counter(row["source"] for row in manifest),
        "excluded_prior_record_count": len(excluded),
        "contains_prior_records": False,
        "contains_heldout_adversarial_records": False,
        "contains_test_split": False,
        "contains_explicit_minor_terms": False,
        "selection_salt": SALT,
        "required_human_terms": list(HUMAN_TERMS),
        "hard_benign_terms": list(HARD_BENIGN_TERMS),
        "excluded_minor_terms": list(MINOR_TERMS),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
