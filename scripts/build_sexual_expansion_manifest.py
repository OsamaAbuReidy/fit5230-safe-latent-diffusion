"""Build the development-only sexual/nudity expansion manifest.

The expansion contains 200 direct sexual/nudity candidates and 200 hard
benign human-subject controls. It excludes every record used by the original
pilot and all held-out adversarial sources. No test split is created: final
evaluation remains reserved for a fresh attack set after model development is
frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED = ROOT / "data" / "processed" / "prompt_sources_normalized.csv"
DEFAULT_PILOT = ROOT / "data" / "manifests" / "latent_guard_pilot_400.csv"
DEFAULT_OUTPUT = ROOT / "data" / "manifests" / "latent_guard_sexual_expansion_400.csv"
DEFAULT_SUMMARY = ROOT / "data" / "manifests" / "latent_guard_sexual_expansion_400_summary.json"
SALT = "fit5230-sexual-expansion-v1"

# Benign prompts are deliberately close to the visual domain of the target
# class: people, skin, fashion, intimacy, bathing, anatomy, and figure art.
# Human output labels, rather than these terms, determine the training target.
HUMAN_TERMS = (
    "woman",
    "women",
    "man",
    "men",
    "person",
    "people",
    "lady",
    "ladies",
    "gentleman",
    "gentlemen",
    "couple",
    "adult",
    "adults",
)

HARD_BENIGN_TERMS = (
    "fashion",
    "model",
    "models",
    "romantic",
    "kiss",
    "kissing",
    "embrace",
    "bedroom",
    "bed",
    "bath",
    "bathing",
    "beach",
    "swim",
    "swimming",
    "swimsuit",
    "pool",
    "body",
    "skin",
    "shirtless",
    "dress",
    "wedding",
    "dance",
    "spa",
    "massage",
    "figure",
    "anatomy",
)

# The experiment concerns consenting adults. Prompts with explicit minor terms
# are excluded before selection rather than filtered after seeing outputs.
MINOR_TERMS = (
    "baby",
    "babies",
    "child",
    "children",
    "kid",
    "kids",
    "boy",
    "boys",
    "girl",
    "girls",
    "teen",
    "teenager",
    "minor",
    "schoolboy",
    "schoolgirl",
)


def contains_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) for term in terms)


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(
        bool(re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE))
        for term in terms
    )


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SALT}|rank|{value}".encode()).hexdigest()


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(f"{SALT}|seed|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def choose(
    rows: list[dict[str, str]],
    *,
    source: str,
    policy: str,
    count: int,
    excluded: set[str],
    hard_benign: bool = False,
) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        if row["record_id"] in excluded:
            continue
        if row["source"] != source or row["eligibility"] != "development":
            continue
        if row["candidate_policies"] != policy:
            continue
        if contains_term(row["prompt"], MINOR_TERMS):
            continue
        if hard_benign:
            if not contains_term(row["prompt"], HUMAN_TERMS):
                continue
            matches = count_terms(row["prompt"], HARD_BENIGN_TERMS)
            row = {**row, "hard_benign_term_matches": str(matches)}
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            -int(row.get("hard_benign_term_matches", "0")),
            stable_rank(row["record_id"]),
        )
    )
    selected: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for row in candidates:
        if row["prompt_sha256"] in seen_hashes:
            continue
        seen_hashes.add(row["prompt_sha256"])
        selected.append(row)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} eligible {source}/{policy} rows; need {count}")


def manifest_row(
    source: dict[str, str], sample_id: str, policy: str, split: str
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "record_id": source["record_id"],
        "source": source["source"],
        "source_id": source["source_id"],
        "scenario_id": source["scenario_id"],
        "prompt_sha256": source["prompt_sha256"],
        "prompt_intent": source["prompt_intent"],
        "candidate_policy": policy,
        "attack_type": source["attack_type"],
        "split": split,
        "seed": stable_seed(sample_id),
        "image_label_status": "pending_generation",
    }


def assign_group(
    selected: list[dict[str, str]], policy: str, prefix: str
) -> list[dict[str, object]]:
    ordered = sorted(selected, key=lambda row: stable_rank(f"split|{row['scenario_id']}"))
    output = []
    for index, row in enumerate(ordered, start=1):
        split = "train" if index <= 160 else "validation"
        output.append(manifest_row(row, f"lgs_{prefix}_{index:03d}", policy, split))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    rows = read_csv(args.normalized)
    pilot = read_csv(args.pilot)
    excluded = {row["record_id"] for row in pilot}

    sexual = [
        *choose(rows, source="i2p", policy="sexual", count=100, excluded=excluded),
        *choose(
            rows,
            source="t2i_risky_prompt",
            policy="sexual",
            count=100,
            excluded=excluded,
        ),
    ]
    hard_benign = [
        *choose(
            rows,
            source="coco30k",
            policy="benign",
            count=125,
            excluded=excluded,
            hard_benign=True,
        ),
        *choose(
            rows,
            source="parti_prompts",
            policy="benign",
            count=75,
            excluded=excluded,
            hard_benign=True,
        ),
    ]
    manifest = [
        *assign_group(sexual, "sexual", "sexual"),
        *assign_group(hard_benign, "hard_benign", "hard_benign"),
    ]

    if len(manifest) != 400:
        raise RuntimeError(f"Expected 400 rows, found {len(manifest)}")
    if len({row["record_id"] for row in manifest}) != 400:
        raise RuntimeError("Duplicate source records in expansion")
    if len({row["prompt_sha256"] for row in manifest}) != 400:
        raise RuntimeError("Duplicate prompt text in expansion")
    if {row["record_id"] for row in manifest} & excluded:
        raise RuntimeError("Original pilot record leaked into expansion")
    if any(row["split"] == "test" for row in manifest):
        raise RuntimeError("Expansion must not create a test split")

    source_by_id = {row["record_id"]: row for row in rows}
    if any(source_by_id[row["record_id"]]["eligibility"] != "development" for row in manifest):
        raise RuntimeError("Held-out adversarial source leaked into expansion")
    if any(contains_term(source_by_id[row["record_id"]]["prompt"], MINOR_TERMS) for row in manifest):
        raise RuntimeError("Prompt containing an excluded minor term entered the manifest")

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
        "contains_original_pilot_records": False,
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
