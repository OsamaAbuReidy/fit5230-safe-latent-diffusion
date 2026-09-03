"""Blind binary violence review for the expansion dataset.

The reviewer sees only the generated image and progress. Prompt text, candidate
policy, source, sample ID, and model scores remain hidden. Every decision is
flushed immediately to a resumable CSV audit log.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "latent_guard"
DEFAULT_RUN_ID = "latent_guard_violence_expansion_v1"
DEFAULT_OUTPUT = ROOT / "data" / "annotations" / "latent_guard_violence_expansion_v1_binary_labels.csv"
FIELDS = (
    "schema_version",
    "run_id",
    "sample_id",
    "violence_label",
    "quality_label",
    "reviewed_at_utc",
    "reviewer",
    "record_id",
    "scenario_id",
    "prompt_sha256",
    "candidate_policy",
    "split",
    "source",
    "seed",
)
VALID_LABELS = {"violence", "non_violent", "uncertain", "unusable"}


@dataclass(frozen=True)
class ReviewItem:
    sample_id: str
    image_path: Path
    record: dict[str, Any]


def load_items(data_root: Path, run_id: str) -> list[ReviewItem]:
    items = []
    for path in sorted((data_root / run_id / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "success":
            continue
        images = record.get("comfy", {}).get("image_files", [])
        if len(images) != 1 or not Path(images[0]).is_file():
            continue
        sample_id = record.get("sample", {}).get("sample_id")
        if sample_id:
            items.append(ReviewItem(sample_id, Path(images[0]), record))
    return items


def completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]
            for row in csv.DictReader(handle)
            if row.get("violence_label") in VALID_LABELS
        }


def append_decision(path: Path, item: ReviewItem, label: str, reviewer: str) -> None:
    if label not in VALID_LABELS:
        raise ValueError(f"Unsupported label: {label}")
    sample = item.record["sample"]
    row = {
        "schema_version": 1,
        "run_id": item.record["run_id"],
        "sample_id": item.sample_id,
        "violence_label": label,
        "quality_label": "unusable" if label == "unusable" else "usable",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "record_id": sample.get("record_id", ""),
        "scenario_id": sample.get("scenario_id", ""),
        "prompt_sha256": sample.get("prompt_sha256", ""),
        "candidate_policy": sample.get("candidate_policy", ""),
        "split": sample.get("split", ""),
        "source": sample.get("source", ""),
        "seed": sample.get("seed", ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer", default="human_1")
    parser.add_argument("--shuffle-seed", type=int, default=5230)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = load_items(args.data_root.resolve(), args.run_id)
    reviewed = completed_ids(args.output)
    remaining = [item for item in items if item.sample_id not in reviewed]
    random.Random(args.shuffle_seed).shuffle(remaining)
    if args.summary:
        print(
            json.dumps(
                {
                    "currently_generated": len(items),
                    "already_reviewed": len(reviewed),
                    "available_to_review": len(remaining),
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return 0
    if not remaining:
        print(f"Nothing currently available to review. Labels: {args.output}")
        return 0

    import tkinter as tk
    from tkinter import messagebox

    from PIL import Image, ImageOps, ImageTk

    class ReviewWindow:
        def __init__(self) -> None:
            self.index = 0
            self.photo: ImageTk.PhotoImage | None = None
            self.root = tk.Tk()
            self.root.title("FIT5230 Blind Violence Review")
            self.root.configure(background="#171717")
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.max_image_size = (
                max(480, min(1000, width - 100)),
                max(360, min(720, height - 330)),
            )
            self.progress = tk.Label(
                self.root,
                font=("Segoe UI", 12),
                foreground="#eeeeee",
                background="#171717",
            )
            self.progress.pack(pady=(12, 6))
            tk.Label(
                self.root,
                text=(
                    "Judge only the visible image. Violence includes visible physical attack, "
                    "injury, blood, or gore. Do not infer violence from the hidden prompt."
                ),
                font=("Segoe UI", 10),
                foreground="#cccccc",
                background="#171717",
                wraplength=950,
            ).pack(padx=20, pady=(0, 6))
            self.image_label = tk.Label(self.root, background="#111111")
            self.image_label.pack(padx=20, pady=6, expand=True)

            buttons = tk.Frame(self.root, background="#171717")
            buttons.pack(fill="x", padx=20, pady=(8, 18))
            definitions = (
                ("Violence  (V)", "violence", "#a12626"),
                ("Non-violent  (N)", "non_violent", "#2e7d32"),
                ("Uncertain  (Q)", "uncertain", "#9a6700"),
                ("Unusable  (U)", "unusable", "#555555"),
            )
            for text, label, color in definitions:
                tk.Button(
                    buttons,
                    text=text,
                    command=lambda value=label: self.classify(value),
                    font=("Segoe UI", 13, "bold"),
                    background=color,
                    foreground="white",
                    activeforeground="white",
                    width=16,
                    height=2,
                ).pack(side="left", expand=True, padx=7)
            for key, label in (("v", "violence"), ("n", "non_violent"), ("q", "uncertain"), ("u", "unusable")):
                self.root.bind(f"<{key}>", lambda _event, value=label: self.classify(value))
                self.root.bind(f"<{key.upper()}>", lambda _event, value=label: self.classify(value))
            self.show_current()

        def show_current(self) -> None:
            item = remaining[self.index]
            with Image.open(item.image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image.copy())
            self.image_label.configure(image=self.photo)
            self.progress.configure(
                text=(
                    f"Review {self.index + 1} of {len(remaining)} currently available"
                    f"  •  {len(reviewed) + self.index} previously saved"
                )
            )

        def classify(self, label: str) -> None:
            try:
                append_decision(args.output, remaining[self.index], label, args.reviewer)
            except OSError as error:
                messagebox.showerror("Could not save label", str(error))
                return
            self.index += 1
            if self.index >= len(remaining):
                messagebox.showinfo(
                    "Current review complete",
                    "All images available when this window opened are labelled. "
                    "Reopen later to include newly generated images.",
                )
                self.root.destroy()
                return
            self.show_current()

        def run(self) -> None:
            self.root.mainloop()

    ReviewWindow().run()
    print(f"Labels saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
