"""Build publication-ready Milestone 2 Ed-post figures from frozen results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "milestone2"
SPATIAL = ROOT / "data" / "results" / "spatial_latent_cnn_violence_expanded_interim_v1.json"
LATENCY = ROOT / "data" / "results" / "spatial_latent_cnn_gpu_latency_v1.json"
MULTIHEAD_ROWS = ROOT / "data" / "results" / "multihead_phase1_predictions.csv"

NAVY = "#132238"
BLUE = "#2F6BFF"
TEAL = "#16A085"
ORANGE = "#F39C3D"
RED = "#D94A4A"
LIGHT = "#F4F7FB"
GREY = "#687386"


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, xy, width, height, text, color, *, text_color="white", size=12):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=text_color, fontsize=size, weight="bold")


def add_arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=2, color=color))


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "Two safety decision paths after SD3.5 denoising", ha="center", va="top", fontsize=20, weight="bold", color=NAVY)
    ax.text(0.5, 0.875, "PreDecodeGuard decides from the final latent; Multihead requires a decoded image", ha="center", va="top", fontsize=11.5, color=GREY)

    add_box(ax, (0.03, 0.40), 0.18, 0.18, "SD3.5 final latent\n1 × 16 × 128 × 128", NAVY)
    add_box(ax, (0.30, 0.66), 0.20, 0.16, "PreDecodeGuard CNN\n1.90 ms", TEAL)
    add_box(ax, (0.61, 0.66), 0.16, 0.16, "Safety decision", TEAL)
    add_box(ax, (0.84, 0.66), 0.13, 0.16, "Block\nbefore decode", "#087F5B")

    add_box(ax, (0.30, 0.18), 0.20, 0.16, "Tiled VAE decode\n2492.75 ms", BLUE)
    add_box(ax, (0.57, 0.18), 0.20, 0.16, "Multihead image model\n183.13 ms", ORANGE, text_color=NAVY)
    add_box(ax, (0.84, 0.18), 0.13, 0.16, "Safety decision\nafter decode", RED)

    add_arrow(ax, (0.21, 0.53), (0.30, 0.74), TEAL)
    add_arrow(ax, (0.50, 0.74), (0.61, 0.74), TEAL)
    add_arrow(ax, (0.77, 0.74), (0.84, 0.74), TEAL)
    add_arrow(ax, (0.21, 0.45), (0.30, 0.26), BLUE)
    add_arrow(ax, (0.50, 0.26), (0.57, 0.26), BLUE)
    add_arrow(ax, (0.77, 0.26), (0.84, 0.26), BLUE)

    ax.text(0.395, 0.60, "OUR PATH", ha="center", fontsize=10, color=TEAL, weight="bold")
    ax.text(0.395, 0.12, "POST-DECODE BASELINE", ha="center", fontsize=10, color=BLUE, weight="bold")
    ax.text(0.70, 0.52, "Blocked-output decision path", ha="center", fontsize=11, color=NAVY, weight="bold")
    ax.text(0.70, 0.46, "1.90 ms vs 2675.88 ms  •  1410.9× faster", ha="center", fontsize=12, color=TEAL, weight="bold")
    ax.text(0.5, 0.03, "SD3.5 denoising is common to both paths and excluded from the timing comparison.", ha="center", fontsize=10, color=GREY)
    save(fig, "pipeline_comparison.png")


def performance_figure() -> None:
    names = ["Pooled\nlatent", "Initial\nCNN", "Expanded\nCNN", "Multihead"]
    balanced = np.array([0.7270, 0.7674, 0.8524, 0.8766])
    recall = np.array([0.6154, 0.6154, 0.7692, 0.7692])
    auc = np.array([0.8449, 0.8685, 0.9045, 0.9504])
    fpr = np.array([0.1613, 0.0806, 0.0645, 0.0161])
    colors = ["#93A4B8", "#5B8FF9", TEAL, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [2.15, 1]})
    fig.suptitle("Frozen 75-image violence test set", fontsize=20, weight="bold", color=NAVY, y=0.99)
    x = np.arange(len(names))
    width = 0.23
    for offset, values, label, color in [(-width, balanced, "Balanced accuracy", BLUE), (0, recall, "Harmful recall", TEAL), (width, auc, "AUROC", ORANGE)]:
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_ylabel("Score (higher is better)")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title("Detection quality", color=NAVY, weight="bold")

    bars = axes[1].bar(x, fpr, color=colors, width=0.68)
    axes[1].bar_label(bars, labels=[f"{100*v:.1f}%" for v in fpr], padding=4, fontsize=10, weight="bold")
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 0.19)
    axes[1].set_ylabel("False-positive rate (lower is better)")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].set_title("Benign collateral", color=NAVY, weight="bold")
    fig.text(0.5, 0.015, "Expanded CNN matches Multihead harmful recall while remaining weaker in AUROC and benign FPR.", ha="center", fontsize=10.5, color=GREY)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save(fig, "performance_comparison.png")


def latency_figure() -> None:
    labels = ["PreDecodeGuard\nCNN", "Multihead\nclassifier", "Tiled VAE\ndecode", "Decode +\nMultihead"]
    values = np.array([1.8965, 183.1275, 2492.7486, 2675.8761])
    colors = [TEAL, ORANGE, BLUE, RED]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 6000)
    ax.set_ylabel("Median latency in milliseconds (log scale)")
    ax.set_title("Post-denoising safety pipeline latency", fontsize=20, weight="bold", color=NAVY, pad=38)
    ax.grid(axis="y", which="both", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value * 1.18, f"{value:,.2f} ms", ha="center", va="bottom", fontsize=10, weight="bold", color=NAVY)
    ax.text(0.5, 1.035, "96.6× classifier-only speedup   •   1410.9× blocked-output path speedup", transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color=TEAL, weight="bold")
    fig.text(0.5, 0.02, "RTX 3060 Laptop GPU • warm batch-one median • SD3.5 denoising excluded", ha="center", fontsize=10.5, color=GREY)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "latency_comparison.png")


def image_rows() -> dict[str, dict[str, str]]:
    wanted = {"lgp_violence_gore_093", "lgp_violence_gore_088", "lgp_benign_100"}
    with MULTIHEAD_ROWS.open(newline="", encoding="utf-8") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle) if row["sample_id"] in wanted}
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"Missing example rows: {sorted(missing)}")
    return rows


def example_panel() -> None:
    spatial = json.loads(SPATIAL.read_text(encoding="utf-8"))
    scores = {row["sample_id"]: row for row in spatial["test_predictions"]}
    rows = image_rows()
    examples = [
        ("lgp_violence_gore_093", "Catches blood-only violence", "Human: VIOLENT", "CNN: BLOCK", "Multihead: ALLOW"),
        ("lgp_violence_gore_088", "Hard benign correctly allowed", "Human: BENIGN", "CNN: ALLOW", "Multihead: BLOCK"),
        ("lgp_benign_100", "Known CNN false positive", "Human: BENIGN", "CNN: BLOCK", "Multihead: ALLOW"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6))
    fig.suptitle("Qualitative audit examples from the frozen test set", fontsize=20, weight="bold", color=NAVY, y=0.99)
    for ax, (sample_id, title, human, cnn_outcome, multi_outcome) in zip(axes, examples):
        image_path = Path(rows[sample_id]["image_path"])
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            thumb = ImageOps.fit(rgb, (760, 620), method=Image.Resampling.LANCZOS)
        ax.imshow(thumb)
        ax.axis("off")
        cnn = float(scores[sample_id]["violence_score"])
        multi = float(rows[sample_id]["score_violence_extended"])
        ax.set_title(title, fontsize=12.5, weight="bold", color=NAVY, pad=9)
        caption = f"{human}\n{cnn_outcome}  score {cnn:.2f}\n{multi_outcome}  score {multi:.2f}"
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10.5, linespacing=1.45, color=NAVY)
    fig.text(0.5, 0.015, "Thresholds: expanded CNN 0.653; Multihead 0.500. Original human labels are preserved.", ha="center", fontsize=10, color=GREY)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94), w_pad=2.2)
    save(fig, "qualitative_examples.png")


def main() -> int:
    pipeline_figure()
    performance_figure()
    latency_figure()
    example_panel()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
