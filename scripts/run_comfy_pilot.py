"""Queue a reproducible FIT5230 prompt pilot through the local ComfyUI API."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = REPOSITORY_ROOT / "data" / "manifests" / "grsp_pilot_02_prompts.csv"
DEFAULT_FEATURE_ROOT = REPOSITORY_ROOT / "data" / "raw" / "grsp_pilot" / "features"
DEFAULT_RUN_MANIFEST = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "grsp_pilot"
    / "manifests"
    / "grsp_pilot_02_runs.csv"
)
DEFAULT_SEEDS = (
    225174911108442,
    225174911108443,
    225174911108444,
    225174911108445,
)
RUN_MANIFEST_FIELDS = (
    "queued_at_utc",
    "pair_id",
    "prompt_id",
    "seed",
    "class_label",
    "attack_type",
    "comfy_prompt_id",
    "status",
    "feature_file",
    "image_files",
)


def api_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def find_latest_template(base_url: str) -> dict[str, Any]:
    history = api_json(base_url, "/history?max_items=30")
    required = {"KSampler", "GuidanceResidualLogger", "SaveImage"}
    for entry in history.values():
        if entry.get("status", {}).get("status_str") != "success":
            continue
        graph = entry.get("prompt", [None, None, {}])[2]
        classes = {node.get("class_type") for node in graph.values()}
        if required.issubset(classes):
            return copy.deepcopy(graph)
    raise RuntimeError(
        "No successful ComfyUI history entry contains KSampler, "
        "GuidanceResidualLogger, and SaveImage. Generate one image in the UI first."
    )


def single_node(graph: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [(node_id, node) for node_id, node in graph.items() if node.get("class_type") == class_type]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {class_type} node, found {len(matches)}")
    return matches[0]


def conditioning_node(graph: dict[str, Any], sampler: dict[str, Any], input_name: str) -> dict[str, Any]:
    link = sampler["inputs"][input_name]
    if not isinstance(link, list) or not link:
        raise RuntimeError(f"KSampler {input_name} input is not linked to a node")
    node = graph[str(link[0])]
    if node.get("class_type") != "CLIPTextEncode":
        raise RuntimeError(f"KSampler {input_name} is not supplied by CLIPTextEncode")
    return node


def neutralize_negative_prompt(text: str) -> str:
    """Remove the experiment-confounding standalone negative term 'clothes'."""

    terms = [term.strip() for term in text.replace("\n", " ").split(",")]
    return ", ".join(term for term in terms if term and term.casefold() != "clothes")


def load_prompt_rows(
    path: Path,
    group_filter: set[str],
    attack_filter: set[str],
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    group_field = "pair_id" if rows and "pair_id" in rows[0] else "target_id"
    if group_filter:
        rows = [row for row in rows if row[group_field] in group_filter]
    if attack_filter:
        rows = [row for row in rows if row["attack_type"] in attack_filter]
    if not rows:
        raise RuntimeError("The selected prompt manifest contains no matching rows")
    return rows


def existing_feature(feature_root: Path, run_id: str, prompt_id: str, seed: int) -> Path | None:
    directory = feature_root / run_id / prompt_id
    matches = sorted(directory.glob(f"seed_{seed}_*.npz"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def image_paths(history_entry: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for output in history_entry.get("outputs", {}).values():
        for image in output.get("images", []):
            subfolder = image.get("subfolder", "")
            filename = image.get("filename", "")
            paths.append(str(Path(subfolder) / filename) if subfolder else filename)
    return paths


def wait_for_completion(base_url: str, prompt_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = api_json(base_url, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in {"success", "error"}:
                return entry
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not finish prompt {prompt_id} within {timeout_seconds} seconds")


def append_run_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def prepare_graph(
    template: dict[str, Any],
    row: dict[str, str],
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    graph = copy.deepcopy(template)
    _, sampler = single_node(graph, "KSampler")
    _, logger = single_node(graph, "GuidanceResidualLogger")
    _, saver = single_node(graph, "SaveImage")
    positive = conditioning_node(graph, sampler, "positive")
    negative = conditioning_node(graph, sampler, "negative")

    positive["inputs"]["text"] = row["positive_prompt"]
    negative["inputs"]["text"] = neutralize_negative_prompt(negative["inputs"]["text"])
    sampler["inputs"]["seed"] = seed

    logger["inputs"].update(
        {
            "run_id": run_id,
            "prompt_id": row["prompt_id"],
            "seed": seed,
            "class_label": row["class_label"],
            "attack_type": row["attack_type"],
            "pool_size": 8,
            "save_enabled": True,
        }
    )
    saver["inputs"]["filename_prefix"] = (
        f"FIT5230/{run_id}/{row['prompt_id']}/seed_{seed}"
    )
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--run-id", default="grsp_pilot_02")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument(
        "--pair",
        "--group",
        dest="group",
        action="append",
        default=[],
        help="Run only this pair_id or target_id; repeat as needed",
    )
    parser.add_argument(
        "--attack-type",
        action="append",
        default=[],
        help="Run only this attack_type; repeat as needed",
    )
    parser.add_argument("--seed", action="append", type=int, default=[], help="Override seed list; repeat as needed")
    parser.add_argument("--timeout", type=int, default=900, help="Per-image timeout in seconds")
    parser.add_argument("--force", action="store_true", help="Regenerate samples whose feature file already exists")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.url.rstrip("/")
    api_json(base_url, "/system_stats")
    template = find_latest_template(base_url)
    rows = load_prompt_rows(args.prompts.resolve(), set(args.group), set(args.attack_type))
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS

    jobs = [(row, seed) for row in rows for seed in seeds]
    print(f"ComfyUI: {base_url}")
    print(f"Run: {args.run_id}; planned jobs: {len(jobs)}")

    client_id = str(uuid.uuid4())
    queued = skipped = failed = 0
    for index, (row, seed) in enumerate(jobs, start=1):
        feature = existing_feature(args.feature_root, args.run_id, row["prompt_id"], seed)
        label = f"{row['prompt_id']} seed={seed}"
        if feature is not None and not args.force:
            skipped += 1
            print(f"[{index}/{len(jobs)}] SKIP {label}: {feature.name}")
            continue

        graph = prepare_graph(template, row, seed, args.run_id)
        if args.dry_run:
            print(f"[{index}/{len(jobs)}] PLAN {label}")
            continue

        queued_at = datetime.now(timezone.utc).isoformat()
        response = api_json(base_url, "/prompt", {"prompt": graph, "client_id": client_id})
        comfy_prompt_id = response["prompt_id"]
        queued += 1
        print(f"[{index}/{len(jobs)}] RUN  {label} ({comfy_prompt_id})")

        try:
            entry = wait_for_completion(base_url, comfy_prompt_id, args.timeout)
            status = entry.get("status", {}).get("status_str", "unknown")
            feature = existing_feature(args.feature_root, args.run_id, row["prompt_id"], seed)
            images = image_paths(entry)
            if status != "success" or feature is None:
                failed += 1
            append_run_manifest(
                args.run_manifest,
                {
                    "queued_at_utc": queued_at,
                    "pair_id": row.get("pair_id", row.get("target_id", "")),
                    "prompt_id": row["prompt_id"],
                    "seed": seed,
                    "class_label": row["class_label"],
                    "attack_type": row["attack_type"],
                    "comfy_prompt_id": comfy_prompt_id,
                    "status": status if feature is not None else f"{status}:missing_feature",
                    "feature_file": str(feature) if feature else "",
                    "image_files": "|".join(images),
                },
            )
            print(f"                 {status}; feature={'yes' if feature else 'NO'}; images={len(images)}")
        except (TimeoutError, urllib.error.URLError) as error:
            failed += 1
            print(f"                 ERROR: {error}")

    print(f"Done: queued={queued}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
