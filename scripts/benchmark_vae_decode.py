"""Benchmark SD3.5 tiled VAE decoding through the local ComfyUI API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "data" / "raw" / "latent_guard" / "latent_guard_pilot_v1" / "records"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "vae_decode_latency_v1.json"
CHECKPOINT = "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"


def api_json(url: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def select_latent(records: Path) -> Path:
    for record_path in sorted(records.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        latent = Path(record.get("latent_file", ""))
        if record.get("status") == "success" and latent.is_file():
            return latent
    raise FileNotFoundError(f"No completed latent was found under {records}")


def graph(latent: Path, output: Path, warmups: int, repeats: int) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "2": {
            "class_type": "FIT5230VAEDecodeBenchmark",
            "inputs": {
                "vae": ["1", 2],
                "latent_file": str(latent),
                "tile_size": 512,
                "overlap": 64,
                "warmup_iterations": warmups,
                "timed_iterations": repeats,
                "output_json": str(output),
            },
        },
    }


def wait(url: str, prompt_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = api_json(url, f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not finish benchmark {prompt_id} within {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--latent", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    latent = args.latent.resolve() if args.latent else select_latent(args.records.resolve())
    output = args.output.resolve()
    api_json(args.url, "/system_stats")
    response = api_json(
        args.url,
        "/prompt",
        {"prompt": graph(latent, output, args.warmups, args.repeats), "client_id": str(uuid.uuid4())},
    )
    prompt_id = response["prompt_id"]
    entry = wait(args.url, prompt_id, args.timeout)
    status = entry.get("status", {}).get("status_str")
    if status != "success" or not output.is_file():
        raise RuntimeError(f"VAE benchmark failed: status={status}; history={entry}")
    report = json.loads(output.read_text(encoding="utf-8"))
    try:
        report["latent_file"] = str(Path(report["latent_file"]).resolve().relative_to(ROOT))
    except (KeyError, ValueError):
        pass
    report["comfy_prompt_id"] = prompt_id
    report["checkpoint"] = CHECKPOINT
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as error:
        raise SystemExit(f"ComfyUI API is unavailable: {error}") from error
