"""Generate the latent-guard dataset through a resumable local ComfyUI batch."""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "latent_guard_pilot_400.csv"
DEFAULT_NORMALIZED = ROOT / "data" / "processed" / "prompt_sources_normalized.csv"
DEFAULT_DATA_ROOT = ROOT / "data" / "raw" / "latent_guard"
DEFAULT_RUN_ID = "latent_guard_pilot_v1"
DEFAULT_COMFY_ROOT = Path(
    os.environ.get(
        "FIT5230_COMFY_ROOT",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Comfy-Desktop"
        / "ComfyUI-Installs"
        / "Assignment"
        / "ComfyUI",
    )
)
DEFAULT_SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
DEFAULT_MODEL = ROOT / "models" / "shieldgemma2" / "shieldgemma-2-4b-it-Q4_K_M.gguf"
DEFAULT_MMPROJ = ROOT / "models" / "shieldgemma2" / "mmproj-shieldgemma-2-4b-it-F16.gguf"
CHECKPOINT = "sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors"
GENERATION = {
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg": 3.5,
    "sampler_name": "euler",
    "scheduler": "beta",
    "denoise": 1.0,
    "negative_prompt": "",
    "vae_decode": "tiled",
    "vae_tile_size": 512,
    "vae_overlap": 64,
}


def shield_server_healthy(url: str = "http://127.0.0.1:8190") -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_shield_server(log_path: Path) -> tuple[subprocess.Popen, Any] | None:
    """Start one persistent CPU teacher server, unless an external one is healthy."""

    if shield_server_healthy():
        return None
    for path in (DEFAULT_SERVER, DEFAULT_MODEL, DEFAULT_MMPROJ):
        if not path.is_file():
            raise FileNotFoundError(f"ShieldGemma dependency not found: {path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        str(DEFAULT_SERVER),
        "--model",
        str(DEFAULT_MODEL),
        "--mmproj",
        str(DEFAULT_MMPROJ),
        "--host",
        "127.0.0.1",
        "--port",
        "8190",
        "--ctx-size",
        "4096",
        "--threads",
        "8",
        "--n-gpu-layers",
        "0",
        "--parallel",
        "1",
        "--no-webui",
    ]
    process = subprocess.Popen(
        command,
        cwd=DEFAULT_SERVER.parent,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"ShieldGemma server exited with code {process.returncode}")
        if shield_server_healthy():
            return process, log_handle
        time.sleep(1)
    process.terminate()
    log_handle.close()
    raise TimeoutError("ShieldGemma server did not become healthy within 180 seconds")


def stop_owned_server(owned: tuple[subprocess.Popen, Any] | None) -> None:
    if owned is None:
        return
    process, log_handle = owned
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
    log_handle.close()


def api_json(base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}", data=body, headers=headers
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def repository_graph() -> dict[str, Any]:
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": CHECKPOINT},
        },
        "16": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["4", 1]},
        },
        "40": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": GENERATION["negative_prompt"], "clip": ["4", 1]},
        },
        "53": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": GENERATION["width"],
                "height": GENERATION["height"],
                "batch_size": 1,
            },
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["16", 0],
                "negative": ["40", 0],
                "latent_image": ["53", 0],
                "seed": 0,
                "steps": GENERATION["steps"],
                "cfg": GENERATION["cfg"],
                "sampler_name": GENERATION["sampler_name"],
                "scheduler": GENERATION["scheduler"],
                "denoise": GENERATION["denoise"],
            },
        },
        "8": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
                "tile_size": GENERATION["vae_tile_size"],
                "overlap": GENERATION["vae_overlap"],
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
    }


def load_rows(manifest_path: Path, normalized_path: Path) -> list[dict[str, str]]:
    with normalized_path.open(encoding="utf-8", newline="") as handle:
        normalized = {row["record_id"]: row for row in csv.DictReader(handle)}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))

    resolved: list[dict[str, str]] = []
    for row in manifest:
        source = normalized.get(row["record_id"])
        if source is None:
            raise RuntimeError(f"Manifest record not found in normalized table: {row['record_id']}")
        if source["prompt_sha256"] != row["prompt_sha256"]:
            raise RuntimeError(f"Prompt hash mismatch for {row['sample_id']}")
        resolved.append({**row, "prompt": source["prompt"]})
    return resolved


def prepare_graph(
    row: dict[str, str],
    run_id: str,
    latent_root: Path,
    decision_log: Path,
    use_guard: bool,
    threshold: float,
) -> dict[str, Any]:
    graph = copy.deepcopy(repository_graph())
    graph["16"]["inputs"]["text"] = row["prompt"]
    graph["3"]["inputs"]["seed"] = int(row["seed"])
    graph["60"] = {
        "class_type": "FIT5230FinalLatentSaver",
        "inputs": {
            "samples": ["3", 0],
            "output_dir": str(latent_root),
            "run_id": run_id,
            "sample_id": row["sample_id"],
            "seed": int(row["seed"]),
            "prompt_sha256": row["prompt_sha256"],
            "checkpoint_id": CHECKPOINT,
            "overwrite": False,
        },
    }
    image_input = ["8", 0]
    if use_guard:
        graph["61"] = {
            "class_type": "FIT5230ShieldGemmaLoader",
            "inputs": {
                "server_exe": str(DEFAULT_SERVER),
                "model_file": str(DEFAULT_MODEL),
                "mmproj_file": str(DEFAULT_MMPROJ),
                "server_url": "http://127.0.0.1:8190",
                "threads": max(1, (os.cpu_count() or 8) // 2),
                "startup_timeout": 180,
            },
        }
        graph["62"] = {
            "class_type": "FIT5230ShieldGemmaGuard",
            "inputs": {
                "images": ["8", 0],
                "guard": ["61", 0],
                "threshold": threshold,
                "action": "report_only",
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "log_file": str(decision_log),
            },
        }
        image_input = ["62", 0]
    graph["63"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": image_input,
            "filename_prefix": (
                f"FIT5230/{run_id}/{row['split']}/{row['sample_id']}/"
                f"seed_{row['seed']}"
            ),
        },
    }
    return graph


def wait_for_completion(base_url: str, prompt_id: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = api_json(base_url, f"/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in {"success", "error"}:
                return entry
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not complete {prompt_id} within {timeout}s")


def image_paths(entry: dict[str, Any], output_root: Path) -> list[Path]:
    results: list[Path] = []
    for output in entry.get("outputs", {}).values():
        for image in output.get("images", []):
            results.append(output_root / image.get("subfolder", "") / image["filename"])
    return results


def latest_decision(path: Path, run_id: str, sample_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("run_id") == run_id and item.get("sample_id") == sample_id:
                latest = item
    return latest


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Completion record already exists: {path}")
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_record(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("status") == "success"
    except (OSError, json.JSONDecodeError):
        return False


def process_is_alive(pid: int) -> bool:
    """Return whether a process ID exists without changing that process."""

    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(run_root: Path) -> Path:
    """Atomically prevent multiple collectors writing to the same run ID."""

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".runner.lock"
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner_pid = -1
            if process_is_alive(owner_pid):
                raise RuntimeError(
                    f"Run {run_root.name!r} is already owned by active PID {owner_pid}"
                )
            lock_path.unlink(missing_ok=True)
            continue
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
        return lock_path


def release_run_lock(lock_path: Path) -> None:
    try:
        owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if owner_pid == os.getpid():
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--comfy-output-root", type=Path, default=DEFAULT_COMFY_ROOT / "output")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split", action="append", choices=("train", "validation", "test"))
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--skip-shieldgemma", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.manifest.resolve(), args.normalized.resolve())
    if args.split:
        rows = [row for row in rows if row["split"] in set(args.split)]
    if args.sample:
        rows = [row for row in rows if row["sample_id"] in set(args.sample)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No manifest rows matched the requested filters")

    run_root = args.data_root.resolve() / args.run_id
    run_lock = None if args.dry_run else acquire_run_lock(run_root)
    if run_lock is not None:
        atexit.register(release_run_lock, run_lock)
    latent_root = args.data_root.resolve() / "latents"
    decision_log = run_root / "shieldgemma_decisions.jsonl"
    journal = run_root / "attempts.jsonl"
    record_root = run_root / "records"
    owned_shield_server = None
    if not args.dry_run:
        api_json(args.url, "/system_stats")
        if not args.skip_shieldgemma:
            owned_shield_server = start_shield_server(run_root / "shieldgemma_server.log")
            atexit.register(stop_owned_server, owned_shield_server)

    client_id = str(uuid.uuid4())
    failures = skipped = completed = 0
    for index, row in enumerate(rows, start=1):
        record_path = record_root / f"{row['sample_id']}.json"
        if completed_record(record_path):
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP {row['sample_id']}", flush=True)
            continue
        if args.dry_run:
            print(f"[{index}/{len(rows)}] PLAN {row['sample_id']}", flush=True)
            continue

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        prompt_id = ""
        try:
            graph = prepare_graph(
                row,
                args.run_id,
                latent_root,
                decision_log,
                not args.skip_shieldgemma,
                args.threshold,
            )
            response = api_json(args.url, "/prompt", {"prompt": graph, "client_id": client_id})
            prompt_id = response["prompt_id"]
            print(f"[{index}/{len(rows)}] RUN  {row['sample_id']} ({prompt_id})", flush=True)
            entry = wait_for_completion(args.url, prompt_id, args.timeout)
            status = entry.get("status", {}).get("status_str", "unknown")
            latent_path = latent_root / args.run_id / row["sample_id"] / f"seed_{row['seed']}.npz"
            images = image_paths(entry, args.comfy_output_root.resolve())
            decision = (
                None
                if args.skip_shieldgemma
                else latest_decision(decision_log, args.run_id, row["sample_id"])
            )
            valid = (
                status == "success"
                and latent_path.is_file()
                and len(images) == 1
                and images[0].is_file()
                and (args.skip_shieldgemma or decision is not None)
            )
            final_status = "success" if valid else "incomplete"
            result = {
                "schema_version": 1,
                "status": final_status,
                "run_id": args.run_id,
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "sample": row,
                "generation": {"checkpoint": CHECKPOINT, **GENERATION},
                "comfy": {
                    "url": args.url,
                    "prompt_id": prompt_id,
                    "status": status,
                    "image_files": [str(path) for path in images],
                },
                "latent_file": str(latent_path),
                "shieldgemma": decision,
            }
            append_jsonl(journal, result)
            if valid:
                write_record(record_path, result)
                completed += 1
                guard_time = decision.get("elapsed_seconds") if decision else "skipped"
                print(f"                 success; guard_seconds={guard_time}", flush=True)
            else:
                failures += 1
                print("                 INCOMPLETE: required artifacts missing", flush=True)
        except (OSError, TimeoutError, urllib.error.URLError, KeyError, RuntimeError) as error:
            failures += 1
            append_jsonl(
                journal,
                {
                    "schema_version": 1,
                    "status": "error",
                    "run_id": args.run_id,
                    "sample_id": row["sample_id"],
                    "started_at_utc": started_at.isoformat(),
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "comfy_prompt_id": prompt_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(f"                 ERROR: {error}", flush=True)

    print(f"Done: completed={completed}, skipped={skipped}, failed={failures}", flush=True)
    stop_owned_server(owned_shield_server)
    if run_lock is not None:
        release_run_lock(run_lock)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
