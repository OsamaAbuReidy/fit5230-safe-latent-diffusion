"""ComfyUI node for reproducibly saving final SD3.5 latent tensors."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_ROOT = Path(
    os.environ.get("FIT5230_SLD_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DEFAULT_OUTPUT = DEFAULT_ROOT / "data" / "raw" / "latent_guard" / "latents"


def _safe_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:120] or fallback


def _atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class FIT5230FinalLatentSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "output_dir": ("STRING", {"default": str(DEFAULT_OUTPUT)}),
                "run_id": ("STRING", {"default": "latent_guard_pilot_v1"}),
                "sample_id": ("STRING", {"default": "sample_0001"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "prompt_sha256": ("STRING", {"default": ""}),
                "checkpoint_id": ("STRING", {"default": "sd3.5_medium"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "latent_file")
    FUNCTION = "save"
    CATEGORY = "FIT5230/Data"
    OUTPUT_NODE = True

    def save(
        self,
        samples,
        output_dir,
        run_id,
        sample_id,
        seed,
        prompt_sha256,
        checkpoint_id,
        overwrite,
    ):
        if "samples" not in samples:
            raise KeyError("LATENT input does not contain a 'samples' tensor")
        tensor = samples["samples"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise TypeError("Expected a BCHW latent tensor")

        root = Path(output_dir).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        destination = (
            root.resolve()
            / _safe_component(run_id, "run")
            / _safe_component(sample_id, "sample")
        )
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"seed_{int(seed)}.npz"
        if path.exists() and not overwrite:
            return samples, str(path)

        latent = tensor.detach().to(device="cpu", dtype=torch.float16).numpy()
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "sample_id": sample_id,
            "seed": int(seed),
            "prompt_sha256": prompt_sha256,
            "checkpoint_id": checkpoint_id,
            "tensor_key": "latent",
            "shape": list(latent.shape),
            "stored_dtype": "float16",
            "comfy_latent_stage": "KSampler final output before VAE decode",
        }
        _atomic_savez(
            path,
            latent=latent,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return samples, str(path)


class FIT5230VAEDecodeBenchmark:
    """Measure SD3.5 tiled VAE decoding independently of denoising."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent_file": ("STRING", {"default": ""}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "warmup_iterations": ("INT", {"default": 1, "min": 0, "max": 10}),
                "timed_iterations": ("INT", {"default": 5, "min": 1, "max": 100}),
                "output_json": (
                    "STRING",
                    {"default": str(DEFAULT_ROOT / "data" / "results" / "vae_decode_latency_v1.json")},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "benchmark"
    CATEGORY = "FIT5230/Evaluation"
    OUTPUT_NODE = True

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def benchmark(
        self,
        vae,
        latent_file,
        tile_size,
        overlap,
        warmup_iterations,
        timed_iterations,
        output_json,
    ):
        source = Path(latent_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Latent archive does not exist: {source}")
        with np.load(source, allow_pickle=False) as archive:
            latent = np.asarray(archive["latent"], dtype=np.float32)
        if latent.shape != (1, 16, 128, 128):
            raise ValueError(f"Expected SD3.5 1024px latent [1,16,128,128], got {latent.shape}")

        samples = torch.from_numpy(np.ascontiguousarray(latent))
        compression = vae.spacial_compression_decode()
        tile_x = tile_size // compression
        tile_y = tile_size // compression
        tile_overlap = overlap // compression

        def decode_once():
            return vae.decode_tiled(
                samples,
                tile_x=tile_x,
                tile_y=tile_y,
                overlap=tile_overlap,
                tile_t=None,
                overlap_t=8,
            )

        for _ in range(warmup_iterations):
            decode_once()
        self._sync()

        timings: list[float] = []
        for _ in range(timed_iterations):
            self._sync()
            started = time.perf_counter()
            decode_once()
            self._sync()
            timings.append((time.perf_counter() - started) * 1000)

        values = np.asarray(timings, dtype=np.float64)
        report = {
            "schema_version": 1,
            "benchmark": "ComfyUI VAEDecodeTiled model call",
            "latent_file": str(source),
            "latent_shape": list(latent.shape),
            "tile_size": int(tile_size),
            "overlap": int(overlap),
            "warmup_iterations": int(warmup_iterations),
            "timed_iterations": int(timed_iterations),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "latency_ms": {
                "mean": round(float(values.mean()), 4),
                "median": round(float(np.median(values)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
                "minimum": round(float(values.min()), 4),
                "maximum": round(float(values.max()), 4),
            },
            "timings_ms": [round(float(value), 4) for value in values],
            "scope_note": "Denoising, latent loading, PNG encoding, and image-classifier inference are excluded.",
        }
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        summary = json.dumps(report["latency_ms"], sort_keys=True)
        return {"ui": {"text": [summary]}, "result": (summary,)}


NODE_CLASS_MAPPINGS = {
    "FIT5230FinalLatentSaver": FIT5230FinalLatentSaver,
    "FIT5230VAEDecodeBenchmark": FIT5230VAEDecodeBenchmark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FIT5230FinalLatentSaver": "Final Latent Saver (FIT5230)",
    "FIT5230VAEDecodeBenchmark": "VAE Decode Benchmark (FIT5230)",
}
