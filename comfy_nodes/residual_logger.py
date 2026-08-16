"""ComfyUI node for logging classifier-free-guidance residual trajectories.

The node patches a copy of a ComfyUI MODEL and observes the conditional and
unconditional denoised predictions at every sampler step. It returns the
standard CFG result unchanged, so attaching the logger must not alter image
generation.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


DEFAULT_OUTPUT_DIR = (
    r"C:\Users\raexc\OneDrive\Documents\FIT5230-SLD\data\raw\grsp_pilot\features"
)


def _safe_component(value: str, fallback: str) -> str:
    """Return a filesystem-safe experiment identifier."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:120] or fallback


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Write an NPZ atomically so an interrupted sampler leaves no corrupt file."""

    temporary_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, path)


class GuidanceResidualLogger:
    """Attach a passive per-step CFG-residual logger to a ComfyUI model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "run_id": ("STRING", {"default": "grsp_pilot_01"}),
                "prompt_id": ("STRING", {"default": "prompt_0001"}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "class_label": (["unknown", "benign", "unsafe", "uncertain"],),
                "attack_type": (["benign", "direct", "DACA", "PGJ", "other"],),
                "pool_size": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "output_dir": ("STRING", {"default": DEFAULT_OUTPUT_DIR}),
                "save_enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "feature_file")
    FUNCTION = "attach_logger"
    CATEGORY = "FIT5230/Safety"

    def attach_logger(
        self,
        model,
        run_id: str,
        prompt_id: str,
        seed: int,
        class_label: str,
        attack_type: str,
        pool_size: int,
        output_dir: str,
        save_enabled: bool,
    ):
        patched_model = model.clone()

        safe_run_id = _safe_component(run_id, "run")
        safe_prompt_id = _safe_component(prompt_id, "prompt")
        started_at = datetime.now(timezone.utc)
        timestamp = started_at.strftime("%Y%m%dT%H%M%S_%fZ")

        destination = Path(output_dir).expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        destination = destination.resolve() / safe_run_id / safe_prompt_id
        feature_path = destination / f"seed_{seed}_{timestamp}.npz"

        state: dict[str, Any] = {
            "pooled": [],
            "statistics": [],
            "sigmas": [],
            "cond_scales": [],
            "step_indices": [],
            "residual_shape": None,
        }
        state_lock = threading.Lock()

        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "prompt_id": prompt_id,
            "seed": int(seed),
            "class_label": class_label,
            "attack_type": attack_type,
            "pool_size": int(pool_size),
            "created_at_utc": started_at.isoformat(),
            "feature_definition": "cond_denoised - uncond_denoised",
            "cfg_behavior": "passive_standard_cfg",
        }

        def save_state() -> None:
            if not save_enabled or not state["pooled"]:
                return

            destination.mkdir(parents=True, exist_ok=True)
            _atomic_savez(
                feature_path,
                pooled_residuals=np.stack(state["pooled"]).astype(np.float16),
                statistics=np.asarray(state["statistics"], dtype=np.float32),
                sigmas=np.asarray(state["sigmas"], dtype=np.float32),
                cond_scales=np.asarray(state["cond_scales"], dtype=np.float32),
                step_indices=np.asarray(state["step_indices"], dtype=np.int32),
                residual_shape=np.asarray(state["residual_shape"], dtype=np.int32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )

        def cfg_logger(args: dict[str, Any]):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]

            # This is the prompt-induced direction in denoised-prediction space.
            residual = (cond_denoised - uncond_denoised).detach().float()
            if residual.ndim != 4:
                raise ValueError(
                    "Guidance Residual Logger expected a BCHW prediction tensor, "
                    f"but received shape {tuple(residual.shape)}"
                )

            pooled = functional.adaptive_avg_pool2d(residual, (pool_size, pool_size))
            flattened = residual.flatten(1)
            rms = torch.sqrt(torch.mean(flattened.square(), dim=1))
            l2_per_element = torch.linalg.vector_norm(flattened, dim=1) / math.sqrt(
                flattened.shape[1]
            )
            statistics = torch.stack(
                (
                    flattened.mean(dim=1),
                    flattened.std(dim=1, unbiased=False),
                    flattened.abs().mean(dim=1),
                    rms,
                    l2_per_element,
                    flattened.amin(dim=1),
                    flattened.amax(dim=1),
                ),
                dim=1,
            )

            sigma_tensor = torch.as_tensor(args["sigma"]).detach().float().flatten()
            cond_scale = float(args["cond_scale"])

            with state_lock:
                step_index = len(state["pooled"])
                state["pooled"].append(pooled.cpu().numpy())
                state["statistics"].append(statistics.cpu().numpy())
                state["sigmas"].append(sigma_tensor.cpu().numpy())
                state["cond_scales"].append(cond_scale)
                state["step_indices"].append(step_index)
                state["residual_shape"] = list(residual.shape)
                save_state()

            # ComfyUI's custom CFG hook expects the noise-space form. Returning
            # this expression produces exactly the default denoised CFG result.
            return args["uncond"] + (args["cond"] - args["uncond"]) * cond_scale

        patched_model.set_model_sampler_cfg_function(
            cfg_logger,
            disable_cfg1_optimization=True,
        )

        feature_file = str(feature_path) if save_enabled else "Saving disabled"
        return (patched_model, feature_file)


NODE_CLASS_MAPPINGS = {
    "GuidanceResidualLogger": GuidanceResidualLogger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GuidanceResidualLogger": "Guidance Residual Logger (FIT5230)",
}
