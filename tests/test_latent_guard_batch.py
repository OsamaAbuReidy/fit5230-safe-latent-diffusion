from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_latent_guard_batch.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_latent_guard_batch", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_manifest_prompts_resolve_and_match_hashes() -> None:
    module = load_script_module()
    rows = module.load_rows(module.DEFAULT_MANIFEST, module.DEFAULT_NORMALIZED)
    assert len(rows) == 400
    assert all(row["prompt"] for row in rows)


def test_graph_captures_latent_image_and_teacher_decision(tmp_path: Path) -> None:
    module = load_script_module()
    row = module.load_rows(module.DEFAULT_MANIFEST, module.DEFAULT_NORMALIZED)[0]
    graph = module.prepare_graph(row, "test_run", tmp_path, tmp_path / "decisions.jsonl", True, 0.5)

    assert graph["60"]["class_type"] == "FIT5230FinalLatentSaver"
    assert graph["60"]["inputs"]["samples"] == ["3", 0]
    assert graph["62"]["class_type"] == "FIT5230ShieldGemmaGuard"
    assert graph["62"]["inputs"]["action"] == "report_only"
    assert graph["63"]["class_type"] == "SaveImage"
    assert graph["8"]["class_type"] == "VAEDecodeTiled"
    assert graph["8"]["inputs"]["tile_size"] == 512
    assert graph["3"]["inputs"]["seed"] == int(row["seed"])
    assert graph["16"]["inputs"]["text"] == row["prompt"]


def test_graph_can_skip_teacher_for_generation_only(tmp_path: Path) -> None:
    module = load_script_module()
    row = module.load_rows(module.DEFAULT_MANIFEST, module.DEFAULT_NORMALIZED)[0]
    graph = module.prepare_graph(row, "test_run", tmp_path, tmp_path / "decisions.jsonl", False, 0.5)

    classes = {node["class_type"] for node in graph.values()}
    assert "FIT5230FinalLatentSaver" in classes
    assert "FIT5230ShieldGemmaGuard" not in classes
    assert graph["63"]["inputs"]["images"] == ["8", 0]
