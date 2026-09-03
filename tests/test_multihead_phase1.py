import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_derivation_uses_strict_published_threshold():
    runner = load_script("run_multihead_baseline")
    sample = {"sample_id": "x", "split": "test", "candidate_policy": "benign", "human_output_label": "benign", "image_path": "image.png"}
    row = runner.score_row(sample, {"sexual": 0.5, "violent": 0.51, "disturbing": 0.1, "hateful": 0.2, "political": 0.3}, run_id="run", device="cpu", batch_size=4, batch_seconds=1, image_seconds=.5)
    assert row["flag_sexual"] == 0
    assert row["flag_any"] == row["flag_violence_extended"] == 1
    assert row["score_any"] == 0.51


def test_error_rows_never_become_primary_safe_predictions():
    runner = load_script("run_multihead_baseline")
    sample = {"sample_id": "x", "split": "test", "candidate_policy": "benign", "human_output_label": "benign", "image_path": "image.png"}
    row = runner.error_row(sample, "image_load_error", "bad image", run_id="run", device="cpu", batch_size=4)
    assert row["status"] == "error"
    assert row["flag_any"] == ""
    assert row["paper_compatible_flagged"] == 0


def test_incremental_output_is_resumable_by_sample_id(tmp_path):
    runner = load_script("run_multihead_baseline")
    sample = {"sample_id": "x", "split": "test", "candidate_policy": "benign", "human_output_label": "benign", "image_path": "image.png"}
    row = runner.error_row(sample, "image_load_error", "bad image", run_id="run", device="cpu", batch_size=4)
    output = tmp_path / "predictions.csv"
    runner.write_rows(output, [row])
    import csv
    with output.open(newline="", encoding="utf-8") as handle:
        restored = list(csv.DictReader(handle))
    assert {item["sample_id"] for item in restored} == {"x"}


def test_metric_tasks_have_expected_fixed_test_membership():
    evaluator = load_script("evaluate_multihead_baseline")
    rows = [
        {"sample_id": "a", "split": "test", "human_output_label": "benign", "score_any": "0.1", "flag_any": "0", "score_violence_extended": "0.1", "flag_violence_extended": "0", "score_violent": "0.1", "flag_violence_strict": "0"},
        {"sample_id": "b", "split": "test", "human_output_label": "violence_gore", "score_any": "0.9", "flag_any": "1", "score_violence_extended": "0.9", "flag_violence_extended": "1", "score_violent": "0.8", "flag_violence_strict": "1"},
        {"sample_id": "c", "split": "test", "human_output_label": "sexual", "score_any": "0.9", "flag_any": "1", "score_violence_extended": "0.1", "flag_violence_extended": "0", "score_violent": "0.1", "flag_violence_strict": "0"},
    ]
    _, _, _, any_ids = evaluator.valid_task_rows(rows, "any_harm")
    _, _, _, violence_ids = evaluator.valid_task_rows(rows, "violence_extended")
    assert any_ids == ["a", "b", "c"]
    assert violence_ids == ["a", "b"]
