import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_validator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_paper_exports.py"
    spec = importlib.util.spec_from_file_location("validate_paper_exports", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validator = _load_validator_module()


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_bundle(exports_dir: Path, *, gpu_list: str = "7", tag: str | None = None):
    exports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        exports_dir / "survival_curve.csv",
        ["persona", "round", "survived", "total", "survival_rate"],
        [{"persona": "neutral_reask_control", "round": 1, "survived": 1, "total": 1, "survival_rate": 100.0}],
    )
    _write_csv(
        exports_dir / "turn_of_failure.csv",
        ["persona", "test_name", "fail_turn", "fail_turn_label", "count", "total", "rate"],
        [{"persona": "neutral_reask_control", "test_name": "smoke", "fail_turn": 0, "fail_turn_label": "never_failed", "count": 1, "total": 1, "rate": 100.0}],
    )
    _write_csv(
        exports_dir / "flip_samples.csv",
        [
            "test_name",
            "persona",
            "fail_turn",
            "question",
            "ground_truth",
            "initial_response",
            "fail_adversarial_claim",
            "fail_model_response",
            "fail_extracted_answer",
            "taxonomy_label",
            "notes",
        ],
        [{"test_name": "smoke", "persona": "neutral_reask_control", "fail_turn": 1, "question": "q", "ground_truth": "42", "initial_response": "42", "fail_adversarial_claim": "retry", "fail_model_response": "41", "fail_extracted_answer": "41", "taxonomy_label": "", "notes": ""}],
    )
    (exports_dir / "metadata.json").write_text(json.dumps({"seed": 1}) + "\n", encoding="utf-8")
    runner_meta = {
        "generated_at": "2026-03-10T00:00:00Z",
        "gpu_list": gpu_list,
        "tensor_parallel_size": 1,
        "num_samples": 1,
        "max_model_len": 4096,
        "max_tokens": 128,
        "conda_env": "galileo",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "seed": 1,
    }
    if tag is not None:
        runner_meta["tag"] = tag
    (exports_dir / "runner_metadata.json").write_text(json.dumps(runner_meta) + "\n", encoding="utf-8")


class ValidatePaperExportsTests(unittest.TestCase):
    def test_validate_one_accepts_minimal_well_formed_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            exports_dir = Path(tmp) / "control" / "paper_exports"
            _write_bundle(exports_dir)
            errors = validator.validate_one(exports_dir, require_control=True)
            self.assertEqual(errors, [])

    def test_runner_metadata_parity_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "paper_exports"
            second = root / "b" / "paper_exports"
            _write_bundle(first, gpu_list="7", tag="smoke")
            _write_bundle(second, gpu_list="6", tag="smoke")
            errors = validator._parity_check_runner_metadata([first, second])
            self.assertEqual(len(errors), 1)
            self.assertIn("gpu_list", errors[0])


if __name__ == "__main__":
    unittest.main()
