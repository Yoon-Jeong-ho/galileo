import importlib.util
import unittest
from pathlib import Path


def _load_paper_export_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paper_export.py"
    spec = importlib.util.spec_from_file_location("paper_export", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


paper_export = _load_paper_export_module()


class PaperExportTests(unittest.TestCase):
    def test_normalize_persona_id_maps_control_aliases(self):
        self.assertEqual(
            paper_export.normalize_persona_id("Control Re-asking"),
            "neutral_reask_control",
        )
        self.assertEqual(
            paper_export.normalize_persona_id("control_reask"),
            "neutral_reask_control",
        )

    def test_first_failure_turn_uses_earliest_incorrect_turn(self):
        turns = [
            {"turn": 3, "is_correct": False},
            {"turn": 1, "is_correct": True},
            {"turn": 2, "is_correct": False},
        ]
        self.assertEqual(paper_export.first_failure_turn(turns), 2)


if __name__ == "__main__":
    unittest.main()
