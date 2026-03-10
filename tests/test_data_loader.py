import unittest

from data_loader import prepare_problem


class DataLoaderTests(unittest.TestCase):
    def test_prepare_problem_preserves_correction_evidence_aliases(self):
        item = {
            "task": "qa",
            "question": "Who wrote Hamlet?",
            "answers": ["William Shakespeare"],
            "supporting_facts": ["A Shakespeare play", "Tragedy"],
        }
        prepared = prepare_problem(item)
        self.assertEqual(prepared["ground_truth"], ["William Shakespeare"])
        self.assertEqual(prepared["correction_evidence"], "A Shakespeare play\nTragedy")

    def test_prepare_problem_mcqa_uses_label(self):
        item = {
            "task": "mcqa",
            "question": "Pick one",
            "label": "b",
            "choices": [{"label": "B", "text": "Option B"}],
        }
        prepared = prepare_problem(item)
        self.assertEqual(prepared["ground_truth"], "B")
        self.assertEqual(prepared["choices"][0]["text"], "Option B")


if __name__ == "__main__":
    unittest.main()
