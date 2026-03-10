import importlib
import os
import sys
import unittest
from unittest import mock

from data_loader import prepare_problem
from evaluation import evaluate_response, extract_math_answer
from personas import get_recovery_prompt


class ConfigTests(unittest.TestCase):
    def test_config_respects_env_overrides_and_visible_gpu_count(self):
        old_module = sys.modules.pop("config", None)
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "GALILEO_DATA_DIR": "/tmp/galileo-data",
                    "GALILEO_RESULTS_DIR": "/tmp/galileo-results",
                    "CUDA_VISIBLE_DEVICES": "7",
                },
                clear=False,
            ):
                config = importlib.import_module("config")
                self.assertEqual(config.DATA_DIR, "/tmp/galileo-data")
                self.assertEqual(config.RESULTS_DIR, "/tmp/galileo-results")
                self.assertEqual(config.TENSOR_PARALLEL_SIZE, 1)
                self.assertEqual(config.infer_tensor_parallel_size("7,8"), 2)
        finally:
            sys.modules.pop("config", None)
            if old_module is not None:
                sys.modules["config"] = old_module


class DataLoaderTests(unittest.TestCase):
    def test_prepare_problem_preserves_choices_and_correction_evidence(self):
        item = {
            "task": "mcqa",
            "question": "Which option is correct?",
            "label": "B",
            "choices": [
                {"label": "A", "text": "alpha"},
                {"label": "B", "text": "beta"},
            ],
            "supporting_facts": ["beta is supported", "alpha is contradicted"],
        }
        problem = prepare_problem(item)
        self.assertEqual(problem["ground_truth"], "B")
        self.assertEqual(problem["choices"][1]["text"], "beta")
        self.assertIn("beta is supported", problem["correction_evidence"])


class EvaluationTests(unittest.TestCase):
    def test_extract_math_answer_handles_boxed_fraction(self):
        self.assertEqual(extract_math_answer(r"Therefore the answer is \boxed{3/2}."), "1.5")

    def test_evaluate_response_mcqa_uses_boxed_label(self):
        extracted, ok, _ = evaluate_response(r"After checking, \boxed{C}", "C", answer_style="mcqa")
        self.assertEqual(extracted, "C")
        self.assertTrue(ok)


class PersonaPromptTests(unittest.TestCase):
    def test_grounded_correction_prompt_uses_answer_and_evidence(self):
        prompt = get_recovery_prompt(
            answer_style="qa",
            variant="grounded_correction",
            ground_truth=["Seoul", "서울"],
            correction_evidence="The passage explicitly states the capital is Seoul.",
        )
        self.assertIn("Verified correct answer: Seoul.", prompt)
        self.assertIn("Supporting evidence:", prompt)
        self.assertIn(r"\boxed{}", prompt)

    def test_grounded_correction_prompt_formats_mcqa_choice_text(self):
        prompt = get_recovery_prompt(
            answer_style="mcqa",
            variant="grounded_correction",
            ground_truth="B",
            choices=[
                {"label": "A", "text": "alpha"},
                {"label": "B", "text": "beta"},
            ],
        )
        self.assertIn("Verified correct answer: B (beta).", prompt)
        self.assertIn("A/B/C/D", prompt)


if __name__ == "__main__":
    unittest.main()
