import unittest

from evaluation import evaluate_response, extract_math_answer


class EvaluationTests(unittest.TestCase):
    def test_math_fraction_normalization(self):
        self.assertEqual(extract_math_answer(r"The answer is \boxed{1/2}."), "0.5")
        extracted, ok, _ = evaluate_response(r"\boxed{1/2}", "0.5", answer_style="math")
        self.assertEqual(extracted, "0.5")
        self.assertTrue(ok)

    def test_qa_boxed_answer(self):
        extracted, ok, scores = evaluate_response(
            r"Reasoning...\n\boxed{William Shakespeare}",
            ["William Shakespeare", "Shakespeare"],
            answer_style="qa",
        )
        self.assertEqual(extracted, "William Shakespeare")
        self.assertTrue(ok)
        self.assertEqual(scores["em"], 1.0)

    def test_mcqa_extracts_label(self):
        extracted, ok, _ = evaluate_response(
            r"After checking, \boxed{C}",
            "C",
            answer_style="mcqa",
        )
        self.assertEqual(extracted, "C")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
