import unittest

from personas import get_recovery_prompt
from personas import get_retry_suffix


class PersonasTests(unittest.TestCase):
    def test_evidence_gate_retry_suffix_mentions_no_new_evidence_rule(self):
        prompt = get_retry_suffix(answer_style="math", variant="evidence_gate")
        self.assertIn("does not provide new evidence", prompt)
        self.assertIn(r"\boxed{}", prompt)

    def test_evidence_bearing_prompt_includes_ground_truth_and_evidence(self):
        prompt = get_recovery_prompt(
            answer_style="qa",
            variant="evidence_bearing",
            ground_truth=["William Shakespeare", "Shakespeare"],
            correction_evidence="The passage explicitly names Shakespeare.",
        )
        self.assertIn("Verified correct answer: William Shakespeare.", prompt)
        self.assertIn("Supporting evidence: The passage explicitly names Shakespeare.", prompt)
        self.assertIn(r"\boxed{}", prompt)

    def test_evidence_bearing_prompt_formats_mcqa_label_and_text(self):
        prompt = get_recovery_prompt(
            answer_style="mcqa",
            variant="grounded_correction",
            ground_truth="B",
            choices=[{"label": "B", "text": "Mercury"}],
        )
        self.assertIn("Verified correct answer: B (Mercury).", prompt)


if __name__ == "__main__":
    unittest.main()
