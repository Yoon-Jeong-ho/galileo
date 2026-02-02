"""Answer evaluation.

Supports answer styles:
- math: numeric normalization / exact match.
- qa: SQuAD-style EM/F1 with alias list.
- mcqa: label extraction + exact match (A/B/C/D).

We intentionally avoid regex word-boundary (\b) patterns because some
transports may treat \b as a control character.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional, Tuple, List, Dict, Union

from config import ANSWER_PATTERNS


def normalize_number(text: str) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace(" ", "")
    text = text.replace("%", "")

    try:
        if "/" in text:
            parts = text.split("/")
            if len(parts) == 2:
                result = float(parts[0]) / float(parts[1])
                if result == int(result):
                    return str(int(result))
                return str(result)

        num = float(text)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return text.lower().strip()


def extract_math_answer(response: str) -> Optional[str]:
    if not response:
        return None

    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])

    numbers = re.findall(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", response)
    if numbers:
        return normalize_number(numbers[-1])

    return None


def compare_math(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    return normalize_number(predicted) == normalize_number(ground_truth)


# --- QA (SQuAD-like) ---

_ARTICLES = {"a", "an", "the"}


def normalize_qa(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    toks = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(toks)


def f1_score(pred: str, truth: str) -> float:
    pred_toks = normalize_qa(pred).split()
    truth_toks = normalize_qa(truth).split()

    if not pred_toks and not truth_toks:
        return 1.0
    if not pred_toks or not truth_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(truth_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(truth_toks)
    return (2 * precision * recall) / (precision + recall)


def exact_match(pred: str, truth: str) -> bool:
    return normalize_qa(pred) == normalize_qa(truth)


def extract_text_answer(response: str) -> str:
    if not response:
        return ""

    s = response.strip()
    s = re.sub(r"^(final\s+answer|answer)\s*[:\-]\s*", "", s, flags=re.I)
    s = s.splitlines()[0].strip()

    # accept boxed answers too
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        s = m.group(1).strip()

    # strip quotes
    s = s.strip().strip('"').strip("'")
    return s


def score_qa(pred: str, answers: List[str]) -> Dict[str, float]:
    if not answers:
        return {"em": 0.0, "f1": 0.0}

    em = max(1.0 if exact_match(pred, a) else 0.0 for a in answers)
    f1 = max(f1_score(pred, a) for a in answers)
    return {"em": em, "f1": f1}


# --- MCQA ---


def extract_choice_label(response: str, valid: str = "ABCD") -> Optional[str]:
    if not response:
        return None
    s = response.strip().upper()

    if len(s) == 1 and s in valid:
        return s

    tokens = re.findall(r"[A-Z]", s)
    tokens = [t for t in tokens if t in valid]
    if tokens:
        return tokens[-1]

    return None


def evaluate_response(
    response: str,
    ground_truth: Union[str, List[str]],
    answer_style: str = "math",
    valid_labels: str = "ABCD",
) -> Tuple[Optional[str], bool, Dict[str, float]]:
    """Return (extracted, is_correct, metrics)."""

    style = (answer_style or "math").lower()

    if style == "math":
        extracted = extract_math_answer(response)
        ok = compare_math(extracted, str(ground_truth))
        return extracted, ok, {}

    if style == "mcqa":
        extracted = extract_choice_label(response, valid=valid_labels)
        ok = extracted is not None and extracted == str(ground_truth).strip().upper()
        return extracted, ok, {}

    answers = [str(a) for a in ground_truth] if isinstance(ground_truth, list) else [str(ground_truth)]
    pred = extract_text_answer(response)
    scores = score_qa(pred, answers)
    ok = scores["em"] >= 1.0
    return pred, ok, scores
