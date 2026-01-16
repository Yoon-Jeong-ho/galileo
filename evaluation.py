"""
Answer evaluation module for extracting and comparing answers.
"""

import re
from typing import Optional, Tuple, List
from config import ANSWER_PATTERNS


def normalize_number(text: str) -> Optional[str]:
    """
    Normalize a number string for comparison.
    
    Args:
        text: Raw number string (may contain commas, dollar signs, etc.)
    
    Returns:
        Normalized number string or None if invalid
    """
    if not text:
        return None
    
    # Remove common formatting
    text = text.strip()
    text = text.replace(",", "")  # Remove thousand separators
    text = text.replace("$", "")  # Remove dollar signs
    text = text.replace(" ", "")  # Remove spaces
    text = text.replace("%", "")  # Remove percent sign
    
    # Try to parse as number
    try:
        # Handle fractions like "1/2"
        if "/" in text:
            parts = text.split("/")
            if len(parts) == 2:
                result = float(parts[0]) / float(parts[1])
                # Return as integer if whole number
                if result == int(result):
                    return str(int(result))
                return str(result)
        
        # Parse as float and normalize
        num = float(text)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return text.lower().strip()


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from a model response.
    
    Args:
        response: Full model response text
    
    Returns:
        Extracted answer string or None if not found
    """
    if not response:
        return None
    
    # Try each pattern in priority order
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Use the last match (most likely the final answer)
            answer = matches[-1]
            return normalize_number(answer)
    
    # Fallback: try to find the last number in the response
    numbers = re.findall(r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", response)
    if numbers:
        return normalize_number(numbers[-1])
    
    return None


def compare_answers(predicted: Optional[str], ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth.
    
    Args:
        predicted: Extracted answer from model
        ground_truth: Expected correct answer
    
    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False
    
    pred_norm = normalize_number(predicted)
    truth_norm = normalize_number(ground_truth)
    
    if pred_norm is None or truth_norm is None:
        return False
    
    return pred_norm == truth_norm


def compute_accuracy(results: List[Tuple[str, str, bool]]) -> dict:
    """
    Compute accuracy statistics from results.
    
    Args:
        results: List of (predicted, ground_truth, is_correct) tuples
    
    Returns:
        Dictionary with accuracy statistics
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    
    correct = sum(1 for _, _, is_correct in results if is_correct)
    total = len(results)
    
    return {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total
    }


def evaluate_response(response: str, ground_truth: str) -> Tuple[Optional[str], bool]:
    """
    Extract answer from response and compare with ground truth.
    
    Args:
        response: Model response
        ground_truth: Expected answer
    
    Returns:
        Tuple of (extracted_answer, is_correct)
    """
    extracted = extract_answer(response)
    is_correct = compare_answers(extracted, ground_truth)
    return extracted, is_correct
