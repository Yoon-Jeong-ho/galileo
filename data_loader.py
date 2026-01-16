"""
Data loading utilities for the experiment.
"""

import json
import os
from typing import List, Dict, Any, Iterator, Optional


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
    
    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(item: Dict[str, Any], file_path: str) -> None:
    """
    Append a single item to a JSONL file.
    
    Args:
        item: Dictionary to append
        file_path: Path to the output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def iterate_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    Iterate over a JSONL file without loading everything into memory.
    
    Args:
        file_path: Path to the JSONL file
    
    Yields:
        Parsed JSON objects one at a time
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_test_name(file_path: str) -> str:
    """
    Extract test name from file path.
    
    Args:
        file_path: Path to the data file
    
    Returns:
        Test name (filename without extension)
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def load_dataset(
    file_path: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load a dataset with optional sampling.
    
    Args:
        file_path: Path to the JSONL file
        num_samples: Number of samples to load (-1 for all)
        shuffle: Whether to shuffle before sampling
        seed: Random seed for shuffling
    
    Returns:
        List of data samples
    """
    data = load_jsonl(file_path)
    
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(data)
    
    if num_samples > 0:
        data = data[:num_samples]
    
    return data


def prepare_problem(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a problem item for the experiment.
    
    Expected input format: {"question": "...", "answer": "..."}
    
    Args:
        item: Raw data item
    
    Returns:
        Prepared problem with question and ground truth answer
    """
    return {
        "question": item.get("question", ""),
        "answer": str(item.get("answer", "")),
        "original": item,
    }
