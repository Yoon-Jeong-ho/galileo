"""Data loading utilities.

Datasets are provided as JSONL. Each row should have at minimum:
- question: str
- task: one of {math, qa, mcqa}

Task-specific fields:
- math: answer: str
- qa: answers: list[str] (or answer: str)
- mcqa: choices: list[{label,text}], label: str (A/B/C/D)
"""

import json
import os
from typing import List, Dict, Any, Iterator


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def iterate_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_test_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def load_dataset(file_path: str, num_samples: int = -1, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    data = load_jsonl(file_path)
    if shuffle:
        import random

        random.seed(seed)
        random.shuffle(data)
    if num_samples > 0:
        data = data[:num_samples]
    return data


def prepare_problem(item: Dict[str, Any]) -> Dict[str, Any]:
    task = str(item.get("task", "math")).lower().strip()
    question = str(item.get("question", "")).strip()
    correction_evidence = None
    for key in (
        "correction_evidence",
        "evidence",
        "supporting_evidence",
        "supporting_facts",
        "explanation",
        "rationale",
    ):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            parts = [str(v).strip() for v in value if str(v).strip()]
            correction_evidence = "\n".join(parts) if parts else None
        else:
            text = str(value).strip()
            correction_evidence = text or None
        if correction_evidence:
            break

    if task == "qa":
        gt = item.get("answers", None)
        if gt is None:
            gt = [str(item.get("answer", ""))]
        if isinstance(gt, str):
            gt = [gt]
        gt = [str(a).strip() for a in gt if str(a).strip() != ""]
    elif task == "mcqa":
        gt = str(item.get("label", item.get("answer", ""))).strip().upper()
    else:
        gt = str(item.get("answer", "")).strip()

    choices = item.get("choices", [])
    if not isinstance(choices, list):
        choices = []

    return {
        "task": task,
        "question": question,
        "ground_truth": gt,
        "choices": choices,
        "correction_evidence": correction_evidence,
        "original": item,
    }
