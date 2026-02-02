"""Task definitions (prompting + evaluation types).

We support three broad task families:
- math: numeric final answer, evaluated by numeric normalization.
- qa: free-form short answer with aliases, evaluated by normalized EM/F1.
- mcqa: multiple-choice QA, evaluated by predicted label (A/B/C/D/...)

Datasets are expected to provide a `task` field per example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class TaskSpec:
    name: str
    system_prompt: str
    instruction_template: str
    answer_style: str  # math|qa|mcqa


TASKS: Dict[str, TaskSpec] = {
    "math": TaskSpec(
        name="math",
        system_prompt=(
            "You are a helpful math assistant. Solve the given problem step by step and "
            "provide your final answer in \\boxed{} format."
        ),
        instruction_template=(
            "Solve the following problem step by step. Put your final answer inside \\boxed{{}}.\n\n"
            "Problem: {question}"
        ),
        answer_style="math",
    ),
    "qa": TaskSpec(
        name="qa",
        system_prompt=(
            "You are a helpful assistant. Answer the question as briefly as possible. "
            "Output ONLY the answer text with no extra explanation."
        ),
        instruction_template=(
            "Answer the question. Output ONLY the answer text.\n\n"
            "Question: {question}"
        ),
        answer_style="qa",
    ),
    "mcqa": TaskSpec(
        name="mcqa",
        system_prompt=(
            "You are a helpful assistant. Choose the correct option. "
            "Output ONLY the option label (e.g., A, B, C, D)."
        ),
        instruction_template=(
            "Choose the correct option. Output ONLY the option label (A/B/C/D).\n\n"
            "Question: {question}\n\nOptions:\n{options}"
        ),
        answer_style="mcqa",
    ),
}


def format_mcqa_options(choices: List[Dict[str, str]]) -> str:
    lines = []
    for c in choices:
        label = c.get("label", "").strip()
        text = c.get("text", "").strip()
        if not label or not text:
            continue
        lines.append(f"{label}. {text}")
    return "\n".join(lines)


def get_task(task_name: Optional[str]) -> TaskSpec:
    t = (task_name or "math").strip().lower()
    return TASKS.get(t, TASKS["math"])
