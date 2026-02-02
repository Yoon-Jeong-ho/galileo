"""Task definitions (prompting + evaluation types).

We support three broad task families:
- math: numeric final answer
- qa: short answer (alias list)
- mcqa: multiple choice (label)

Important: We allow chain-of-thought / reasoning for *all* tasks, but we
standardize the *final answer format* across tasks:

  - The final answer MUST be inside \boxed{...}.
  - The content inside \boxed{...} must be task-appropriate:
      * math: a single number (or simple fraction)
      * qa: answer text only
      * mcqa: label only (A/B/C/D)

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
            "You are a helpful math assistant. You may reason step by step. "
            "Put ONLY your final answer inside \\boxed{...}."
        ),
        instruction_template=(
            "Solve the following problem. You may reason step by step, but put ONLY your final answer inside \\boxed{{}}.\n\n"
            "Problem: {question}"
        ),
        answer_style="math",
    ),
    "qa": TaskSpec(
        name="qa",
        system_prompt=(
            "You are a helpful assistant. You may reason step by step. "
            "Put ONLY the final answer text inside \\boxed{...}."
        ),
        instruction_template=(
            "Answer the question. You may reason step by step, but put ONLY the final answer text inside \\boxed{{}}.\n\n"
            "Question: {question}"
        ),
        answer_style="qa",
    ),
    "mcqa": TaskSpec(
        name="mcqa",
        system_prompt=(
            "You are a helpful assistant. You may reason step by step. "
            "Choose the correct option and put ONLY the option label (A/B/C/D) inside \\boxed{...}."
        ),
        instruction_template=(
            "Choose the correct option. You may reason step by step, but put ONLY the option label (A/B/C/D) inside \\boxed{{}}.\n\n"
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
