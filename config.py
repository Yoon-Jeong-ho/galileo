"""Configuration for Galileo Adversarial Persona Experiment Pipeline.

This project is designed to run on shared GPU servers.
Defaults are intentionally conservative to avoid OOM / max-context errors.

Key ideas:
- MAX_MODEL_LEN controls the context window allocated by vLLM.
- MAX_TOKENS controls the generation length requested per call.
  (Generation must fit inside the model context together with the prompt.)

If you see max-length errors, reduce MAX_TOKENS and/or MAX_MODEL_LEN.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent


def _split_visible_devices(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def infer_tensor_parallel_size(raw_visible_devices: str | None) -> int:
    visible = _split_visible_devices(raw_visible_devices)
    return max(1, len(visible) or 1)


# GPU configuration
# Shared-server safety: do not silently bind GPUs here. Experiment launch
# scripts should set CUDA_VISIBLE_DEVICES explicitly for auditable runs.
TENSOR_PARALLEL_SIZE = int(
    os.environ.get(
        "GALILEO_TENSOR_PARALLEL_SIZE",
        str(infer_tensor_parallel_size(os.environ.get("CUDA_VISIBLE_DEVICES"))),
    )
)

# Models to evaluate
# Keep the default list small; pass --model to override.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
]

# vLLM model context window
MAX_MODEL_LEN = 16384

# Max new tokens per generation
MAX_TOKENS = 2048

# Beam search for initial evaluation
BEAM_SEARCH_N = 10
BEAM_SEARCH_TEMPERATURE = 1.0

# Greedy decoding for adversarial rounds
GREEDY_TEMPERATURE = 1.0
GREEDY_N = 1

# Adversarial testing
MAX_ADVERSARIAL_ROUNDS = 5

# Answer extraction patterns (priority order)
ANSWER_PATTERNS = [
    r"\boxed\{([^}]+)\}",
    r"boxed\{([^}]+)\}",
    r"\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*",
    r"(?:answer|result|solution)\s*(?:is|=|:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    r"=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
]

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the given problem step by step and "
    "provide your final answer in \boxed{} format. "
    "For example, if the answer is 42, write \boxed{42}."
)

INSTRUCTION_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer inside \\boxed{{}}.\n\n"
    "Problem: {question}"
)

# Paths
DATA_DIR = os.environ.get("GALILEO_DATA_DIR", str(REPO_ROOT / "data"))
RESULTS_DIR = os.environ.get("GALILEO_RESULTS_DIR", str(REPO_ROOT / "results"))


@dataclass
class ExperimentConfig:
    models: List[str] = field(default_factory=lambda: MODELS.copy())
    data_files: List[str] = field(default_factory=list)
    results_dir: str = RESULTS_DIR

    max_model_len: int = MAX_MODEL_LEN
    max_tokens: int = MAX_TOKENS

    beam_search_n: int = BEAM_SEARCH_N
    beam_search_temperature: float = BEAM_SEARCH_TEMPERATURE

    greedy_temperature: float = GREEDY_TEMPERATURE
    max_adversarial_rounds: int = MAX_ADVERSARIAL_ROUNDS

    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE

    # vLLM stability knobs
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False

    # Personas to run in Phase 2. Empty means "all".
    personas: List[str] = field(default_factory=list)

    test_mode: bool = False
    num_samples: int = -1

    # Stability
    # If True, recreate the vLLM engine between Phase 1/2/3 to reduce long-run stalls.
    reset_engine_between_phases: bool = False

    # If True, recreate the vLLM engine between datasets/tasks (each data_file).
    # This is slower but strongly isolates long runs from vLLM state accumulation.
    reset_engine_between_tasks: bool = False

    # Reproducibility
    seed: int = 42
    retry_variant: str = "baseline"
    recovery_variant: str = "baseline"

    def __post_init__(self):
        if not self.data_files and os.path.exists(DATA_DIR):
            self.data_files = [
                os.path.join(DATA_DIR, f)
                for f in os.listdir(DATA_DIR)
                if f.endswith(".jsonl")
            ]
            self.data_files.sort()
