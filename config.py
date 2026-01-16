"""
Configuration for Galileo Adversarial Persona Experiment Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

# GPU Configuration
CUDA_VISIBLE_DEVICES = "3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
TENSOR_PARALLEL_SIZE = 4

# Models to evaluate
MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

# Generation parameters
MAX_TOKENS = 16384

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
    r"\\boxed\{([^}]+)\}",           # LaTeX boxed format
    r"boxed\{([^}]+)\}",             # Without backslash
    r"\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*",  # Bold number
    r"(?:answer|result|solution)\s*(?:is|=|:)\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",  # "answer is X"
    r"=\s*\$?([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",  # Final equation result
]

# System prompt for math problem solving
SYSTEM_PROMPT = """You are a helpful math assistant. Solve the given problem step by step and provide your final answer in \\boxed{} format. For example, if the answer is 42, write \\boxed{42}."""

# Instruction template for initial evaluation
INSTRUCTION_TEMPLATE = """Solve the following math problem step by step. Put your final numerical answer inside \\boxed{}.

Problem: {question}"""

# Paths
DATA_DIR = "/data_x/aa007878/galileo/data"
RESULTS_DIR = "/data_x/aa007878/galileo/results"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    models: List[str] = field(default_factory=lambda: MODELS.copy())
    data_files: List[str] = field(default_factory=list)
    results_dir: str = RESULTS_DIR
    max_tokens: int = MAX_TOKENS
    beam_search_n: int = BEAM_SEARCH_N
    beam_search_temperature: float = BEAM_SEARCH_TEMPERATURE
    greedy_temperature: float = GREEDY_TEMPERATURE
    max_adversarial_rounds: int = MAX_ADVERSARIAL_ROUNDS
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    test_mode: bool = False
    num_samples: int = -1  # -1 means all samples
    
    def __post_init__(self):
        if not self.data_files:
            # Auto-discover data files
            if os.path.exists(DATA_DIR):
                self.data_files = [
                    os.path.join(DATA_DIR, f) 
                    for f in os.listdir(DATA_DIR) 
                    if f.endswith('.jsonl')
                ]
