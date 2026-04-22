from dataclasses import dataclass
from shared.config import BaseConfig


@dataclass
class ReasoningConfig(BaseConfig):
    # Model
    model_name: str = "gpt2"
    # Inference
    max_new_tokens: int = 200
    beam_width: int = 4
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    # Evaluation
    gsm8k_subset_size: int = 50
    bigbench_subset_size: int = 50
    # Paths
    benchmark_file: str = "outputs/project5/benchmark_results.json"
    log_path: str = "outputs/project5/experiment_log.jsonl"
