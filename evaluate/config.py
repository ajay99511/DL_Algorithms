from __future__ import annotations

from dataclasses import dataclass, field
from shared.config import BaseConfig


@dataclass
class EvalConfig(BaseConfig):
    # Models to evaluate
    models: list[str] = field(default_factory=lambda: ["gpt2", "EleutherAI/pythia-160m"])
    # Benchmarks: task_name -> num_fewshot (used for primary evaluation)
    tasks: dict[str, int] = field(default_factory=lambda: {
        "arc_challenge": 25,
        "hellaswag": 10,
        "mmlu": 5,
        "truthfulqa_mc": 0,
    })
    # Few-shot sensitivity: task_name -> list of shot counts to sweep
    shot_counts: dict[str, list[int]] = field(default_factory=lambda: {
        "arc_challenge": [0, 1, 5, 25],
        "hellaswag": [0, 1, 10],
        "mmlu": [0, 5],
        "truthfulqa_mc": [0],
    })
    # Local model checkpoint (pretrain/); None = skip local model evaluation
    local_checkpoint_path: str | None = None
    # Path to a plain-text held-out corpus for perplexity; empty string = skip
    perplexity_corpus: str = ""
    # Paths
    output_dir: str = "outputs/project6"
    log_path: str = "outputs/project6/experiment_log.jsonl"
    report_csv: str = "outputs/project6/results.csv"
    report_md: str = "outputs/project6/results.md"
    report_narrative: str = "outputs/project6/results_narrative.md"
