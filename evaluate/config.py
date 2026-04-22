from dataclasses import dataclass, field
from shared.config import BaseConfig


@dataclass
class EvalConfig(BaseConfig):
    # Models to evaluate
    models: list[str] = field(default_factory=lambda: ["gpt2", "EleutherAI/pythia-160m"])
    # Benchmarks: task_name -> num_fewshot
    tasks: dict[str, int] = field(default_factory=lambda: {
        "arc_challenge": 25,
        "hellaswag": 10,
        "mmlu": 5,
        "truthfulqa_mc": 0,
    })
    # Paths
    output_dir: str = "outputs/project6"
    log_path: str = "outputs/project6/experiment_log.jsonl"
    report_csv: str = "outputs/project6/results.csv"
    report_md: str = "outputs/project6/results.md"
