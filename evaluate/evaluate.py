"""Benchmark evaluation wrapper around lm-evaluation-harness.

Benchmarks covered:
  - arc_challenge:   science reasoning (multiple choice) — Clark et al. 2018
  - hellaswag:       commonsense completion — Zellers et al. 2019
  - mmlu:            world knowledge, 5-shot — Hendrycks et al. 2021
  - truthfulqa_mc:   factual accuracy / avoiding hallucination — Lin et al. 2022

CLI usage:
  python -m evaluate.evaluate [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any

from evaluate.config import EvalConfig
from shared.config import load_config
from shared.logging_utils import JSONLogger

# Optional dependency — lm-evaluation-harness may not be installed
try:
    import lm_eval  # type: ignore
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False


def run_evaluation(
    model_name: str,
    tasks: dict[str, int],
    config: EvalConfig,
    logger: JSONLogger,
) -> dict[str, float | None]:
    """Wrap lm_eval.simple_evaluate() to run benchmark evaluation.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. "gpt2", "EleutherAI/pythia-160m").
    tasks:
        Mapping of task_name -> num_fewshot.
    config:
        EvalConfig instance (used for output_dir etc.).
    logger:
        JSONLogger for structured error logging.

    Returns
    -------
    dict mapping task_name -> accuracy (float) or None on failure.
    Empty dict if the model itself failed to load.
    """
    if not _HAS_LM_EVAL:
        logger.log({
            "type": "warning",
            "model_name": model_name,
            "message": "lm_eval is not installed; skipping evaluation",
        })
        return {task: None for task in tasks}

    # Attempt to load the model — catch hard failures early
    try:
        lm = lm_eval.models.get_model("hf")(
            pretrained=model_name,
            device="cpu",  # CPU-only
        )
    except (OSError, ValueError) as exc:
        logger.log({
            "type": "model_load_error",
            "model_name": model_name,
            "error_type": type(exc).__name__,
            "traceback_summary": traceback.format_exc(limit=5),
        })
        return {}

    results: dict[str, float | None] = {}

    for task_name, num_fewshot in tasks.items():
        try:
            # arc_challenge: science reasoning (multiple choice)
            # hellaswag:     commonsense completion
            # mmlu:          world knowledge (5-shot)
            # truthfulqa_mc: factual accuracy / avoiding hallucination
            output: dict[str, Any] = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                device="cpu",
            )
            # Extract normalised accuracy from results dict
            task_results = output.get("results", {}).get(task_name, {})
            accuracy: float | None = (
                task_results.get("acc_norm,none")
                or task_results.get("acc,none")
                or task_results.get("mc2,none")  # TruthfulQA uses mc2
            )
            results[task_name] = accuracy
        except Exception as exc:  # noqa: BLE001
            logger.log({
                "type": "task_eval_error",
                "model_name": model_name,
                "task_name": task_name,
                "traceback_summary": traceback.format_exc(limit=5),
            })
            results[task_name] = None

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full evaluation pipeline on configured models and benchmarks.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML config file (default: use EvalConfig defaults)",
    )
    args = parser.parse_args()

    if args.config is not None:
        import os
        if not os.path.exists(args.config):
            print(
                f"ERROR: config file not found: {args.config}",
                file=sys.stderr,
            )
            sys.exit(1)
        cfg = load_config(args.config, EvalConfig)
    else:
        cfg = EvalConfig()

    from evaluate.pipeline import EvaluationPipeline
    from pathlib import Path

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    _logger = JSONLogger(cfg.log_path)
    pipeline = EvaluationPipeline(config=cfg, logger=_logger)
    pipeline.run()
