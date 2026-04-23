"""Few-shot sensitivity analysis.

Runs each benchmark task at multiple shot counts and records how accuracy
changes, revealing whether model performance is sensitive to the number of
in-context examples.
"""

from __future__ import annotations

import traceback
from typing import Any

from evaluate.config import EvalConfig
from shared.logging_utils import JSONLogger

try:
    import lm_eval  # type: ignore
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False


class FewShotAnalyzer:
    """Run benchmark tasks at multiple shot counts for sensitivity analysis.

    Args:
        config:  EvalConfig containing ``shot_counts`` mapping task → [n_shots].
        logger:  JSONLogger for structured error logging.
    """

    def __init__(self, config: EvalConfig, logger: JSONLogger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        model_name: str,
        lm_model: Any,
    ) -> dict[tuple[str, int], float | None]:
        """Evaluate *lm_model* on every (task, n_shots) combination.

        Args:
            model_name: Display name used in log entries.
            lm_model:   A loaded lm_eval model object.

        Returns:
            Dict keyed by ``(task_name, n_shots)`` → accuracy (float) or None
            if that specific evaluation failed.
        """
        results: dict[tuple[str, int], float | None] = {}

        if not _HAS_LM_EVAL:
            self.logger.log({
                "type": "warning",
                "component": "FewShotAnalyzer",
                "model_name": model_name,
                "message": "lm_eval not installed; skipping few-shot sensitivity analysis.",
            })
            # Return None for every (task, n_shots) pair
            for task_name, shot_list in self.config.shot_counts.items():
                for n_shots in shot_list:
                    results[(task_name, n_shots)] = None
            return results

        for task_name, shot_list in self.config.shot_counts.items():
            for n_shots in shot_list:
                try:
                    output: dict[str, Any] = lm_eval.simple_evaluate(
                        model=lm_model,
                        tasks=[task_name],
                        num_fewshot=n_shots,
                        device="cpu",
                    )
                    task_results = output.get("results", {}).get(task_name, {})
                    accuracy: float | None = (
                        task_results.get("acc_norm,none")
                        or task_results.get("acc,none")
                        or task_results.get("mc2,none")
                    )
                    results[(task_name, n_shots)] = accuracy
                except Exception:  # noqa: BLE001
                    self.logger.log({
                        "type": "few_shot_eval_error",
                        "component": "FewShotAnalyzer",
                        "model_name": model_name,
                        "task_name": task_name,
                        "n_shots": n_shots,
                        "traceback_summary": traceback.format_exc(limit=5),
                    })
                    results[(task_name, n_shots)] = None

        return results
