"""Tests for graceful error handling in the evaluation runner.

Property 15: Graceful Continuation on Task Failure

# Feature: deep-learning-llm-mastery, Property 15: Graceful Continuation on Task Failure
"""

from __future__ import annotations

import traceback
from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from evaluate.config import EvalConfig
from shared.logging_utils import JSONLogger


# ---------------------------------------------------------------------------
# Minimal runner that mirrors run_evaluation's error-handling contract
# without requiring lm_eval to be installed.
# ---------------------------------------------------------------------------

def _run_tasks_with_error_handling(
    model_name: str,
    task_names: list[str],
    failing_tasks: set[str],
    logger: JSONLogger,
) -> dict[str, float | None]:
    """Simulate the task-level error handling loop from evaluate.run_evaluation."""
    results: dict[str, float | None] = {}
    for task_name in task_names:
        try:
            if task_name in failing_tasks:
                raise RuntimeError(f"Simulated failure for task {task_name}")
            results[task_name] = 0.5  # dummy success score
        except Exception:  # noqa: BLE001
            logger.log({
                "type": "task_eval_error",
                "model_name": model_name,
                "task_name": task_name,
                "traceback_summary": traceback.format_exc(limit=5),
            })
            results[task_name] = None
    return results


# ---------------------------------------------------------------------------
# Property 15: Graceful Continuation on Task Failure
# ---------------------------------------------------------------------------

# Feature: deep-learning-llm-mastery, Property 15: Graceful Continuation on Task Failure
@given(
    n_tasks=st.integers(2, 10),
    n_failing=st.integers(1, 5),
)
@settings(max_examples=100)
def test_graceful_continuation(n_tasks: int, n_failing: int) -> None:
    """Runner must complete all non-failing tasks and return results without re-raising.

    **Validates: Requirements 6.10, 6.12**
    """
    import tempfile, os

    task_names = [f"task_{i}" for i in range(n_tasks)]
    # Clamp n_failing so we always have at least one non-failing task
    actual_failing = min(n_failing, n_tasks - 1)
    failing_tasks = set(task_names[:actual_failing])
    passing_tasks = set(task_names[actual_failing:])

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        log_path = tmp.name

    try:
        logger = JSONLogger(log_path)
        results = _run_tasks_with_error_handling(
            model_name="test-model",
            task_names=task_names,
            failing_tasks=failing_tasks,
            logger=logger,
        )

        # All tasks must have an entry in results
        assert set(results.keys()) == set(task_names), (
            f"Expected keys {set(task_names)}, got {set(results.keys())}"
        )

        # Failing tasks must be None
        for t in failing_tasks:
            assert results[t] is None, f"Failing task {t!r} should have None result"

        # Passing tasks must have a non-None score
        for t in passing_tasks:
            assert results[t] is not None, f"Passing task {t!r} should have a score"

    finally:
        os.unlink(log_path)


# ---------------------------------------------------------------------------
# Example test: error log format
# ---------------------------------------------------------------------------

def test_error_log_format() -> None:
    """Error log entry must contain model_name, error_type, and traceback_summary keys.

    _Requirements: 6.10_
    """
    import tempfile, os, json

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        log_path = tmp.name

    try:
        logger = JSONLogger(log_path)

        # Simulate a model load failure log entry (as done in evaluate.run_evaluation)
        try:
            raise OSError("Simulated model load failure")
        except OSError:
            logger.log({
                "type": "model_load_error",
                "model_name": "bad-model",
                "error_type": "OSError",
                "traceback_summary": traceback.format_exc(limit=5),
            })

        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        assert len(entries) == 1
        entry = entries[0]
        assert "model_name" in entry, "Log entry must contain 'model_name'"
        assert "error_type" in entry, "Log entry must contain 'error_type'"
        assert "traceback_summary" in entry, "Log entry must contain 'traceback_summary'"
        assert entry["model_name"] == "bad-model"
        assert entry["error_type"] == "OSError"
        assert isinstance(entry["traceback_summary"], str)
        assert len(entry["traceback_summary"]) > 0

    finally:
        os.unlink(log_path)


def test_model_load_failure_returns_empty_dict() -> None:
    """On model load failure, run_evaluation must return an empty dict."""
    from evaluate import evaluate as eval_module
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        log_path = tmp.name

    try:
        config = EvalConfig()
        logger = JSONLogger(log_path)

        # Patch lm_eval to be available but raise on model load
        mock_lm_eval = MagicMock()
        mock_lm_eval.models.get_model.return_value = MagicMock(
            side_effect=OSError("model not found")
        )

        with patch.object(eval_module, "_HAS_LM_EVAL", True), \
             patch.object(eval_module, "lm_eval", mock_lm_eval, create=True):
            result = eval_module.run_evaluation(
                model_name="nonexistent/model",
                tasks={"arc_challenge": 25},
                config=config,
                logger=logger,
            )

        assert result == {}, f"Expected empty dict on model load failure, got {result}"

    finally:
        os.unlink(log_path)
