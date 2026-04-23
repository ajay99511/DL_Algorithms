"""Tests for evaluate/few_shot.py — FewShotAnalyzer."""
from __future__ import annotations

import tempfile
import os
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
import hypothesis.strategies as st

from evaluate.config import EvalConfig
from evaluate.few_shot import FewShotAnalyzer
from shared.logging_utils import JSONLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger() -> tuple[JSONLogger, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    return JSONLogger(tmp.name), tmp.name


def _config_with_shots(shot_counts: dict[str, list[int]]) -> EvalConfig:
    cfg = EvalConfig()
    cfg.shot_counts = shot_counts
    return cfg


def _mock_lm_eval_success(accuracy: float = 0.5):
    """Return a mock lm_eval module that always succeeds with given accuracy."""
    mock = MagicMock()
    mock.simple_evaluate.return_value = {
        "results": {
            "arc_challenge": {"acc_norm,none": accuracy},
            "hellaswag": {"acc_norm,none": accuracy},
            "mmlu": {"acc,none": accuracy},
            "truthfulqa_mc": {"mc2,none": accuracy},
        }
    }
    return mock


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_all_task_shot_keys_present() -> None:
    """All (task, n_shots) keys must be present in the returned dict."""
    log, path = _make_logger()
    try:
        shot_counts = {"arc_challenge": [0, 5, 25], "hellaswag": [0, 10]}
        cfg = _config_with_shots(shot_counts)
        analyzer = FewShotAnalyzer(cfg, log)

        mock_lm_eval = _mock_lm_eval_success()
        mock_model = MagicMock()

        import evaluate.few_shot as fs_module
        with patch.object(fs_module, "_HAS_LM_EVAL", True), \
             patch.object(fs_module, "lm_eval", mock_lm_eval, create=True):
            results = analyzer.run("test-model", mock_model)

        expected_keys = {
            ("arc_challenge", 0), ("arc_challenge", 5), ("arc_challenge", 25),
            ("hellaswag", 0), ("hellaswag", 10),
        }
        assert set(results.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(results.keys())}"
        )
    finally:
        os.unlink(path)


def test_failing_pair_records_none_and_continues() -> None:
    """A failing (task, n_shots) pair must record None without aborting the loop."""
    log, path = _make_logger()
    try:
        shot_counts = {"arc_challenge": [0, 5], "hellaswag": [0]}
        cfg = _config_with_shots(shot_counts)
        analyzer = FewShotAnalyzer(cfg, log)

        call_count = [0]

        def _side_effect(model, tasks, num_fewshot, device):
            call_count[0] += 1
            # Fail on arc_challenge with 5 shots
            if tasks == ["arc_challenge"] and num_fewshot == 5:
                raise RuntimeError("simulated failure")
            return {"results": {tasks[0]: {"acc_norm,none": 0.6}}}

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.side_effect = _side_effect
        mock_model = MagicMock()

        import evaluate.few_shot as fs_module
        with patch.object(fs_module, "_HAS_LM_EVAL", True), \
             patch.object(fs_module, "lm_eval", mock_lm_eval, create=True):
            results = analyzer.run("test-model", mock_model)

        # All 3 keys must be present
        assert ("arc_challenge", 0) in results
        assert ("arc_challenge", 5) in results
        assert ("hellaswag", 0) in results

        # The failing pair must be None
        assert results[("arc_challenge", 5)] is None

        # The passing pairs must have scores
        assert results[("arc_challenge", 0)] is not None
        assert results[("hellaswag", 0)] is not None

        # All 3 calls were attempted
        assert call_count[0] == 3
    finally:
        os.unlink(path)


def test_n_shots_values_match_config() -> None:
    """n_shots values in result keys must match those in config.shot_counts."""
    log, path = _make_logger()
    try:
        shot_counts = {"mmlu": [0, 3, 7]}
        cfg = _config_with_shots(shot_counts)
        analyzer = FewShotAnalyzer(cfg, log)

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.return_value = {
            "results": {"mmlu": {"acc,none": 0.4}}
        }
        mock_model = MagicMock()

        import evaluate.few_shot as fs_module
        with patch.object(fs_module, "_HAS_LM_EVAL", True), \
             patch.object(fs_module, "lm_eval", mock_lm_eval, create=True):
            results = analyzer.run("test-model", mock_model)

        for (task, n_shots) in results.keys():
            assert n_shots in shot_counts[task], (
                f"n_shots={n_shots} not in config shot_counts for task {task}"
            )
    finally:
        os.unlink(path)


def test_no_lm_eval_returns_all_none() -> None:
    """When lm_eval is not installed, all results must be None."""
    log, path = _make_logger()
    try:
        shot_counts = {"arc_challenge": [0, 5]}
        cfg = _config_with_shots(shot_counts)
        analyzer = FewShotAnalyzer(cfg, log)

        import evaluate.few_shot as fs_module
        with patch.object(fs_module, "_HAS_LM_EVAL", False):
            results = analyzer.run("test-model", MagicMock())

        assert all(v is None for v in results.values()), (
            "All results must be None when lm_eval is not installed"
        )
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Property 3: Few-shot result completeness
# Feature: evaluate-improvements, Property 3: Few-shot result completeness
# Validates: Requirements 4.1, 4.5
# ---------------------------------------------------------------------------

@given(
    task_shot_map=st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        values=st.lists(st.integers(min_value=0, max_value=25), min_size=1, max_size=4, unique=True),
        min_size=1,
        max_size=4,
    ),
    fail_fraction=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=50)
def test_few_shot_result_completeness(
    task_shot_map: dict[str, list[int]],
    fail_fraction: float,
) -> None:
    """Result dict must have exactly T×S_t entries; None for failures."""
    log, path = _make_logger()
    try:
        cfg = _config_with_shots(task_shot_map)
        analyzer = FewShotAnalyzer(cfg, log)

        # Compute expected total entries
        expected_count = sum(len(shots) for shots in task_shot_map.values())

        call_idx = [0]
        total_calls = expected_count

        def _side_effect(model, tasks, num_fewshot, device):
            call_idx[0] += 1
            # Fail approximately fail_fraction of calls
            if call_idx[0] / max(total_calls, 1) <= fail_fraction:
                raise RuntimeError("simulated failure")
            return {"results": {tasks[0]: {"acc_norm,none": 0.5}}}

        mock_lm_eval = MagicMock()
        mock_lm_eval.simple_evaluate.side_effect = _side_effect

        import evaluate.few_shot as fs_module
        with patch.object(fs_module, "_HAS_LM_EVAL", True), \
             patch.object(fs_module, "lm_eval", mock_lm_eval, create=True):
            results = analyzer.run("model", MagicMock())

        assert len(results) == expected_count, (
            f"Expected {expected_count} entries, got {len(results)}"
        )
    finally:
        os.unlink(path)
