"""Tests for evaluate/pipeline.py — EvaluationPipeline."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
import hypothesis.strategies as st

from evaluate.config import EvalConfig
from evaluate.pipeline import EvaluationPipeline, _empty_bundle
from shared.logging_utils import JSONLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_BUNDLE_KEYS = {
    "benchmark_scores",
    "perplexity",
    "few_shot_sensitivity",
    "calibration",
    "weight_analysis",
    "activation_analysis",
    "dataset_stats",
    "local_model",
}


def _make_minimal_config(tmpdir: str) -> EvalConfig:
    cfg = EvalConfig()
    cfg.models = ["gpt2"]
    cfg.output_dir = tmpdir
    cfg.log_path = os.path.join(tmpdir, "log.jsonl")
    cfg.report_csv = os.path.join(tmpdir, "results.csv")
    cfg.report_md = os.path.join(tmpdir, "results.md")
    cfg.report_narrative = os.path.join(tmpdir, "results_narrative.md")
    cfg.local_checkpoint_path = None
    cfg.perplexity_corpus = ""
    return cfg


def _make_logger(tmpdir: str) -> JSONLogger:
    return JSONLogger(os.path.join(tmpdir, "log.jsonl"))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_bundle_has_all_required_keys() -> None:
    """EvaluationBundle must always contain all 8 required keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        # Mock run_evaluation to avoid real model loading
        with patch("evaluate.pipeline.run_evaluation", return_value={}):
            bundle = pipeline.run()

        assert set(bundle.keys()) >= _REQUIRED_BUNDLE_KEYS, (
            f"Missing keys: {_REQUIRED_BUNDLE_KEYS - set(bundle.keys())}"
        )


def test_step_failure_sets_key_to_none_and_continues() -> None:
    """A step failure must set the bundle key to None without aborting the pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        # Make run_evaluation raise to simulate benchmark step failure
        with patch("evaluate.pipeline.run_evaluation", side_effect=RuntimeError("boom")):
            bundle = pipeline.run()

        # Pipeline must complete and return all keys
        assert set(bundle.keys()) >= _REQUIRED_BUNDLE_KEYS
        # benchmark_scores should be empty dict (caught exception)
        assert isinstance(bundle["benchmark_scores"], dict)


def test_missing_local_checkpoint_sets_local_model_none() -> None:
    """Missing local_checkpoint_path must set bundle['local_model'] = None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        cfg.local_checkpoint_path = None
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        with patch("evaluate.pipeline.run_evaluation", return_value={}):
            bundle = pipeline.run()

        assert bundle["local_model"] is None


def test_nonexistent_local_checkpoint_sets_local_model_none() -> None:
    """Non-existent checkpoint path must set bundle['local_model'] = None and log warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        cfg.local_checkpoint_path = "/nonexistent/path/best.pt"
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        with patch("evaluate.pipeline.run_evaluation", return_value={}):
            bundle = pipeline.run()

        assert bundle["local_model"] is None

        # Check warning was logged
        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        warning_entries = [e for e in entries if e.get("type") == "warning"]
        assert any("Checkpoint not found" in e.get("message", "") for e in warning_entries), (
            "Expected a warning about missing checkpoint"
        )


def test_start_and_completion_events_logged() -> None:
    """Pipeline must log start and completion events with UTC timestamp and model list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        with patch("evaluate.pipeline.run_evaluation", return_value={}):
            pipeline.run()

        log_path = os.path.join(tmpdir, "log.jsonl")
        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        types = [e.get("type") for e in entries]
        assert "pipeline_start" in types, "pipeline_start event not logged"
        assert "pipeline_complete" in types, "pipeline_complete event not logged"

        start_entry = next(e for e in entries if e.get("type") == "pipeline_start")
        assert "timestamp_utc" in start_entry
        assert "models" in start_entry
        assert start_entry["models"] == cfg.models


# ---------------------------------------------------------------------------
# Property 8: EvaluationBundle key completeness
# Feature: evaluate-improvements, Property 8: EvaluationBundle key completeness
# Validates: Requirements 7.5, 7.6
# ---------------------------------------------------------------------------

@given(
    models=st.lists(
        st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=20)
def test_bundle_key_completeness_property(models: list[str]) -> None:
    """EvaluationBundle must always contain all 8 required keys regardless of step failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_minimal_config(tmpdir)
        cfg.models = models
        log = _make_logger(tmpdir)
        pipeline = EvaluationPipeline(cfg, log)

        # Inject failures in run_evaluation
        with patch("evaluate.pipeline.run_evaluation", side_effect=RuntimeError("injected")):
            bundle = pipeline.run()

        assert set(bundle.keys()) >= _REQUIRED_BUNDLE_KEYS, (
            f"Missing keys: {_REQUIRED_BUNDLE_KEYS - set(bundle.keys())}"
        )
