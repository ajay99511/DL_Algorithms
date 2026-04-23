# Implementation Plan: evaluate-improvements

## Overview

Extend the `evaluate` module from a thin `lm-evaluation-harness` wrapper into a professional
evaluation pipeline. Eight gaps are closed in sequence: CLI entry point, local model
evaluation, perplexity metric, few-shot sensitivity analysis, calibration analysis, rich
dataset statistics, unified orchestration pipeline, and a cross-model comparison narrative
with statistical testing.

All new code is pure Python / PyTorch, CPU-only. Tests use `unittest.mock` to avoid real
model loading and `lm_eval` calls; property-based tests use Hypothesis.

---

## Tasks

- [x] 1. Extend `EvalConfig` with new fields
  - Add `shot_counts: dict[str, list[int]]` with per-task default lists to `evaluate/config.py`
  - Add `local_checkpoint_path: str | None = None` field
  - Add `perplexity_corpus: str = ""` field (path to plain-text held-out corpus; empty = skip)
  - Verify `EvalConfig` still loads correctly from `evaluate/config.yaml` via `BaseConfig`
  - _Requirements: 2.1, 3.1, 4.1_

- [x] 2. Add CLI entry point to `evaluate/evaluate.py`
  - Add `if __name__ == "__main__":` block using `argparse` with a single optional `--config PATH` flag
  - When `--config` is given and the file exists, load `EvalConfig` via `load_config(path, EvalConfig)` and pass to `EvaluationPipeline`
  - When `--config` path does not exist, print a descriptive error to stderr and `sys.exit(1)`
  - When no flags are given, instantiate `EvalConfig()` with defaults and proceed
  - `--help` must print usage and exit 0 (handled automatically by `argparse`)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement `PerplexityCalculator` in `evaluate/perplexity.py`
  - [x] 3.1 Create `evaluate/perplexity.py` with `PerplexityCalculator` class
    - Constructor accepts `model: nn.Module`, `tokenizer: Any`, `context_length: int`, `logger: JSONLogger`
    - `compute(corpus: str) -> float | None` tokenizes corpus, splits into non-overlapping windows of `context_length` tokens, runs forward pass with `targets = input_ids[1:]`, accumulates cross-entropy loss, returns `math.exp(mean_loss)`
    - Return `None` and log a warning via `JSONLogger` if corpus is empty or tokenization fails
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [x]* 3.2 Write property test for `PerplexityCalculator` — lower bound and finiteness
    - **Property 1: Perplexity lower bound and finiteness**
    - Generate random token sequences (lengths 1–512) and a tiny randomly-initialised `GPTModel`; assert `compute()` returns a finite float ≥ 1.0
    - **Validates: Requirements 3.1, 3.2**

  - [x]* 3.3 Write property test for `PerplexityCalculator` — windowing consistency
    - **Property 2: Perplexity windowing consistency**
    - Generate sequences longer than `context_length`; compute perplexity via `PerplexityCalculator` and also manually window-by-window; assert results match within 1e-5
    - **Validates: Requirements 3.3**

  - [x]* 3.4 Write unit tests for `PerplexityCalculator` in `evaluate/tests/test_perplexity.py`
    - Test empty corpus returns `None` and logs a warning
    - Test single-window sequence returns finite float ≥ 1.0
    - Test multi-window sequence (length > context_length) returns finite float ≥ 1.0
    - _Requirements: 3.2, 3.3, 3.6_

- [x] 4. Implement `FewShotAnalyzer` in `evaluate/few_shot.py`
  - [x] 4.1 Create `evaluate/few_shot.py` with `FewShotAnalyzer` class
    - Constructor accepts `config: EvalConfig`, `logger: JSONLogger`
    - `run(model_name: str, lm_model: Any) -> dict[tuple[str, int], float | None]` iterates over `config.shot_counts`, calls `lm_eval.simple_evaluate` for each `(task, n_shots)` pair, catches per-pair failures (log error, record `None`, continue), returns dict keyed by `(task_name, n_shots)`
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x]* 4.2 Write property test for `FewShotAnalyzer` — result completeness
    - **Property 3: Few-shot result completeness**
    - Generate random `(M, T, S_t)` configurations with mock `lm_eval`; inject random failures; assert result dict has exactly `M × Σ S_t` entries with `None` for failures
    - **Validates: Requirements 4.1, 4.5**

  - [x]* 4.3 Write unit tests for `FewShotAnalyzer` in `evaluate/tests/test_few_shot.py`
    - Test that all `(task, n_shots)` keys are present in the returned dict
    - Test that a failing `(task, n_shots)` pair records `None` and does not abort the loop
    - Test that `n_shots` values in keys match those in `config.shot_counts`
    - _Requirements: 4.2, 4.3, 4.5_

- [x] 5. Implement `CalibrationAnalyzer` in `evaluate/calibration.py`
  - [x] 5.1 Create `evaluate/calibration.py` with `CalibrationAnalyzer` class
    - Constructor accepts `n_bins: int = 10`, `logger: JSONLogger | None = None`
    - `compute_ece(confidences: list[float], labels: list[int]) -> float | None` partitions into 10 equal-width bins over [0, 1], computes `bin_weight = bin_count / total`, returns `sum(bin_weight * |mean_conf - accuracy|)`; returns `None` and logs warning if `confidences` is empty
    - `plot_reliability_diagram(confidences, labels, model_name, save_dir) -> str | None` saves a PNG to `save_dir` and returns the file path
    - _Requirements: 5.1, 5.2, 5.5_

  - [x]* 5.2 Write property test for `CalibrationAnalyzer` — ECE range
    - **Property 4: ECE range**
    - Generate random lists of `(confidence, label)` pairs with confidence ∈ [0, 1] and label ∈ {0, 1}; assert ECE ∈ [0, 1]
    - **Validates: Requirements 5.1**

  - [x]* 5.3 Write property test for `CalibrationAnalyzer` — perfect calibration
    - **Property 5: ECE perfect calibration**
    - Construct synthetic data where mean confidence equals empirical accuracy in every non-empty bin; assert ECE = 0.0 within 1e-6
    - **Validates: Requirements 5.1**

  - [x]* 5.4 Write unit tests for `CalibrationAnalyzer` in `evaluate/tests/test_calibration.py`
    - Test empty `confidences` returns `None` and logs a warning
    - Test `plot_reliability_diagram` creates a PNG file at the expected path
    - Test ECE is 0.0 for a perfectly calibrated toy example
    - _Requirements: 5.1, 5.2, 5.5_

- [x] 6. Extend `evaluate/dataset_explorer.py` with rich statistics
  - [x] 6.1 Add `compute_ngram_overlap(train_texts, test_texts, n=1) -> float`
    - Returns the fraction of test n-grams that appear in the training set
    - Supports unigrams (`n=1`) and bigrams (`n=2`)
    - _Requirements: 6.1_

  - [x] 6.2 Add `compute_domain_distribution(samples, label_field="source") -> dict[str, float]`
    - Returns a dict of label → proportion summing to 1.0
    - _Requirements: 6.2_

  - [x] 6.3 Add `compute_length_distribution(texts, tokenizer=None, save_path=None) -> list[int]`
    - Uses the BPE tokenizer if available (loaded from `pretrain/config.py`'s `tokenizer_dir`), falls back to whitespace splitting with a logged warning
    - Saves a length distribution histogram PNG to `save_path` if provided
    - _Requirements: 6.3, 6.4, 6.6_

  - [x]* 6.4 Write property test for `compute_ngram_overlap` — range and self-overlap
    - **Property 6: N-gram overlap range**
    - Generate random text corpora; assert overlap ∈ [0, 1] and `overlap(A, A) = 1.0`
    - **Validates: Requirements 6.1**

  - [x]* 6.5 Write property test for `compute_domain_distribution` — sums to 1
    - **Property 7: Domain distribution sums to 1**
    - Generate random labeled sample lists; assert sum of distribution values = 1.0 within 1e-6
    - **Validates: Requirements 6.2**

  - [x]* 6.6 Write unit tests for new dataset explorer functions in `evaluate/tests/test_dataset_explorer.py`
    - Test `compute_ngram_overlap` returns 1.0 when train == test corpus
    - Test `compute_ngram_overlap` returns 0.0 when corpora share no n-grams
    - Test `compute_domain_distribution` proportions sum to 1.0
    - Test `compute_length_distribution` returns non-negative integers
    - Test tokenizer fallback logs a warning when tokenizer cannot be loaded
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

- [x] 7. Checkpoint — verify new modules are importable and unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Add narrative report functions to `evaluate/report.py`
  - [x] 8.1 Add `find_best_models(results) -> dict[str, str]`
    - Returns a dict mapping task_name → model_name with the highest non-None score
    - _Requirements: 8.1_

  - [x] 8.2 Add `wilson_ci(p, n, z=1.96) -> tuple[float, float]`
    - Computes the Wilson score interval; returns `(lower, upper)` satisfying `0 ≤ lower ≤ p ≤ upper ≤ 1`
    - _Requirements: 8.2_

  - [x] 8.3 Add `two_proportion_ztest(n1, k1, n2, k2) -> float`
    - Performs a two-proportion z-test; returns a p-value in [0.0, 1.0]
    - _Requirements: 8.3_

  - [x] 8.4 Add `generate_narrative_report(results, perplexity, calibration, few_shot, output_path, n_samples=1000) -> None`
    - Writes a Markdown file with: results table (Perplexity + ECE columns added), best-model ★ annotations per task, Wilson 95% CI bounds per cell, pairwise z-test p-values (flagged with * when p < 0.05), and a narrative paragraph (≥ 50 words) naming the overall best model, tasks with significant differences, and calibration concerns
    - When only one model is evaluated, skip pairwise testing and note in the narrative that no comparison is possible
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x]* 8.5 Write property test for `find_best_models` — best model correctness
    - **Property 9: Best model correctness**
    - Generate random results dicts with 2–5 models and random scores; assert identified best model has maximum score per task
    - **Validates: Requirements 8.1**

  - [x]* 8.6 Write property test for `wilson_ci` — CI validity
    - **Property 10: Wilson CI validity**
    - Generate random `(p, n)` pairs with `p ∈ [0, 1]`, `n ≥ 1`; assert `0 ≤ lower ≤ p ≤ upper ≤ 1`
    - **Validates: Requirements 8.2**

  - [x]* 8.7 Write property test for `two_proportion_ztest` — p-value range
    - **Property 11: Z-test p-value range**
    - Generate random binary outcome counts; assert p-value ∈ [0.0, 1.0]
    - **Validates: Requirements 8.3**

  - [x]* 8.8 Write unit tests for narrative report in `evaluate/tests/test_narrative_report.py`
    - Test `find_best_models` returns the correct model for a known toy example
    - Test `generate_narrative_report` produces a file containing ≥ 50-word narrative paragraph
    - Test single-model run skips pairwise testing and includes "no comparison" note
    - Test best-model ★ annotation appears in the output Markdown
    - _Requirements: 8.1, 8.4, 8.5_

- [x] 9. Implement `EvaluationPipeline` in `evaluate/pipeline.py`
  - [x] 9.1 Create `evaluate/pipeline.py` with `EvaluationPipeline` class and `EvaluationBundle` TypedDict
    - Define `EvaluationBundle` TypedDict with keys: `benchmark_scores`, `perplexity`, `few_shot_sensitivity`, `calibration`, `weight_analysis`, `activation_analysis`, `dataset_stats`, `local_model`
    - Constructor accepts `config: EvalConfig`, `logger: JSONLogger`
    - _Requirements: 7.6_

  - [x] 9.2 Implement `EvaluationPipeline.run() -> EvaluationBundle`
    - Execute steps in order: dataset stats → benchmark evaluation (HF models) → local model evaluation → perplexity → few-shot sensitivity → calibration → weight analysis → activation analysis → report generation
    - Wrap each step in `try/except`; on unhandled exception log via `JSONLogger` and set the corresponding bundle key to `None`; always continue
    - Log a structured start event and completion event (UTC timestamp + model list) via `JSONLogger`
    - Write all output files to `config.output_dir`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 9.3 Wire local model evaluation into `EvaluationPipeline`
    - When `config.local_checkpoint_path` is set and the file exists, load `LocalModel` via `shared.checkpointing.load_checkpoint` and evaluate on all configured tasks
    - When the path does not exist, log a warning and set `bundle["local_model"] = None`
    - Include `LocalModel` results in `NarrativeReport` table labelled with the checkpoint filename
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 9.4 Write property test for `EvaluationPipeline` — bundle key completeness
    - **Property 8: EvaluationBundle key completeness**
    - Generate random `EvalConfig` instances; inject random step failures via mocks; assert all 8 bundle keys are always present in the returned `EvaluationBundle`
    - **Validates: Requirements 7.5, 7.6**

  - [x]* 9.5 Write unit tests for `EvaluationPipeline` in `evaluate/tests/test_pipeline.py`
    - Test that a step failure sets the corresponding bundle key to `None` without aborting the pipeline
    - Test that all 8 required bundle keys are present in the returned dict
    - Test that start and completion events are logged with UTC timestamp and model list
    - Test that missing `local_checkpoint_path` sets `bundle["local_model"] = None`
    - _Requirements: 2.3, 7.4, 7.5, 7.6_

- [x] 10. Update `evaluate/__init__.py` to expose the public API
  - Import and re-export `EvaluationPipeline`, `EvalConfig`, `PerplexityCalculator`, `FewShotAnalyzer`, `CalibrationAnalyzer`
  - _Requirements: 7.1_

- [x] 11. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Properties 1–11 from design.md)
- Unit tests validate specific examples and edge cases
- All mocking uses `unittest.mock`; no real model loading or `lm_eval` calls in tests
