# Requirements Document

## Introduction

The `evaluate` module (Project 6) wraps `lm-evaluation-harness` to benchmark GPT-2 and
Pythia-160M on ARC, HellaSwag, MMLU, and TruthfulQA. The current implementation has eight
gaps compared to what professional evaluation teams produce: no CLI entry point, no evaluation
of the project's own locally-trained GPT model, no perplexity metric, hardcoded shot counts
with no sensitivity analysis, no calibration analysis, superficial dataset statistics, analysis
utilities that are never called from the main pipeline, and reports that contain tables but no
analytical narrative or statistical testing.

These requirements address all eight gaps. The target environment is CPU-only (AMD Ryzen 9
laptop). The locally-trained model lives in `pretrain/` and is loaded via
`shared/checkpointing.load_checkpoint`.

---

## Glossary

- **CLI**: Command-line interface; the entry point invoked via `python -m evaluate.evaluate`.
- **EvalConfig**: The `EvalConfig` dataclass in `evaluate/config.py` that holds model lists,
  task definitions, and output paths.
- **EvaluationPipeline**: The top-level orchestrator that runs benchmark evaluation, perplexity
  measurement, few-shot sensitivity analysis, calibration analysis, weight analysis, activation
  analysis, and report generation in a single coordinated run.
- **LocalModel**: The GPT-style `Transformer` defined in `pretrain/model.py`, loaded from a
  `.pt` checkpoint produced by `pretrain/train.py`.
- **HFModel**: A HuggingFace pretrained model (e.g. `gpt2`, `EleutherAI/pythia-160m`) loaded
  via `lm-evaluation-harness`.
- **Perplexity**: `exp(mean cross-entropy loss)` computed over a held-out token sequence; a
  primary language model quality metric. Always ≥ 1.0 for valid inputs.
- **FewShotSensitivityAnalysis**: Running the same benchmark task at multiple shot counts
  (e.g. 0, 1, 5) and recording how accuracy changes.
- **ECE**: Expected Calibration Error — the weighted mean absolute difference between model
  confidence and empirical accuracy across equal-width confidence bins.
- **ReliabilityDiagram**: A plot of mean confidence vs. empirical accuracy per bin, used to
  visualise calibration.
- **NGramOverlap**: The fraction of n-grams in a test split that also appear in the
  corresponding training split; used to detect benchmark contamination.
- **DomainDistribution**: The proportion of dataset samples belonging to each identified
  domain or source category.
- **LengthDistribution**: The distribution of tokenized sequence lengths across dataset
  samples.
- **WeightAnalysis**: Frobenius norm, SVD singular value spectrum, and dead neuron ratio
  computations from `evaluate/weight_analysis.py`.
- **ActivationAnalysis**: Forward-hook activation capture and histogram generation from
  `evaluate/activation_analysis.py`.
- **EvaluationBundle**: The unified output dict produced by `EvaluationPipeline` containing
  benchmark scores, perplexity, few-shot sensitivity results, calibration metrics, weight
  analysis, and activation analysis for all evaluated models.
- **NarrativeReport**: A Markdown report that includes a written comparison of models,
  identifies the best-performing model per task, and includes confidence intervals and
  statistical significance results.
- **JSONLogger**: The structured logger from `shared/logging_utils.py`.

---

## Requirements

### Requirement 1: CLI Entry Point

**User Story:** As a developer, I want to run `python -m evaluate.evaluate` from the command
line, so that I can launch the full evaluation pipeline without writing a driver script.

#### Acceptance Criteria

1. THE `evaluate.evaluate` module SHALL provide an `if __name__ == "__main__"` block that
   invokes `EvaluationPipeline`.
2. WHEN `python -m evaluate.evaluate --help` is executed, THE CLI SHALL print a usage message
   listing all supported flags and exit with code 0.
3. WHEN `python -m evaluate.evaluate --config <path>` is executed with a valid YAML config
   path, THE CLI SHALL load `EvalConfig` from that file and pass it to `EvaluationPipeline`.
4. IF `python -m evaluate.evaluate --config <path>` is executed with a path that does not
   exist, THEN THE CLI SHALL print a descriptive error message to stderr and exit with a
   non-zero exit code.
5. WHEN `python -m evaluate.evaluate` is executed with no flags, THE CLI SHALL use the default
   `EvalConfig` values and proceed with the full pipeline run.

---

### Requirement 2: Local Model Evaluation

**User Story:** As a developer, I want the evaluation pipeline to benchmark my locally-trained
GPT model alongside GPT-2 and Pythia-160M, so that I can compare my training results against
established baselines in a single report.

#### Acceptance Criteria

1. THE `EvaluationPipeline` SHALL accept a `local_checkpoint_path` parameter that, when
   provided, loads the `LocalModel` from that path using
   `shared.checkpointing.load_checkpoint`.
2. WHEN `local_checkpoint_path` is provided and the checkpoint file exists, THE
   `EvaluationPipeline` SHALL evaluate the `LocalModel` on all configured benchmark tasks
   using the same shot counts as the `HFModel` baselines.
3. WHEN `local_checkpoint_path` is provided and the checkpoint file does not exist, THE
   `EvaluationPipeline` SHALL log a warning via `JSONLogger`, skip local model evaluation,
   and continue evaluating the `HFModel` baselines.
4. THE `NarrativeReport` SHALL include the `LocalModel` results in the same table as the
   `HFModel` baselines, labelled with the checkpoint filename.
5. THE `EvaluationBundle` SHALL contain a `"local_model"` key whose value is the benchmark
   results dict for the `LocalModel`, or `None` if no checkpoint was provided.

---

### Requirement 3: Perplexity Metric

**User Story:** As a developer, I want the pipeline to compute perplexity on a held-out corpus
for every evaluated model, so that I have a primary language model quality metric alongside
classification benchmark accuracy.

#### Acceptance Criteria

1. THE `EvaluationPipeline` SHALL compute perplexity for each model on a configurable
   held-out text corpus by computing `exp(mean cross-entropy loss)` over tokenized sequences.
2. WHEN a model is evaluated, THE `Perplexity_Calculator` SHALL return a perplexity value
   that is a finite float greater than or equal to 1.0.
3. WHEN the held-out corpus contains sequences longer than the model's `context_length`, THE
   `Perplexity_Calculator` SHALL split each sequence into non-overlapping windows of
   `context_length` tokens and average the loss across all windows.
4. THE `EvaluationBundle` SHALL contain a `"perplexity"` key mapping each model name to its
   perplexity value.
5. THE `NarrativeReport` SHALL include a perplexity column in the results table.
6. IF the held-out corpus is empty or cannot be loaded, THEN THE `Perplexity_Calculator`
   SHALL log a warning via `JSONLogger` and return `None` for that model's perplexity.

#### Correctness Properties

- **P3.1 — Perplexity lower bound**: For any non-empty token sequence and any model,
  `Perplexity_Calculator` SHALL return a value ≥ 1.0. (Cross-entropy is non-negative, so
  `exp(loss) ≥ 1`.)
- **P3.2 — Finite output**: For any valid token sequence with no padding tokens, the returned
  perplexity SHALL be a finite float (not `inf`, not `nan`).
- **P3.3 — Window invariant**: For any sequence of length `L > context_length`, splitting
  into windows and averaging loss SHALL produce the same result regardless of whether windows
  are processed sequentially or in a batch.

---

### Requirement 4: Few-Shot Sensitivity Analysis

**User Story:** As a developer, I want the pipeline to run each benchmark task at multiple
shot counts and report how accuracy changes, so that I can understand whether my model's
performance is sensitive to the number of in-context examples.

#### Acceptance Criteria

1. THE `EvalConfig` SHALL include a `shot_counts` field mapping each task name to a list of
   integer shot counts to evaluate (e.g. `{"arc_challenge": [0, 1, 5, 25]}`).
2. WHEN `EvaluationPipeline` runs, THE `FewShotAnalyzer` SHALL evaluate each task at every
   shot count in `shot_counts[task_name]` and record the accuracy for each `(task, n_shots)`
   pair.
3. THE `EvaluationBundle` SHALL contain a `"few_shot_sensitivity"` key whose value is a dict
   mapping `(model_name, task_name, n_shots)` to accuracy.
4. THE `NarrativeReport` SHALL include a few-shot sensitivity table showing accuracy per task
   per shot count for each model.
5. IF a specific `(task, n_shots)` evaluation fails, THEN THE `FewShotAnalyzer` SHALL log the
   error via `JSONLogger`, record `None` for that entry, and continue with remaining
   combinations.

#### Correctness Properties

- **P4.1 — Result completeness**: For any configuration with `M` models, `T` tasks, and
  `S_t` shot counts for task `t`, the `"few_shot_sensitivity"` dict in `EvaluationBundle`
  SHALL contain exactly `M × Σ S_t` entries (one per `(model, task, n_shots)` triple),
  with `None` for failed evaluations.
- **P4.2 — Shot count key invariant**: For any `(model, task, n_shots)` key in the
  `"few_shot_sensitivity"` dict, `n_shots` SHALL be a non-negative integer that appears in
  `shot_counts[task]`.

---

### Requirement 5: Calibration Analysis

**User Story:** As a developer, I want the pipeline to compute Expected Calibration Error and
generate reliability diagrams for each model, so that I can assess whether model confidence
scores are trustworthy.

#### Acceptance Criteria

1. THE `Calibration_Analyzer` SHALL compute ECE for each model by partitioning model
   confidence scores into 10 equal-width bins over [0, 1] and computing the weighted mean
   absolute difference between mean confidence and empirical accuracy per bin.
2. THE `Calibration_Analyzer` SHALL generate a `ReliabilityDiagram` PNG for each model,
   plotting mean confidence vs. empirical accuracy per bin, and save it to the configured
   `output_dir`.
3. THE `EvaluationBundle` SHALL contain a `"calibration"` key mapping each model name to a
   dict with keys `"ece"` (float) and `"reliability_diagram_path"` (str).
4. THE `NarrativeReport` SHALL include an ECE column in the results table.
5. IF a model produces no confidence scores (e.g. the task does not expose logits), THEN THE
   `Calibration_Analyzer` SHALL log a warning via `JSONLogger` and record `None` for that
   model's ECE.

#### Correctness Properties

- **P5.1 — ECE range**: For any list of `(confidence, label)` pairs where confidence ∈ [0, 1]
  and label ∈ {0, 1}, THE `Calibration_Analyzer` SHALL return an ECE value in [0, 1].
- **P5.2 — Perfect calibration**: For any set of predictions where confidence equals empirical
  accuracy in every bin, THE `Calibration_Analyzer` SHALL return ECE = 0.0 (within floating-
  point tolerance of 1e-6).
- **P5.3 — Bin weight normalization**: The sum of bin weights used in the ECE computation
  SHALL equal 1.0 for any non-empty input (each bin weight = bin_count / total_count).

---

### Requirement 6: Rich Dataset Statistics

**User Story:** As a developer, I want the dataset explorer to compute n-gram overlap,
domain distribution, and tokenizer-based length distributions, so that I can detect benchmark
contamination and understand the composition of training and evaluation data.

#### Acceptance Criteria

1. THE `Dataset_Explorer` SHALL compute the unigram and bigram overlap ratio between a
   training split and a test split, defined as the fraction of test n-grams that also appear
   in the training split.
2. THE `Dataset_Explorer` SHALL compute a domain distribution dict mapping each domain or
   source label to its proportion of the total sample count.
3. THE `Dataset_Explorer` SHALL compute sequence length distributions using the project's
   BPE tokenizer (loaded from `pretrain/config.py`'s `tokenizer_dir`) rather than
   whitespace splitting.
4. THE `Dataset_Explorer` SHALL generate a length distribution histogram PNG and save it to
   the configured `output_dir`.
5. THE `EvaluationBundle` SHALL contain a `"dataset_stats"` key with sub-keys `"ngram_overlap"`,
   `"domain_distribution"`, and `"length_distribution"`.
6. IF the tokenizer cannot be loaded, THEN THE `Dataset_Explorer` SHALL fall back to
   whitespace-split token counts and log a warning via `JSONLogger`.

#### Correctness Properties

- **P6.1 — N-gram overlap range**: For any two non-empty text corpora, the n-gram overlap
  ratio computed by `Dataset_Explorer` SHALL be in [0.0, 1.0].
- **P6.2 — N-gram overlap symmetry is NOT required** (test-in-train vs. train-in-test differ),
  but overlap(A, A) SHALL equal 1.0 for any non-empty corpus A.
- **P6.3 — Domain distribution sums to 1**: For any non-empty list of labeled samples, the
  values of the domain distribution dict SHALL sum to 1.0 (within floating-point tolerance
  of 1e-6).
- **P6.4 — Length non-negativity**: For any list of text strings, all computed sequence
  lengths SHALL be non-negative integers.

---

### Requirement 7: Unified Orchestration Pipeline

**User Story:** As a developer, I want a single `EvaluationPipeline` that runs benchmark
evaluation, perplexity, few-shot sensitivity, calibration, weight analysis, and activation
analysis in one command, so that I get a complete picture of model quality without manually
coordinating separate scripts.

#### Acceptance Criteria

1. THE `EvaluationPipeline` SHALL call `WeightAnalysis` for each evaluated model and include
   the results in `EvaluationBundle` under the key `"weight_analysis"`.
2. THE `EvaluationPipeline` SHALL call `ActivationAnalysis` for each evaluated model and
   include the results in `EvaluationBundle` under the key `"activation_analysis"`.
3. THE `EvaluationPipeline` SHALL write all output files (CSV report, Markdown report,
   reliability diagram PNGs, weight norm PNG, singular value PNG, activation histogram PNGs)
   to the directory specified by `EvalConfig.output_dir`.
4. THE `EvaluationPipeline` SHALL log a structured start event and a structured completion
   event to `JSONLogger`, each containing a UTC timestamp and the list of models evaluated.
5. IF any individual analysis step raises an unhandled exception, THEN THE
   `EvaluationPipeline` SHALL log the error via `JSONLogger`, skip that step, and continue
   with the remaining steps.
6. THE `EvaluationBundle` SHALL always contain the keys `"benchmark_scores"`,
   `"perplexity"`, `"few_shot_sensitivity"`, `"calibration"`, `"weight_analysis"`,
   `"activation_analysis"`, and `"dataset_stats"`, with `None` values for any step that
   failed or was skipped.

#### Correctness Properties

- **P7.1 — Bundle key completeness**: For any valid `EvalConfig` with at least one model,
  the `EvaluationBundle` returned by `EvaluationPipeline` SHALL always contain all seven
  required top-level keys, regardless of which individual steps succeed or fail.

---

### Requirement 8: Cross-Model Comparison Narrative

**User Story:** As a developer, I want the report to identify the best-performing model per
task, include confidence intervals, and flag statistically significant differences, so that
my portfolio shows I understand rigorous model comparison rather than just table generation.

#### Acceptance Criteria

1. THE `Report_Generator` SHALL identify the model with the highest score for each task and
   include its name in the `NarrativeReport` with a "best" annotation.
2. THE `Report_Generator` SHALL compute 95% confidence intervals for each model's score on
   each task using the Wilson score interval for proportions, and include the interval bounds
   in the `NarrativeReport`.
3. THE `Report_Generator` SHALL perform a two-proportion z-test between each pair of models
   for each task and include the p-value in the `NarrativeReport`, flagging pairs where
   p < 0.05 as statistically significant.
4. THE `NarrativeReport` SHALL include a written summary paragraph (minimum 50 words) that
   names the overall best model, identifies tasks where models differ significantly, and
   notes any calibration concerns.
5. IF only one model is evaluated, THEN THE `Report_Generator` SHALL skip pairwise
   significance testing and note in the narrative that no comparison is possible.

#### Correctness Properties

- **P8.1 — Best model correctness**: For any results dict with two or more models and
  non-None scores, the model identified as best for a given task by `Report_Generator` SHALL
  have the maximum score for that task among all models with non-None scores.
- **P8.2 — Confidence interval validity**: For any score `p` ∈ [0, 1] and sample count
  `n` ≥ 1, the Wilson score interval [lower, upper] computed by `Report_Generator` SHALL
  satisfy `0 ≤ lower ≤ p ≤ upper ≤ 1`.
- **P8.3 — P-value range**: For any two sets of binary outcomes, the p-value returned by the
  two-proportion z-test in `Report_Generator` SHALL be in [0.0, 1.0].
