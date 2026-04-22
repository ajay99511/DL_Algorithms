# Requirements Document

## Introduction

This document specifies improvements to the `backprop` module — a multi-layer perceptron
(MLP) trained on the California Housing regression dataset. The module currently has nine
identified gaps: an empty public API, missing CLIs on `evaluate.py` and `visualize.py`,
a missing R² metric, incomplete test coverage, an unused `checkpoint_every_n_epochs`
config field, no early-stopping mechanism, inability to load a model from a checkpoint
file in `evaluate.py`, and dead-neuron statistics that are computed but never logged
during training. These requirements address each gap in a way that is consistent with the
existing architecture and coding conventions.

## Glossary

- **Backprop_Module**: The Python package located at `backprop/`, containing `model.py`,
  `data.py`, `train.py`, `evaluate.py`, `visualize.py`, `config.py`, and `__init__.py`.
- **MLP**: The `MLP` class defined in `backprop/model.py` — a multi-layer perceptron for
  scalar regression.
- **MLPConfig**: The `MLPConfig` dataclass defined in `backprop/config.py`, which holds
  all hyperparameters and path settings for a training run.
- **Trainer**: The `train()` function in `backprop/train.py` and its surrounding CLI
  entry point.
- **Evaluator**: The `evaluate()` function in `backprop/evaluate.py` and its surrounding
  CLI entry point.
- **Visualizer**: The functions `plot_loss_curves()` and `plot_init_comparison()` in
  `backprop/visualize.py` and their surrounding CLI entry point.
- **Checkpoint**: A `.pt` file written by `shared.checkpointing.save_checkpoint` that
  stores model weights, optimizer state, scheduler state, epoch, step, and best metric.
- **JSONL_Log**: A newline-delimited JSON file written by `shared.logging_utils.JSONLogger`
  during training, where each line is a JSON object with a `"type"` field.
- **R²**: The coefficient of determination, defined as
  `1 − SS_res / SS_tot`, where `SS_res = Σ(y_pred − y_true)²` and
  `SS_tot = Σ(y_true − ȳ)²`. A value of 1.0 indicates a perfect fit; 0.0 indicates the
  model performs no better than predicting the mean.
- **Dead_Neuron_Fraction**: The fraction of ReLU activations that are ≤ 0 for a given
  batch, as returned by `activation_stats()` in `backprop/model.py`.
- **Early_Stopping**: A training-loop mechanism that halts training when the validation
  RMSE has not improved by more than a configurable `early_stopping_delta` for
  `early_stopping_patience` consecutive validation epochs.
- **Public_API**: The set of names exported from `backprop/__init__.py` via `__all__`.

---

## Requirements

### Requirement 1: Public API in `__init__.py`

**User Story:** As a developer importing the `backprop` package, I want a well-defined
public API, so that I can discover and use the module's key classes and functions without
reading every source file.

#### Acceptance Criteria

1. THE Backprop_Module SHALL export `MLP`, `MLPConfig`, `initialize_weights`,
   `activation_stats`, `load_california_housing`, `evaluate`, `plot_loss_curves`, and
   `plot_init_comparison` via `__all__` in `backprop/__init__.py`.
2. WHEN a caller executes `from backprop import MLP`, THE Backprop_Module SHALL make the
   `MLP` class available without raising an `ImportError`.
3. WHEN a caller executes `from backprop import evaluate`, THE Backprop_Module SHALL make
   the `evaluate` function from `backprop/evaluate.py` available without raising an
   `ImportError`.

---

### Requirement 2: CLI for `evaluate.py`

**User Story:** As a practitioner, I want to run `python -m backprop.evaluate` with a
checkpoint path and config file, so that I can evaluate a saved model on the test set
without writing a custom script.

#### Acceptance Criteria

1. WHEN `backprop/evaluate.py` is executed as `python -m backprop.evaluate --checkpoint
   <path> --config <path>`, THE Evaluator SHALL load the MLPConfig from the YAML file,
   reconstruct the MLP from the Checkpoint, run evaluation on the test DataLoader, and
   print the results table.
2. WHEN the `--checkpoint` argument points to a file that does not exist, THE Evaluator
   SHALL print an error message to stderr and exit with a non-zero exit code.
3. WHEN the `--config` argument is omitted, THE Evaluator SHALL use
   `backprop/config.yaml` as the default config path.
4. THE Evaluator SHALL accept an optional `--split` argument with allowed values `test`
   and `val`, defaulting to `test`, and SHALL evaluate on the corresponding DataLoader.

---

### Requirement 3: Checkpoint Loading in `evaluate()`

**User Story:** As a practitioner, I want the `evaluate` function to optionally accept a
checkpoint path instead of a live model object, so that I can evaluate a persisted model
programmatically.

#### Acceptance Criteria

1. THE Evaluator SHALL provide a `load_model_from_checkpoint(checkpoint_path: str,
   config: MLPConfig) -> MLP` helper function that reconstructs an MLP from a Checkpoint
   file using the architecture defined in the MLPConfig.
2. WHEN `load_model_from_checkpoint` is called with a valid Checkpoint path and a
   matching MLPConfig, THE Evaluator SHALL return an MLP whose weights are identical to
   those stored in the Checkpoint.
3. IF the Checkpoint file cannot be loaded (e.g., file not found, corrupt file), THEN THE
   Evaluator SHALL raise a `RuntimeError` with a descriptive message.

---

### Requirement 4: R² Metric

**User Story:** As a practitioner, I want the evaluation output to include R² alongside
RMSE and MAE, so that I have a standard, scale-independent measure of model fit.

#### Acceptance Criteria

1. THE Evaluator SHALL compute R² as `1 − SS_res / SS_tot` where `SS_res = Σ(y_pred −
   y_true)²` and `SS_tot = Σ(y_true − ȳ)²`, with ȳ computed over the evaluated split.
2. THE Evaluator SHALL return a named tuple or dataclass `EvalResult` with fields `rmse`,
   `mae`, and `r2`, replacing the current `(float, float)` return type.
3. WHEN `SS_tot` is zero (all targets are identical), THE Evaluator SHALL set `r2` to
   `0.0` rather than raising a division-by-zero error.
4. THE Evaluator SHALL include R² in the printed results table alongside RMSE and MAE.
5. THE Trainer SHALL log `r2` in the `val_epoch` JSONL_Log entries alongside `val_rmse`
   and `val_mae`.

---

### Requirement 5: CLI for `visualize.py`

**User Story:** As a practitioner, I want to run `python -m backprop.visualize` from the
command line, so that I can generate plots from experiment logs without writing a custom
script — as documented in the README.

#### Acceptance Criteria

1. WHEN `backprop/visualize.py` is executed as `python -m backprop.visualize
   loss-curves --log <path> --output <path>`, THE Visualizer SHALL call
   `plot_loss_curves` with the provided paths and save the resulting PNG.
2. WHEN `backprop/visualize.py` is executed as `python -m backprop.visualize
   init-comparison --logs <path1> <path2> ... --labels <l1> <l2> ... --output <path>`,
   THE Visualizer SHALL call `plot_init_comparison` with the provided arguments and save
   the resulting PNG.
3. WHEN a `--log` or `--logs` path does not exist, THE Visualizer SHALL print an error
   message to stderr and exit with a non-zero exit code.
4. WHEN the `--output` parent directory does not exist, THE Visualizer SHALL create it
   before saving the PNG.

---

### Requirement 6: Periodic Checkpointing via `checkpoint_every_n_epochs`

**User Story:** As a practitioner, I want periodic checkpoints saved every N epochs in
addition to the best checkpoint, so that I can resume training from an intermediate epoch
if the best checkpoint is corrupted or I want to inspect training dynamics.

#### Acceptance Criteria

1. WHEN `MLPConfig.checkpoint_every_n_epochs` is set to a positive integer N, THE
   Trainer SHALL save a Checkpoint named `epoch_{epoch:04d}.pt` in `checkpoint_dir` at
   the end of every epoch that is a multiple of N (i.e., `(epoch + 1) % N == 0`).
2. WHEN `MLPConfig.checkpoint_every_n_epochs` is set to 0, THE Trainer SHALL skip
   periodic checkpointing and save only the best Checkpoint.
3. THE Trainer SHALL log a JSONL_Log entry of type `"periodic_checkpoint"` containing
   `epoch`, `step`, and `path` whenever a periodic Checkpoint is saved.
4. THE Trainer SHALL continue to save the best Checkpoint independently of the periodic
   checkpointing schedule.

---

### Requirement 7: Early Stopping

**User Story:** As a practitioner, I want training to stop automatically when validation
RMSE stops improving, so that I avoid wasting compute on epochs that no longer reduce
overfitting.

#### Acceptance Criteria

1. THE MLPConfig SHALL include two new fields: `early_stopping_patience: int` (default
   `0`) and `early_stopping_delta: float` (default `0.0`).
2. WHEN `early_stopping_patience` is 0, THE Trainer SHALL disable early stopping and
   always train for `max_epochs` epochs.
3. WHEN `early_stopping_patience` is a positive integer P, THE Trainer SHALL halt
   training after P consecutive validation epochs in which `val_rmse` has not decreased
   by more than `early_stopping_delta` compared to the best observed `val_rmse`.
4. WHEN early stopping is triggered, THE Trainer SHALL log a JSONL_Log entry of type
   `"early_stop"` containing `epoch`, `step`, and `best_val_rmse`.
5. WHEN early stopping is triggered, THE Trainer SHALL log an INFO-level message
   indicating the epoch at which training stopped and the best `val_rmse` achieved.

---

### Requirement 8: Activation Statistics Logging During Training

**User Story:** As a researcher, I want dead-neuron fractions logged to the JSONL log
during training, so that I can diagnose dying-ReLU problems without adding custom
instrumentation.

#### Acceptance Criteria

1. WHEN `MLPConfig.log_every_n_steps` is reached, THE Trainer SHALL call
   `activation_stats(model, x)` on the current micro-batch and log a JSONL_Log entry of
   type `"activation_stats"` containing `epoch`, `step`, and a `layers` dict mapping
   each layer name to its `mean`, `std`, and `dead_fraction`.
2. THE Trainer SHALL compute activation statistics using `torch.no_grad()` to avoid
   affecting the gradient computation of the current training step.
3. WHEN all layers have a `dead_fraction` of 1.0 for three consecutive
   `activation_stats` log entries, THE Trainer SHALL emit a WARNING-level log message
   indicating that all neurons appear to be dead.

---

### Requirement 9: Test Coverage Gaps

**User Story:** As a developer, I want comprehensive tests for all modules in the
`backprop` package, so that regressions are caught automatically in CI.

#### Acceptance Criteria

1. THE Backprop_Module SHALL include a smoke test in `backprop/tests/test_train.py` that
   calls `train()` with a minimal MLPConfig (e.g., `max_epochs=2`, `hidden_dims=[8]`)
   and asserts that the function completes without raising an exception and that a best
   Checkpoint file is created on disk.
2. THE Backprop_Module SHALL include tests in `backprop/tests/test_evaluate.py` that
   verify: (a) `evaluate()` returns an `EvalResult` with finite `rmse`, `mae`, and `r2`
   fields; (b) `r2` is in the range `(−∞, 1.0]`; (c) `load_model_from_checkpoint`
   round-trips model weights correctly.
3. THE Backprop_Module SHALL include a property-based test in
   `backprop/tests/test_evaluate.py` that, for randomly generated predictions and
   targets, verifies that `r2` satisfies `r2 ≤ 1.0` and that `rmse ≥ 0.0` and
   `mae ≥ 0.0`.
4. THE Backprop_Module SHALL include a test in `backprop/tests/test_visualize.py` that
   calls `plot_loss_curves` and `plot_init_comparison` with synthetic JSONL data and
   asserts that the output PNG files are created on disk.
5. THE Backprop_Module SHALL include a test in `backprop/tests/test_data.py` that
   constructs a DataLoader via `load_california_housing` with a specific `batch_size` and
   asserts that every batch returned by the train DataLoader has a first-dimension size
   equal to `batch_size` (except possibly the last batch).
