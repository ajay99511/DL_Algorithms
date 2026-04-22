# Design Document

## backprop-improvements

---

## Overview

This design covers nine targeted improvements to the `backprop/` module — a multi-layer
perceptron trained on the California Housing regression dataset. The changes are purely
additive or minimally invasive: no existing public interfaces are broken, and all new
code follows the conventions already established in the module (dataclasses for config,
`JSONLogger` for structured logging, `shared.checkpointing` for persistence).

The nine areas of work are:

1. Public API (`__init__.py` `__all__`)
2. CLI for `evaluate.py`
3. `load_model_from_checkpoint()` helper
4. `EvalResult` dataclass with R² metric
5. CLI for `visualize.py`
6. Periodic checkpointing wired to `checkpoint_every_n_epochs`
7. Early stopping (`early_stopping_patience` / `early_stopping_delta`)
8. Activation-stats logging during training
9. Test coverage gaps filled

---

## Architecture

The module follows a flat, single-package layout. All changes stay within `backprop/`
and its `tests/` subdirectory. No new files are added outside that tree except the test
files called for by Requirement 9.

```
backprop/
├── __init__.py          ← Req 1: populate __all__
├── config.py            ← Req 7: add early_stopping_* fields
├── data.py              ← unchanged
├── evaluate.py          ← Req 2, 3, 4: CLI + load_model_from_checkpoint + EvalResult
├── model.py             ← unchanged
├── train.py             ← Req 6, 7, 8: periodic ckpt + early stop + activation logging
├── visualize.py         ← Req 5: CLI subcommands
└── tests/
    ├── test_data.py     ← Req 9.5: batch_size property test
    ├── test_evaluate.py ← Req 9.2, 9.3: EvalResult + PBT metric bounds
    ├── test_model.py    ← unchanged
    ├── test_train.py    ← Req 9.1: smoke test
    └── test_visualize.py← Req 9.4: plot output files
```

Data flow is unchanged: `data.py` → `model.py` → `train.py` → checkpoints/logs →
`evaluate.py` / `visualize.py`.

---

## Components and Interfaces

### 1. `backprop/__init__.py` — Public API

Populate `__all__` and add the corresponding imports so that
`from backprop import <name>` works for every exported symbol.

```python
__all__ = [
    "MLP",
    "MLPConfig",
    "initialize_weights",
    "activation_stats",
    "load_california_housing",
    "evaluate",
    "plot_loss_curves",
    "plot_init_comparison",
]
```

### 2. `backprop/evaluate.py` — CLI + `load_model_from_checkpoint` + `EvalResult`

#### `EvalResult` dataclass

```python
@dataclass
class EvalResult:
    rmse: float
    mae: float
    r2: float
```

Replaces the `(float, float)` return type of `evaluate()`.

#### `load_model_from_checkpoint`

```python
def load_model_from_checkpoint(checkpoint_path: str, config: MLPConfig) -> MLP:
    """
    Reconstruct an MLP from a saved checkpoint.

    Raises RuntimeError if the file cannot be loaded.
    """
```

Implementation:
- Construct `MLP(input_dim=8, hidden_dims=config.hidden_dims, dropout=config.dropout)`
- Call `torch.load(checkpoint_path, map_location="cpu")` inside a try/except; on
  `FileNotFoundError` or `RuntimeError` re-raise as `RuntimeError` with a descriptive
  message.
- Call `model.load_state_dict(checkpoint["model_state_dict"])`.
- Return the model in eval mode.

#### Updated `evaluate()`

```python
def evaluate(model: MLP, loader: DataLoader) -> EvalResult:
```

R² computation:
```
SS_res = Σ(y_pred − y_true)²
SS_tot = Σ(y_true − ȳ)²
r2 = 1 − SS_res / SS_tot   (0.0 when SS_tot == 0)
```

Updated `_print_results_table` adds an R² row.

#### CLI (`__main__` block)

```
python -m backprop.evaluate --checkpoint <path> [--config <path>] [--split test|val]
```

- `--config` defaults to `backprop/config.yaml`
- `--split` defaults to `test`
- Missing checkpoint → `sys.stderr` message + `sys.exit(1)`

### 3. `backprop/visualize.py` — CLI subcommands

```
python -m backprop.visualize loss-curves --log <path> --output <path>
python -m backprop.visualize init-comparison --logs <p1> <p2> ... --labels <l1> <l2> ... --output <path>
```

Uses `argparse` with `add_subparsers`. Missing log paths → stderr + `sys.exit(1)`.
Output parent directory is created with `Path.mkdir(parents=True, exist_ok=True)` (already
done inside the plot functions, so the CLI just delegates).

### 4. `backprop/config.py` — Early-stopping fields

```python
@dataclass
class MLPConfig(BaseConfig):
    ...
    early_stopping_patience: int = 0
    early_stopping_delta: float = 0.0
```

`checkpoint_every_n_epochs` already exists in `BaseConfig` (default `1`); no change
needed there.

### 5. `backprop/train.py` — Periodic checkpointing, early stopping, activation logging

#### Periodic checkpointing

At the end of each epoch, after the best-checkpoint logic:

```python
if config.checkpoint_every_n_epochs > 0 and (epoch + 1) % config.checkpoint_every_n_epochs == 0:
    periodic_path = str(Path(config.checkpoint_dir) / f"epoch_{epoch:04d}.pt")
    save_checkpoint(periodic_path, model, optimizer, scheduler,
                    epoch=epoch, step=global_step, best_metric=best_val_rmse)
    json_logger.log({"type": "periodic_checkpoint", "epoch": epoch,
                     "step": global_step, "path": periodic_path})
```

#### Early stopping

State tracked in the training loop:

```python
patience_counter: int = 0
best_val_rmse: float = float("inf")
```

After each validation:

```python
if val_rmse < best_val_rmse - config.early_stopping_delta:
    best_val_rmse = val_rmse
    patience_counter = 0
    # ... save best checkpoint ...
else:
    if config.early_stopping_patience > 0:
        patience_counter += 1
        if patience_counter >= config.early_stopping_patience:
            json_logger.log({"type": "early_stop", "epoch": epoch,
                             "step": global_step, "best_val_rmse": best_val_rmse})
            logger.info("Early stopping at epoch %d. Best val_rmse=%.4f", epoch + 1, best_val_rmse)
            break
```

#### Activation-stats logging

Inside the optimizer step block, when `global_step % config.log_every_n_steps == 0`:

```python
with torch.no_grad():
    stats = activation_stats(model, x)
json_logger.log({"type": "activation_stats", "epoch": epoch,
                 "step": global_step, "layers": stats})
```

Dead-neuron warning: maintain a rolling deque of the last 3 activation-stats entries;
if all layers in all 3 entries have `dead_fraction == 1.0`, emit `logger.warning(...)`.

#### R² in val_epoch log entries

The training loop's internal `evaluate()` call is replaced with the updated
`backprop.evaluate.evaluate()` (or an equivalent inline computation) so that `r2` is
available for logging:

```python
json_logger.log({
    "type": "val_epoch",
    ...,
    "val_rmse": result.rmse,
    "val_mae": result.mae,
    "val_r2": result.r2,
})
```

---

## Data Models

### `EvalResult`

| Field | Type  | Description                        |
|-------|-------|------------------------------------|
| rmse  | float | Root Mean Squared Error (≥ 0)      |
| mae   | float | Mean Absolute Error (≥ 0)          |
| r2    | float | Coefficient of determination (≤ 1) |

### `MLPConfig` additions

| Field                    | Type  | Default | Description                                      |
|--------------------------|-------|---------|--------------------------------------------------|
| early_stopping_patience  | int   | 0       | 0 = disabled; >0 = epochs without improvement   |
| early_stopping_delta     | float | 0.0     | Minimum improvement to reset patience counter    |

`checkpoint_every_n_epochs` is inherited from `BaseConfig` (default `1`).

### JSONL log entry types (new/updated)

| type                  | New fields added                                      |
|-----------------------|-------------------------------------------------------|
| `val_epoch`           | `val_r2`                                              |
| `activation_stats`    | `epoch`, `step`, `layers` (dict of layer → stats)    |
| `periodic_checkpoint` | `epoch`, `step`, `path`                               |
| `early_stop`          | `epoch`, `step`, `best_val_rmse`                      |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid
executions of a system — essentially, a formal statement about what the system should do.
Properties serve as the bridge between human-readable specifications and
machine-verifiable correctness guarantees.*

### Property 1: Checkpoint round-trip preserves weights

*For any* valid `MLPConfig` (varying `hidden_dims` and `dropout`), saving an MLP to a
checkpoint file and then loading it back via `load_model_from_checkpoint` SHALL produce
an MLP whose weight tensors are element-wise identical to the original.

**Validates: Requirements 3.1, 3.2**

---

### Property 2: EvalResult metric bounds

*For any* set of model predictions and ground-truth targets (randomly generated floats),
the `EvalResult` returned by `evaluate()` SHALL satisfy `rmse ≥ 0.0`, `mae ≥ 0.0`, and
`r2 ≤ 1.0`.

**Validates: Requirements 4.1, 4.2, 9.3**

---

### Property 3: Periodic checkpoint files match schedule

*For any* positive integer N and training run of E epochs, the set of
`epoch_XXXX.pt` files written to `checkpoint_dir` SHALL be exactly those epochs where
`(epoch + 1) % N == 0`.

**Validates: Requirements 6.1**

---

### Property 4: Early stopping halts after patience epochs

*For any* positive patience value P, when validation RMSE never improves (i.e., every
epoch returns the same or worse RMSE), training SHALL halt after exactly P consecutive
non-improving epochs, not before and not after.

**Validates: Requirements 7.3**

---

### Property 5: DataLoader batch size invariant

*For any* `batch_size` B passed to `load_california_housing`, every batch returned by
the train DataLoader except possibly the last SHALL have a first-dimension size equal to
B.

**Validates: Requirements 9.5**

---

## Error Handling

| Scenario | Component | Behaviour |
|---|---|---|
| `--checkpoint` path missing | `evaluate.py` CLI | Print to stderr, `sys.exit(1)` |
| `--log`/`--logs` path missing | `visualize.py` CLI | Print to stderr, `sys.exit(1)` |
| Checkpoint file not found in `load_model_from_checkpoint` | `evaluate.py` | Raise `RuntimeError` with path in message |
| Corrupt checkpoint in `load_model_from_checkpoint` | `evaluate.py` | Catch `RuntimeError` from `torch.load`, re-raise as `RuntimeError` |
| `SS_tot == 0` in R² computation | `evaluate.py` | Return `r2 = 0.0` |
| NaN loss during training | `train.py` | Already handled; log warning, skip optimizer step |
| All neurons dead for 3 consecutive log steps | `train.py` | Emit `logger.warning` |

---

## Testing Strategy

The project uses **Hypothesis** for property-based testing (already established in
`test_data.py` and `test_model.py`). All new property tests use `@given` + `@settings`
with `max_examples=100`.

### Unit / example tests

- `test_train.py` — smoke test: `train()` with `max_epochs=2, hidden_dims=[8]` completes
  and writes `best.pt`.
- `test_evaluate.py` — example tests: `EvalResult` fields are finite; `r2 == 0.0` when
  all targets are identical; `load_model_from_checkpoint` raises `RuntimeError` on bad
  path; CLI defaults are correct.
- `test_visualize.py` — example tests: `plot_loss_curves` and `plot_init_comparison`
  write PNG files to disk given synthetic JSONL input.
- `test_data.py` — example test: `load_california_housing` returns three `DataLoader`
  instances (already exists).

### Property-based tests (Hypothesis, min 100 examples each)

| Test | Property | Tag |
|---|---|---|
| `test_checkpoint_roundtrip` | Property 1 | `Feature: backprop-improvements, Property 1` |
| `test_eval_metric_bounds` | Property 2 | `Feature: backprop-improvements, Property 2` |
| `test_periodic_checkpoint_schedule` | Property 3 | `Feature: backprop-improvements, Property 3` |
| `test_early_stopping_patience` | Property 4 | `Feature: backprop-improvements, Property 4` |
| `test_batch_size_invariant` | Property 5 | `Feature: backprop-improvements, Property 5` |

Properties 3 and 4 use a minimal `MLPConfig` (2 epochs, tiny hidden dims) to keep
iteration cost low. Properties 1 and 2 are pure-function tests with no I/O.
