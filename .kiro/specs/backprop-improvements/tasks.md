# Tasks

## Task List

- [x] 1. Populate `backprop/__init__.py` public API
  - [x] 1.1 Add imports for all eight exported symbols
  - [x] 1.2 Define `__all__` listing all eight names
  - [x] 1.3 Verify `from backprop import MLP` and `from backprop import evaluate` work without `ImportError`

- [x] 2. Add `EvalResult` dataclass and update `evaluate()` in `backprop/evaluate.py`
  - [x] 2.1 Define `EvalResult` dataclass with `rmse`, `mae`, `r2` fields
  - [x] 2.2 Update `evaluate()` to accumulate `SS_res` and `SS_tot` and compute R²; guard against `SS_tot == 0`
  - [x] 2.3 Change `evaluate()` return type from `tuple[float, float]` to `EvalResult`
  - [x] 2.4 Update `_print_results_table` to include an R² row

- [x] 3. Add `load_model_from_checkpoint()` to `backprop/evaluate.py`
  - [x] 3.1 Implement `load_model_from_checkpoint(checkpoint_path, config) -> MLP`
  - [x] 3.2 Raise `RuntimeError` with descriptive message on `FileNotFoundError` or corrupt file
  - [x] 3.3 Return model in eval mode with weights loaded from checkpoint

- [x] 4. Add CLI to `backprop/evaluate.py`
  - [x] 4.1 Add `argparse` block with `--checkpoint`, `--config` (default `backprop/config.yaml`), and `--split` (default `test`, choices `test`/`val`) arguments
  - [x] 4.2 On missing checkpoint file, print to stderr and `sys.exit(1)`
  - [x] 4.3 Wire CLI to `load_model_from_checkpoint`, `load_california_housing`, and `evaluate()`

- [x] 5. Add CLI to `backprop/visualize.py`
  - [x] 5.1 Add `argparse` with two subcommands: `loss-curves` and `init-comparison`
  - [x] 5.2 `loss-curves` subcommand: `--log` and `--output` args; call `plot_loss_curves`
  - [x] 5.3 `init-comparison` subcommand: `--logs` (nargs=+), `--labels` (nargs=+), `--output` args; call `plot_init_comparison`
  - [x] 5.4 On missing log path(s), print to stderr and `sys.exit(1)`

- [x] 6. Add early-stopping fields to `MLPConfig` in `backprop/config.py`
  - [x] 6.1 Add `early_stopping_patience: int = 0` field
  - [x] 6.2 Add `early_stopping_delta: float = 0.0` field

- [x] 7. Wire periodic checkpointing in `backprop/train.py`
  - [x] 7.1 At end of each epoch, check `checkpoint_every_n_epochs > 0` and `(epoch + 1) % checkpoint_every_n_epochs == 0`
  - [x] 7.2 Save `epoch_{epoch:04d}.pt` via `save_checkpoint` when condition is met
  - [x] 7.3 Log a `"periodic_checkpoint"` JSONL entry with `epoch`, `step`, and `path`

- [x] 8. Wire early stopping in `backprop/train.py`
  - [x] 8.1 Add `patience_counter` variable initialized to `0` before the epoch loop
  - [x] 8.2 After each validation, increment counter if improvement < `early_stopping_delta`; reset to 0 on improvement
  - [x] 8.3 When `patience_counter >= early_stopping_patience > 0`, log `"early_stop"` JSONL entry and `logger.info` message, then `break`

- [x] 9. Wire activation-stats logging in `backprop/train.py`
  - [x] 9.1 Inside the optimizer step block, when `global_step % log_every_n_steps == 0`, call `activation_stats(model, x)` under `torch.no_grad()`
  - [x] 9.2 Log a `"activation_stats"` JSONL entry with `epoch`, `step`, and `layers` dict
  - [x] 9.3 Maintain a rolling buffer of the last 3 activation-stats entries; emit `logger.warning` when all layers in all 3 entries have `dead_fraction == 1.0`

- [x] 10. Log R² in `val_epoch` entries in `backprop/train.py`
  - [x] 10.1 Replace the inline `evaluate()` call in the training loop with the updated `evaluate()` from `backprop/evaluate.py` (or equivalent inline R² computation)
  - [x] 10.2 Add `"val_r2"` key to the `val_epoch` JSONL log entry

- [x] 11. Write `backprop/tests/test_evaluate.py`
  - [x] 11.1 Example test: `evaluate()` returns `EvalResult` with finite `rmse`, `mae`, `r2`
  - [x] 11.2 Example test: `r2 == 0.0` when all targets are identical (`SS_tot == 0`)
  - [x] 11.3 Example test: `load_model_from_checkpoint` raises `RuntimeError` on non-existent path
  - [x] 11.4 Property test (`@given`): for randomly generated predictions and targets, `rmse >= 0`, `mae >= 0`, `r2 <= 1.0` — tag `Feature: backprop-improvements, Property 2`
  - [x] 11.5 Property test (`@given`): checkpoint round-trip — save MLP weights, load via `load_model_from_checkpoint`, verify tensors are identical — tag `Feature: backprop-improvements, Property 1`

- [x] 12. Write `backprop/tests/test_train.py`
  - [x] 12.1 Smoke test: call `train()` with `max_epochs=2`, `hidden_dims=[8]`, assert no exception and `best.pt` exists on disk

- [x] 13. Write `backprop/tests/test_visualize.py`
  - [x] 13.1 Example test: `plot_loss_curves` with synthetic JSONL data writes a PNG file
  - [x] 13.2 Example test: `plot_init_comparison` with synthetic JSONL data writes a PNG file

- [x] 14. Extend `backprop/tests/test_data.py`
  - [x] 14.1 Property test (`@given` over `batch_size`): every non-last batch from the train DataLoader has first-dimension size equal to `batch_size` — tag `Feature: backprop-improvements, Property 5`
