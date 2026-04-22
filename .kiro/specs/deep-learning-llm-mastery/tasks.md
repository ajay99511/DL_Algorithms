# Implementation Plan: Deep Learning & LLM Mastery Curriculum

## Overview

Six self-contained Python projects built as a monorepo, progressing from MLP fundamentals through transformer pre-training, alignment (SFT + RLHF), vision transformers, reasoning/inference, and professional benchmark evaluation. Each task maps to specific files in the design's module structure and references the requirements that drive it.

---

## Tasks

- [x] 1. Repository scaffold and shared utilities
  - [x] 1.1 Initialize monorepo structure
    - Create top-level `pyproject.toml` with `[build-system]` (setuptools ≥ 68) and `[project]` metadata; list all six packages in `[tool.setuptools.packages.find]`
    - Create `requirements.txt` with all pinned versions from the design: torch==2.3.1, torchvision==0.18.1, transformers==4.41.2, datasets==2.19.2, tokenizers==0.19.1, scikit-learn==1.5.0, numpy==1.26.4, matplotlib==3.9.0, tqdm==4.66.4, pyyaml==6.0.1, lm-eval==0.4.3, accelerate==0.30.1, evaluate==0.4.2, hypothesis==6.103.1, pytest==8.2.2
    - Create `Makefile` with `setup`, `test`, `train-all`, and `clean` targets exactly as specified in the design
    - Create `.gitignore` excluding `outputs/`, `data/`, `*.pt`, `*.bin`, `*.jsonl`, `__pycache__/`, `*.egg-info/`, `.env`, `wandb/`
    - Create `outputs/` directory stubs for all six projects (project1–project6)
    - _Requirements: 7.1, 7.5, 7.9, 7.12_

  - [x] 1.2 Implement `shared/config.py`
    - Define `BaseConfig` dataclass with fields: `seed`, `output_dir`, `log_every_n_steps`, `checkpoint_every_n_epochs`
    - Implement `load_config(path, config_cls)` — reads YAML, instantiates the given dataclass with its values
    - Implement `save_config(config, path)` — serializes dataclass to YAML via `asdict`
    - Add full type annotations on all public functions
    - _Requirements: 7.2, 7.3_

  - [x] 1.3 Implement `shared/logging_utils.py`
    - Implement `JSONLogger` class: `__init__(log_path)` creates/opens the JSONL file; `log(entry)` appends one JSON object per line; `log_config(config)` writes the full config as the first entry with `type: "config"`, timestamp, and library versions
    - _Requirements: 7.3, 1.6_

  - [x] 1.4 Implement `shared/checkpointing.py`
    - Implement `save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_metric, extra)` — saves a `.pt` dict with `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `step`, `best_metric`, `config`
    - Implement `load_checkpoint(path, model, optimizer, scheduler)` — loads the dict, restores all states, returns `{epoch, step, best_metric, extra}`
    - Handle `FileNotFoundError` and `RuntimeError` gracefully: log error with path, return sentinel indicating fresh start
    - _Requirements: 1.5, 2.6, 4.9, 7.3_

  - [x] 1.5 Implement `shared/seed.py`
    - Implement `fix_all_seeds(seed)` — sets `random.seed`, `numpy.random.seed`, `torch.manual_seed`, and `torch.backends.cudnn.deterministic = True`
    - Add CPU-only comment noting CUDA seed would also call `torch.cuda.manual_seed_all`
    - _Requirements: 7.4_

  - [x] 1.6 Implement `shared/lr_schedule.py`
    - Implement `cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)` returning a `LambdaLR` scheduler
    - Linear warmup from 0 to base_lr over `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr`
    - Add reference comment: `# Ref: Loshchilov & Hutter, 2017 — "SGDR: Stochastic Gradient Descent with Warm Restarts"`
    - _Requirements: 1.3, 2.5, 4.10_

  - [x] 1.7 Write shared utility property tests
    - Create `shared/tests/__init__.py` and `shared/tests/test_shared.py`

    - [x] 1.7.1 Write property test for seed reproducibility (Property 16)
      - **Property 16: Seed Reproducibility**
      - Use `@given(seed=st.integers(min_value=0, max_value=2**31))` with `@settings(max_examples=100)`
      - Call `fix_all_seeds(seed)` twice; assert `torch.rand(10)`, `numpy.random.rand(10)`, and `random.random()` sequences are identical across both calls
      - **Validates: Requirements 7.4**

    - [x] 1.7.2 Write property test for config round-trip serialization (Property 17)
      - **Property 17: Config Round-Trip Serialization**
      - Use `@given(seed=st.integers(), output_dir=st.text(min_size=1, max_size=50))` with `@settings(max_examples=100)`
      - Instantiate `BaseConfig`, serialize to a temp YAML file via `save_config`, deserialize via `load_config`, assert all fields are equal
      - **Validates: Requirements 7.2, 7.3**

    - [x] 1.7.3 Write property test for checkpoint round-trip fidelity (Property 3)
      - **Property 3: Checkpoint Round-Trip Fidelity**
      - Use `@given(hidden=st.integers(min_value=4, max_value=64))` with `@settings(max_examples=50)`
      - Build a small `nn.Linear`, save checkpoint to a temp path, load it back, assert `torch.allclose` on all state_dict tensors
      - **Validates: Requirements 1.5, 2.6, 4.9**

    - [x] 1.7.4 Write property test for LR schedule monotonicity (Property 4)
      - **Property 4: LR Schedule Monotonicity**
      - Use `@given(warmup=st.integers(min_value=1, max_value=50), total=st.integers(min_value=51, max_value=500))` with `@settings(max_examples=100)`
      - Collect LR values for all steps; assert non-decreasing during warmup phase and non-increasing during cosine decay phase
      - **Validates: Requirements 1.3, 2.5, 4.10**

  - [x] 1.8 Checkpoint — ensure shared utilities tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [x] 2. Project 1 — MLP Training Loop Fundamentals
  - [x] 2.1 Implement `project1_mlp/config.py`
    - Define `MLPConfig(BaseConfig)` dataclass with all fields from the design: `test_size`, `val_size`, `hidden_dims`, `dropout`, `init_strategy`, `batch_size`, `grad_accum_steps`, `max_epochs`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `warmup_epochs`, `checkpoint_dir`, `log_path`, `plot_dir`
    - Create `project1_mlp/config.yaml` with default values matching the design's hyperparameter table
    - _Requirements: 1.9, 7.2_

  - [x] 2.2 Implement `project1_mlp/data.py`
    - Implement `load_california_housing(val_size, test_size, seed)` returning `(train_loader, val_loader, test_loader)`
    - Load via `sklearn.datasets.fetch_california_housing`; split 80/10/10 with `train_test_split` and fixed seed
    - Fit `StandardScaler` on train split only; transform val and test with the fitted scaler (no data leakage)
    - Wrap in `TensorDataset` + `DataLoader`; shuffle=True for train only
    - Add full type annotations
    - _Requirements: 1.2, 1.10_

  - [x] 2.3 Implement `project1_mlp/model.py`
    - Implement `MLP(nn.Module)` with `__init__(input_dim, hidden_dims, dropout)` and `forward(x) -> Tensor`
    - Architecture: Linear → ReLU → Dropout per hidden layer, final Linear(hidden[-1], 1)
    - Implement `initialize_weights(model, strategy)` supporting `"normal"`, `"xavier"`, `"kaiming"` strategies
    - Implement `activation_stats(model, x) -> dict[str, dict[str, float]]` using forward hooks to capture per-layer `{mean, std, dead_fraction}`
    - Add reference comments for He et al. 2015 (Kaiming) and Glorot & Bengio 2010 (Xavier)
    - _Requirements: 1.1, 1.7, 1.8, 1.10_

  - [x] 2.4 Implement `project1_mlp/train.py`
    - Implement `train(config: MLPConfig)` with the full loop: `fix_all_seeds`, `JSONLogger`, data loading, model init, `initialize_weights`, AdamW optimizer, `cosine_with_warmup` scheduler
    - Gradient accumulation loop: accumulate over `grad_accum_steps` micro-batches, then `clip_grad_norm_`, `optimizer.step()`, `scheduler.step()`
    - Log `train_step` entries (loss, lr, grad_norm) and `val_epoch` entries (val_rmse, val_mae) to JSONL
    - Save best checkpoint when `val_rmse` improves; support `--resume` flag to load from checkpoint
    - Handle NaN loss: log warning with step/epoch, skip optimizer step, continue
    - Add CPU-only comment with BF16 note at the forward pass
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 7.4, 7.6, 7.8_

  - [x] 2.5 Implement `project1_mlp/evaluate.py`
    - Implement `evaluate(model, test_loader) -> tuple[float, float]` returning `(rmse, mae)`
    - Print a formatted results table to stdout
    - _Requirements: 1.2_

  - [x] 2.6 Implement `project1_mlp/visualize.py`
    - Implement `plot_loss_curves(log_path, output_path)` — reads JSONL, plots train/val loss vs epoch, saves PNG
    - Implement `plot_init_comparison(logs: list[str], labels: list[str], output_path)` — overlays val loss curves for all three init strategies on one figure
    - _Requirements: 1.7_

  - [x] 2.7 Write Project 1 tests
    - Create `project1_mlp/tests/__init__.py`, `test_data.py`, `test_model.py`

    - [x] 2.7.1 Write property test for data split disjointness (Property 1)
      - **Property 1: Data Split Disjointness**
      - Use `@given(n=st.integers(min_value=10, max_value=10_000), val_frac=st.floats(0.05, 0.2), test_frac=st.floats(0.05, 0.2))` with `@settings(max_examples=100)`
      - Assert train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅, and |train| + |val| + |test| = n
      - **Validates: Requirements 1.2**

    - [x] 2.7.2 Write property test for MLP forward pass shape invariant (Property 2)
      - **Property 2: MLP Forward Pass Shape Invariant**
      - Use `@given(batch_size=st.integers(1, 64), hidden_dims=st.lists(st.integers(4, 128), min_size=1, max_size=4))` with `@settings(max_examples=100)`
      - Assert output shape is `(batch_size, 1)` for any valid input
      - **Validates: Requirements 1.1, 1.10**

    - [x] 2.7.3 Write property test for activation statistics validity (Property 18)
      - **Property 18: Activation Statistics Validity**
      - Use `@given(batch_size=st.integers(1, 32), hidden_dims=st.lists(st.integers(4, 64), min_size=1, max_size=3))` with `@settings(max_examples=100)`
      - Assert all returned `mean` and `std` values are finite (not NaN, not Inf), and all `dead_fraction` values are in `[0.0, 1.0]`
      - **Validates: Requirements 1.8**

    - [x] 2.7.4 Write example test for no data leakage between splits
      - Assert that the indices used in train, val, and test DataLoaders are pairwise disjoint
      - Assert that the StandardScaler is fit only on train data (verify by checking scaler mean matches train feature mean)
      - _Requirements: 1.2_

    - [x] 2.7.5 Write example test for init strategies producing different weights
      - Instantiate the same MLP twice, apply `"xavier"` to one and `"kaiming"` to the other, assert weight tensors differ
      - _Requirements: 1.7_

  - [x] 2.8 Write `project1_mlp/README.md`
    - Include: motivation, California Housing dataset description with citation (Harrison & Rubinfeld, 1978), methodology section, hyperparameter table (matching design), results table (RMSE, MAE), training curve plot reference, init strategy comparison plot reference
    - Cite at minimum: AdamW (Loshchilov & Hutter, 2019), Kaiming init (He et al., 2015), Xavier init (Glorot & Bengio, 2010)
    - _Requirements: 1.11_

  - [x] 2.9 Checkpoint — ensure Project 1 tests pass end-to-end
    - Ensure all tests pass, ask the user if questions arise.


- [x] 3. Project 2 — Transformer Architecture and Language Model Pre-training
  - [x] 3.1 Implement `project2_transformer/config.py`
    - Define `TransformerConfig(BaseConfig)` with all fields from the design: `dataset_name`, `max_stories`, `val_fraction`, `context_length`, `vocab_size`, `n_layers`, `n_heads`, `d_model`, `d_ff`, `dropout`, `batch_size`, `grad_accum_steps`, `max_steps`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `warmup_steps`, `tokenizer_dir`, `checkpoint_dir`, `log_path`
    - Create `project2_transformer/config.yaml` with default values
    - _Requirements: 2.11, 7.2_

  - [x] 3.2 Implement `project2_transformer/tokenizer.py`
    - Implement `BPETokenizer` wrapping `tokenizers.ByteLevelBPETokenizer`
    - `train(texts, vocab_size, save_dir)` — trains BPE on provided texts, saves vocab and merges to `save_dir`
    - `encode(text) -> list[int]` and `decode(ids) -> str`
    - `save(save_dir)` and `load(save_dir)` class method
    - _Requirements: 2.3, 2.12_

  - [x] 3.3 Implement `project2_transformer/data.py`
    - Stream `roneneldan/TinyStories` via `datasets.load_dataset(..., streaming=True)`, take first `max_stories`
    - Tokenize all stories with `BPETokenizer`, concatenate IDs, chunk into `context_length=256` windows
    - 95/5 train/val split with fixed seed
    - Return `(train_loader, val_loader)` with `DataLoader(shuffle=True)` for train
    - _Requirements: 2.2, 2.12_

  - [x] 3.4 Implement `project2_transformer/model.py`
    - Implement `CausalSelfAttention(nn.Module)` with pre-norm, multi-head attention, causal mask registered as buffer; add reference comments for Vaswani et al. 2017 and Radford et al. 2019
    - Implement `TransformerBlock(nn.Module)` composing attention + FFN with GELU activation and pre-LayerNorm
    - Implement `GPTModel(nn.Module)` with token embeddings, learned positional encodings, N transformer blocks, final LayerNorm, LM head with weight tying to embedding; `forward(input_ids, targets) -> (logits, loss)`; `count_parameters() -> int`
    - Add CPU-only comment with BF16 note
    - _Requirements: 2.1, 2.10, 2.12, 7.6_

  - [x] 3.5 Implement `project2_transformer/train.py`
    - Implement `train(config: TransformerConfig)` with full pre-training loop: `fix_all_seeds`, `JSONLogger`, tokenizer training/loading, data loading, model init, AdamW (b1=0.9, b2=0.95), `cosine_with_warmup`
    - Gradient accumulation over `grad_accum_steps=16` micro-batches; `clip_grad_norm_`; log `train_step` and `val_epoch` entries
    - Save checkpoint when val perplexity improves; support resume from checkpoint
    - Display `tqdm` progress bar with live loss and perplexity
    - _Requirements: 2.4, 2.5, 2.6, 7.4, 7.8_

  - [x] 3.6 Implement `project2_transformer/evaluate.py`
    - Implement `compute_perplexity(model, val_loader) -> float` — `exp(mean cross-entropy)` over validation set
    - _Requirements: 2.4, 2.14_

  - [x] 3.7 Implement `project2_transformer/generate.py`
    - Implement `greedy_decode(model, input_ids, max_new_tokens) -> Tensor`
    - Implement `nucleus_sample(model, input_ids, max_new_tokens, top_p, temperature, seed) -> Tensor`
    - Both functions must be deterministic under fixed seed
    - _Requirements: 2.8_

  - [x] 3.8 Implement `project2_transformer/visualize.py`
    - Implement hook-based `plot_attention_heatmaps(model, input_ids, tokenizer, save_dir)` — captures attention weights per head per layer, saves heatmap PNGs
    - Implement `plot_weight_distributions(model, save_dir)` — histograms and spectral norms per layer
    - _Requirements: 2.7, 2.9_

  - [x] 3.9 Write Project 2 tests
    - Create `project2_transformer/tests/__init__.py`, `test_tokenizer.py`, `test_model.py`, `test_data.py`

    - [x] 3.9.1 Write property test for transformer output shape invariant (Property 5)
      - **Property 5: Transformer Output Shape Invariant**
      - Use `@given(batch_size=st.integers(1, 8), seq_len=st.integers(1, 256))` with `@settings(max_examples=100)`
      - Assert `logits.shape == (batch_size, seq_len, vocab_size)` for any valid input
      - **Validates: Requirements 2.1, 2.10**

    - [x] 3.9.2 Write property test for tokenizer round-trip (Property 6)
      - **Property 6: Tokenizer Round-Trip**
      - Use `@given(text=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("Lu","Ll","Nd","Zs"))))` with `@settings(max_examples=100)`
      - Assert `tokenizer.decode(tokenizer.encode(text)) == text`
      - **Validates: Requirements 2.3, 2.12**

    - [x] 3.9.3 Write property test for causal attention mask correctness (Property 7)
      - **Property 7: Causal Attention Mask Correctness**
      - Use `@given(seq_len=st.integers(2, 64))` with `@settings(max_examples=100)`
      - Run a forward pass, extract attention weights; assert the upper triangle (above diagonal) is zero after softmax for all heads and layers
      - **Validates: Requirements 2.12**

    - [x] 3.9.4 Write property test for inference reproducibility under fixed seed (Property 8)
      - **Property 8: Inference Reproducibility Under Fixed Seed**
      - Use `@given(seed=st.integers(0, 2**31), prompt_len=st.integers(1, 32))` with `@settings(max_examples=50)`
      - Run `nucleus_sample` twice with the same seed; assert output token sequences are byte-identical
      - **Validates: Requirements 2.8, 5.8**

    - [x] 3.9.5 Write property test for config serialization (Property 17 — TransformerConfig)
      - **Property 17: Config Round-Trip Serialization (TransformerConfig)**
      - Use `@given(n_layers=st.integers(1, 8), d_model=st.integers(32, 256))` with `@settings(max_examples=100)`
      - Serialize `TransformerConfig` to YAML and deserialize; assert all fields equal
      - **Validates: Requirements 7.2, 7.3**

    - [x] 3.9.6 Write example test for data loader output shapes
      - Assert `input_ids.shape == (batch_size, context_length)` for a batch from the train loader
      - _Requirements: 2.12_

  - [x] 3.10 Write `project2_transformer/README.md`
    - Include: motivation, TinyStories dataset description with citation (Eldan & Li, 2023), architecture table (layers, heads, d_model, parameters), training curve plots reference, sample generated stories section, citations for Vaswani et al. 2017 and Radford et al. 2019
    - _Requirements: 2.13_

  - [x] 3.11 Checkpoint — ensure Project 2 tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [x] 4. Project 3 — Supervised Fine-Tuning, Instruction Tuning, and RLHF
  - [x] 4.1 Implement `project3_alignment/config.py`
    - Define `AlignmentConfig(BaseConfig)` with all fields from the design: `base_model_name`, SFT fields (`sft_dataset`, `sft_val_fraction`, `sft_max_epochs`, `sft_lr`, `sft_batch_size`, `sft_grad_accum_steps`), reward model fields (`rm_dataset`, `rm_max_epochs`, `rm_lr`), PPO fields (`ppo_steps`, `ppo_lr`, `kl_coeff`, `reward_clip_bound`), RLAIF fields (`rlaif_model`), and path fields
    - Create `project3_alignment/config.yaml` with default values
    - _Requirements: 3.11, 7.2_

  - [x] 4.2 Implement `project3_alignment/data.py`
    - Implement Alpaca loader: `load_alpaca(config) -> (train_loader, val_loader)` — loads `tatsu-lab/alpaca`, formats with Alpaca prompt template (`"### Instruction:\n{instruction}\n\n### Response:\n{output}"`), tokenizes with GPT-2 tokenizer, truncates to 512 tokens, 90/10 split with fixed seed
    - Implement HH-RLHF loader: `load_hh_rlhf(config) -> (chosen_loader, rejected_loader, val_loader)` — loads `Anthropic/hh-rlhf`, tokenizes chosen/rejected pairs, 90/10 split with fixed seed
    - _Requirements: 3.1, 3.3, 3.4, 3.12_

  - [x] 4.3 Implement `project3_alignment/sft.py`
    - Implement `run_sft(config: AlignmentConfig)` — loads GPT-2 small from HF (or project2 checkpoint), fine-tunes on Alpaca with AdamW + `cosine_with_warmup`, logs per-layer gradient magnitudes at configurable intervals, saves checkpoint at each epoch
    - _Requirements: 3.1, 3.2, 3.10_

  - [x] 4.4 Implement `project3_alignment/reward_model.py`
    - Implement `RewardModel(nn.Module)` wrapping a transformer backbone with a scalar linear head; `forward(input_ids, attention_mask) -> Tensor` of shape `(B,)`
    - Implement `train_reward_model(model, chosen_loader, rejected_loader, config)` using Bradley-Terry loss: `log sigmoid(r_chosen - r_rejected)`
    - Add reference comments for Ouyang et al. 2022 and Bai et al. 2022
    - _Requirements: 3.4, 3.12_

  - [x] 4.5 Implement `project3_alignment/rlhf.py`
    - Implement `ppo_step(policy, ref_policy, reward_model, batch, config, logger) -> dict[str, float]` following the design's PPO loop: generate responses, score with reward model, clip rewards to `±reward_clip_bound` (log warning if clipped), compute KL(policy ∥ ref_policy), compute adjusted reward, policy gradient update
    - Implement `run_rlhf(config)` orchestrating the full PPO training loop for `ppo_steps` steps, logging reward, KL, policy_loss, value_loss
    - _Requirements: 3.5, 3.6, 3.9_

  - [x] 4.6 Implement `project3_alignment/rlaif.py`
    - Implement `run_rlaif(config)` — loads FLAN-T5-small, uses it to generate preference scores in place of the human reward model, logs reward distribution comparison between RLHF and RLAIF configurations
    - _Requirements: 3.7_

  - [x] 4.7 Implement `project3_alignment/evaluate.py`
    - Implement `log_gradient_magnitudes(model, logger)` — computes and logs per-layer gradient L2 norms
    - Implement `evaluate_reward_accuracy(reward_model, val_loader) -> float` — fraction of chosen > rejected pairs correctly ranked
    - _Requirements: 3.2, 3.4_

  - [x] 4.8 Implement `project3_alignment/compare.py`
    - Implement `run_comparison(config)` — generates outputs on 20 fixed evaluation prompts at each stage (base, SFT, RLHF, RLAIF), saves structured JSON to `comparison_file`
    - _Requirements: 3.8_

  - [x] 4.9 Write Project 3 tests
    - Create `project3_alignment/tests/__init__.py`, `test_reward_model.py`, `test_rlhf.py`, `test_data.py`

    - [x] 4.9.1 Write property test for reward model output shape invariant (Property 13)
      - **Property 13: Reward Model Output Shape Invariant**
      - Use `@given(batch_size=st.integers(1, 16), seq_len=st.integers(1, 64))` with `@settings(max_examples=100)`
      - Assert `reward_model(input_ids, attention_mask).shape == (batch_size,)`
      - **Validates: Requirements 3.4, 3.12**

    - [x] 4.9.2 Write property test for KL divergence non-negativity (Property 11)
      - **Property 11: KL Divergence Non-Negativity**
      - Use `@given(vocab_size=st.integers(2, 100), batch_size=st.integers(1, 8))` with `@settings(max_examples=100)`
      - Generate random valid probability distributions p and q; assert `KL(p || q) >= 0` for all pairs
      - **Validates: Requirements 3.5**

    - [x] 4.9.3 Write property test for reward clipping invariant (Property 10)
      - **Property 10: Reward Clipping Invariant**
      - Use `@given(reward=st.floats(-1e6, 1e6, allow_nan=False), bound=st.floats(0.01, 100.0, allow_nan=False))` with `@settings(max_examples=100)`
      - Assert `-bound <= clip(reward, -bound, bound) <= bound`
      - **Validates: Requirements 3.9**

    - [x] 4.9.4 Write property test for data split disjointness in alignment data (Property 1)
      - **Property 1: Data Split Disjointness (Alignment)**
      - Use `@given(n=st.integers(min_value=20, max_value=1000), val_frac=st.floats(0.05, 0.3))` with `@settings(max_examples=100)`
      - Assert train and val index sets are disjoint and their union equals the full dataset
      - **Validates: Requirements 3.1, 3.12**

    - [x] 4.9.5 Write example test for Alpaca prompt format
      - Assert that formatted prompts contain `"### Instruction:"` and `"### Response:"` markers
      - _Requirements: 3.3_

  - [x] 4.10 Write `project3_alignment/README.md`
    - Include: motivation, dataset descriptions with citations (Taori et al. 2023 for Alpaca; Bai et al. 2022 for HH-RLHF; Ouyang et al. 2022 for InstructGPT/RLHF), methodology, results table comparing outputs across stages, discussion of alignment tax observations
    - _Requirements: 3.13_

  - [x] 4.11 Checkpoint — ensure Project 3 tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [x] 5. Project 4 — Vision Transformer and Image Classification
  - [x] 5.1 Implement `project4_vit/config.py`
    - Define `ViTConfig(BaseConfig)` with all fields from the design: `dataset`, `image_size`, `patch_size`, `n_channels`, `n_classes`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `dropout`, `batch_size`, `max_epochs`, `learning_rate`, `weight_decay`, `warmup_epochs`, `grad_clip_norm`, `checkpoint_dir`, `log_path`, `plot_dir`
    - Create `project4_vit/config.yaml` with default values
    - _Requirements: 4.11, 7.2_

  - [x] 5.2 Implement `project4_vit/data.py`
    - Implement `load_cifar10(config) -> (train_loader, val_loader, test_loader)` — loads via `torchvision.datasets.CIFAR10`, applies train augmentation (RandomHorizontalFlip, RandomCrop(32, padding=4), Normalize), val/test gets Normalize only; 10% val split from train with fixed seed
    - Implement `load_imagenette(config) -> (train_loader, val_loader, test_loader)` — loads via `datasets.load_dataset("frgfm/imagenette")`, resizes to 224×224, same augmentation pattern
    - Select dataset via `config.dataset` flag
    - _Requirements: 4.2, 4.3, 4.4, 4.12_

  - [x] 5.3 Implement `project4_vit/model.py`
    - Implement `PatchEmbedding(nn.Module)` with `__init__(image_size, patch_size, n_channels, d_model)` and `forward(x: Tensor) -> Tensor` mapping `(B, C, H, W) -> (B, n_patches, d_model)` via learned linear projection
    - Implement `ViT(nn.Module)` with class token, learned 1D positional embeddings, N transformer encoder blocks with pre-norm, classification MLP head; `forward(x) -> (B, n_classes)` logits; `get_attention_weights(x) -> list[Tensor]`
    - Add reference comment for Dosovitskiy et al. 2020
    - _Requirements: 4.1, 4.12_

  - [x] 5.4 Implement `project4_vit/baseline.py`
    - Implement a small ResNet-18 variant adapted for 32×32 CIFAR-10: 4 residual blocks [64, 128, 256, 512] channels, global average pooling, linear classifier
    - _Requirements: 4.8_

  - [x] 5.5 Implement `project4_vit/train.py`
    - Implement `train(model, config: ViTConfig)` — shared training loop for both ViT and CNN; AdamW + `cosine_with_warmup`; logs train loss, val accuracy (top-1), grad norm, LR; saves best checkpoint; supports resume; `tqdm` progress bar
    - Log patch size comparison metrics (parameter count, steps/sec, val accuracy) when running both patch=4 and patch=8 configs
    - _Requirements: 4.6, 4.9, 4.10, 7.8_

  - [x] 5.6 Implement `project4_vit/evaluate.py`
    - Implement `evaluate_top1(model, loader) -> float` — top-1 accuracy on a DataLoader
    - _Requirements: 4.12_

  - [x] 5.7 Implement `project4_vit/visualize.py`
    - Implement `plot_training_curves(log_path, output_path)` — train loss and val accuracy vs epoch
    - Implement `plot_patch_grid(image, patch_size, output_path)` — visualizes how an image is divided into patches
    - _Requirements: 4.6_

  - [x] 5.8 Implement `project4_vit/attention_viz.py`
    - Implement `attention_rollout(attention_weights, discard_ratio=0.9) -> Tensor` — Abnar & Zuidema (2020) rollout returning `(n_patches,)` relevance scores; add reference comment
    - Implement `overlay_attention_on_image(image, rollout, patch_size, save_path)` — overlays attention heatmap on original image, saves PNG
    - _Requirements: 4.5, 4.7_

  - [x] 5.9 Write Project 4 tests
    - Create `project4_vit/tests/__init__.py`, `test_model.py`, `test_data.py`

    - [x] 5.9.1 Write property test for patch embedding shape invariant (Property 12)
      - **Property 12: Patch Embedding Shape Invariant**
      - Use `@given(batch_size=st.integers(1, 16), patch_size=st.sampled_from([4, 8]), image_size=st.sampled_from([32, 64]))` with `@settings(max_examples=100)`
      - Assert output shape is `(batch_size, (image_size//patch_size)**2, d_model)` for any valid input
      - **Validates: Requirements 4.1, 4.12**

    - [x] 5.9.2 Write property test for ViT output shape invariant (Property 5 — adapted)
      - **Property 5 (adapted): ViT Output Shape Invariant**
      - Use `@given(batch_size=st.integers(1, 8))` with `@settings(max_examples=50)`
      - Assert `vit(x).shape == (batch_size, n_classes)` for any valid batch of images
      - **Validates: Requirements 4.1, 4.12**

    - [x] 5.9.3 Write property test for checkpoint round-trip fidelity (Property 3 — ViT)
      - **Property 3: Checkpoint Round-Trip Fidelity (ViT)**
      - Use `@given(patch_size=st.sampled_from([4, 8]))` with `@settings(max_examples=20)`
      - Save and reload a ViT checkpoint; assert all state_dict tensors are equal under `torch.allclose`
      - **Validates: Requirements 4.9**

    - [x] 5.9.4 Write property test for normalization range
      - Use `@given(batch_size=st.integers(1, 16))` with `@settings(max_examples=100)`
      - Assert that normalized CIFAR-10 images have per-channel mean ≈ 0 and std ≈ 1 (within tolerance)
      - _Requirements: 4.4_

    - [x] 5.9.5 Write example test for data loader output shapes and value ranges
      - Assert `images.shape == (batch_size, 3, 32, 32)` and `labels.shape == (batch_size,)` for a CIFAR-10 batch
      - _Requirements: 4.12_

  - [x] 5.10 Write `project4_vit/README.md`
    - Include: motivation, CIFAR-10 dataset description, architecture table (patch=4 and patch=8 configs), training curve plots reference, attention visualization examples, comparison table (ViT vs CNN: accuracy, parameters, training time), citations for Dosovitskiy et al. 2020, He et al. 2016, Abnar & Zuidema 2020
    - _Requirements: 4.13_

  - [x] 5.11 Checkpoint — ensure Project 4 tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [x] 6. Project 5 — Reasoning, Inference Strategies, and KV Cache
  - [x] 6.1 Implement `project5_reasoning/config.py`
    - Define `ReasoningConfig(BaseConfig)` with all fields from the design: `model_name`, `max_new_tokens`, `beam_width`, `top_k`, `top_p`, `temperature`, `gsm8k_subset_size`, `bigbench_subset_size`, `benchmark_file`, `log_path`
    - Create `project5_reasoning/config.yaml` with default values
    - _Requirements: 5.10, 7.2_

  - [x] 6.2 Implement `project5_reasoning/data.py`
    - Implement `load_gsm8k(subset_size) -> list[dict]` — loads `gsm8k` test split via HF Datasets, returns first `subset_size` problems
    - Implement `load_bigbench_hard(subset_size) -> list[dict]` — loads `maveriq/bigbenchhard` via HF Datasets, returns first `subset_size` problems
    - _Requirements: 5.2, 5.6_

  - [x] 6.3 Implement `project5_reasoning/inference.py`
    - Implement `greedy_decode(model, input_ids, max_new_tokens, tokenizer) -> (Tensor, list[Tensor])` returning output IDs and per-step log probs
    - Implement `beam_search(model, input_ids, max_new_tokens, beam_width, tokenizer) -> (Tensor, list[dict])` returning best output IDs and beam log with candidates and cumulative log-probs per step
    - Implement `top_k_sample(model, input_ids, max_new_tokens, k, temperature, seed) -> Tensor`
    - Implement `nucleus_sample(model, input_ids, max_new_tokens, top_p, temperature, seed) -> Tensor`
    - All sampling functions must call `fix_all_seeds(seed)` when seed is provided
    - Add full type annotations
    - _Requirements: 5.1, 5.3, 5.7, 5.8_

  - [x] 6.4 Implement `project5_reasoning/kv_cache.py`
    - Implement `KVCache` class: `__init__(n_layers, n_heads, d_head, max_seq_len)`, `update(layer_idx, k, v) -> (full_k, full_v)`, `clear()`
    - Implement `benchmark_kv_cache(model, prompts, tokenizer, config) -> dict[str, float]` returning `{tokens_per_sec_cached, tokens_per_sec_uncached, speedup_ratio}`
    - Add reference comment for Pope et al. 2022
    - _Requirements: 5.4_

  - [x] 6.5 Implement `project5_reasoning/reasoning.py`
    - Implement `zero_shot_prompt(question) -> str`
    - Implement `few_shot_prompt(question, examples) -> str`
    - Implement `chain_of_thought_prompt(question, examples) -> str` with reference comment for Wei et al. 2022
    - Implement `scratchpad_generate(model, tokenizer, question, config) -> (scratchpad_steps, final_answer)`
    - _Requirements: 5.5, 5.6_

  - [x] 6.6 Implement `project5_reasoning/evaluate.py`
    - Implement `exact_match_accuracy(predictions, references) -> float`
    - Implement `distinct_n(texts, n) -> float` — output diversity metric
    - _Requirements: 5.2, 5.9_

  - [x] 6.7 Implement `project5_reasoning/benchmark.py`
    - Implement `run_benchmark(config) -> list[dict]` — runs all inference strategies on the GSM8K subset, collects exact-match accuracy, distinct-1, distinct-2, tokens/sec, avg response length per strategy
    - Saves results to `config.benchmark_file` as JSON matching the schema in the design
    - _Requirements: 5.9_

  - [x] 6.8 Write Project 5 tests
    - Create `project5_reasoning/tests/__init__.py`, `test_inference.py`, `test_kv_cache.py`, `test_reasoning.py`

    - [x] 6.8.1 Write property test for KV cache output equivalence (Property 9)
      - **Property 9: KV Cache Output Equivalence**
      - Use `@given(seq_len=st.integers(2, 32), n_layers=st.integers(1, 4))` with `@settings(max_examples=50)`
      - Assert that autoregressive generation with KV cache produces logits element-wise identical (within `torch.allclose` tolerance) to generation without KV cache
      - **Validates: Requirements 5.4, 5.11**

    - [x] 6.8.2 Write property test for inference reproducibility under fixed seed (Property 8 — Project 5)
      - **Property 8: Inference Reproducibility Under Fixed Seed (all strategies)**
      - Use `@given(seed=st.integers(0, 2**31), strategy=st.sampled_from(["greedy", "beam", "top_k", "nucleus"]))` with `@settings(max_examples=50)`
      - Run each strategy twice with the same seed; assert output token sequences are byte-identical
      - **Validates: Requirements 2.8, 5.8**

    - [x] 6.8.3 Write property test for beam search width invariant
      - Use `@given(beam_width=st.integers(1, 8), seq_len=st.integers(1, 16))` with `@settings(max_examples=50)`
      - Assert that beam search always returns exactly `beam_width` candidates at each step
      - _Requirements: 5.11_

    - [x] 6.8.4 Write example test for greedy decode output shapes
      - Assert output tensor shape is `(1, max_new_tokens)` for a single-sequence greedy decode
      - _Requirements: 5.1_

  - [x] 6.9 Write `project5_reasoning/README.md`
    - Include: motivation, dataset descriptions with citations (Cobbe et al. 2021 for GSM8K; Wei et al. 2022 for CoT; Srivastava et al. 2022 for BIG-Bench), methodology, results table (accuracy, diversity, tokens/sec per strategy), discussion of when each inference strategy is appropriate
    - _Requirements: 5.12_

  - [x] 6.10 Checkpoint — ensure Project 5 tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [x] 7. Project 6 — Professional Benchmark Evaluation and Model Analysis
  - [x] 7.1 Implement `project6_eval/config.py`
    - Define `EvalConfig(BaseConfig)` with all fields from the design: `models`, `tasks` (dict of task_name → num_fewshot), `output_dir`, `log_path`, `report_csv`, `report_md`
    - Create `project6_eval/config.yaml` with default values (GPT-2 and Pythia-160M; ARC 25-shot, HellaSwag 10-shot, MMLU 5-shot, TruthfulQA 0-shot)
    - _Requirements: 6.11, 7.2_

  - [x] 7.2 Implement `project6_eval/evaluate.py`
    - Implement `run_evaluation(model_name, tasks, config, logger) -> dict[str, float]` wrapping `lm_eval.simple_evaluate()`
    - On model load failure (`OSError`, `ValueError`): log `{model_name, error_type, traceback_summary}`, skip model, continue
    - On task failure (any exception): log `{model_name, task_name, traceback_summary}`, mark task as failed with `None`, continue
    - Add inline comments explaining what each benchmark measures (ARC: science reasoning; HellaSwag: commonsense completion; MMLU: world knowledge; TruthfulQA: factual accuracy)
    - _Requirements: 6.1, 6.2, 6.4, 6.10_

  - [x] 7.3 Implement `project6_eval/weight_analysis.py`
    - Implement `compute_weight_norms(model) -> dict[str, float]` — Frobenius norm per layer
    - Implement `compute_singular_values(model) -> dict[str, Tensor]` — SVD via `torch.linalg.svd` per weight matrix
    - Implement `compute_dead_neuron_ratio(model, dataloader, threshold=1e-6) -> dict[str, float]` — fraction of neurons with mean activation below threshold
    - _Requirements: 6.5_

  - [x] 7.4 Implement `project6_eval/activation_analysis.py`
    - Implement `record_activations(model, dataloader, n_batches) -> dict[str, Tensor]` — uses forward hooks to capture activation distributions across layers
    - Implement `plot_activation_distributions(activations, save_dir)` — plots histograms per layer showing how representations evolve through depth
    - _Requirements: 6.6_

  - [x] 7.5 Implement `project6_eval/dataset_explorer.py`
    - Implement `explore_dataset(dataset_name, config_name, n_samples=1000) -> dict[str, Any]` — streams `n_samples` from the dataset, computes `estimated_token_count`, `vocabulary_size`, `avg_sequence_length`, `sample_texts` (first 3 examples); no full download required
    - Wire up for The Pile (`EleutherAI/pile`, streaming), C4 (`allenai/c4`, streaming), and OpenWebText (`Skylion007/openwebtext`, streaming)
    - _Requirements: 6.7_

  - [x] 7.6 Implement `project6_eval/report.py`
    - Implement `generate_csv_report(results, output_path)` — writes CSV with columns: `Model | ARC-Challenge | HellaSwag | MMLU | TruthfulQA | Average` in that exact order; fill missing tasks with empty string and log warning
    - Implement `generate_markdown_report(results, output_path)` — same data as Markdown table
    - _Requirements: 6.3, 6.4_

  - [x] 7.7 Write Project 6 tests
    - Create `project6_eval/tests/__init__.py`, `test_report.py`, `test_error_handling.py`, `test_dataset_explorer.py`

    - [x] 7.7.1 Write property test for evaluation report column completeness (Property 14)
      - **Property 14: Evaluation Report Column Completeness**
      - Use `@given(models=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5))` with `@settings(max_examples=100)`
      - Generate a results dict for arbitrary model names; assert the CSV contains exactly the columns `Model, ARC-Challenge, HellaSwag, MMLU, TruthfulQA, Average` in that order
      - **Validates: Requirements 6.3, 6.12**

    - [x] 7.7.2 Write property test for graceful continuation on task failure (Property 15)
      - **Property 15: Graceful Continuation on Task Failure**
      - Use `@given(n_tasks=st.integers(2, 10), n_failing=st.integers(1, 5))` with `@settings(max_examples=100)`
      - Inject exceptions into a subset of tasks; assert the runner completes all non-failing tasks and returns results for them without re-raising
      - **Validates: Requirements 6.10, 6.12**

    - [x] 7.7.3 Write property test for dataset streaming format
      - Use `@given(n_samples=st.integers(1, 100))` with `@settings(max_examples=20)`
      - Assert that `explore_dataset` returns a dict with keys `estimated_token_count`, `vocabulary_size`, `avg_sequence_length`, `sample_texts` for any valid `n_samples`
      - _Requirements: 6.7_

    - [x] 7.7.4 Write example test for error logging format
      - Simulate a model load failure; assert the log entry contains `model_name`, `error_type`, and `traceback_summary` keys
      - _Requirements: 6.10_

  - [x] 7.8 Write `project6_eval/README.md`
    - Include: motivation, benchmark descriptions with citations (Clark et al. 2018 for ARC; Zellers et al. 2019 for HellaSwag; Hendrycks et al. 2021 for MMLU; Lin et al. 2022 for TruthfulQA), model descriptions with citations (Radford et al. 2019 for GPT-2; Biderman et al. 2023 for Pythia), results tables, weight and activation analysis plots reference, discussion of what results reveal about each model's strengths and weaknesses
    - Include a research paper reference section with one-line summaries for each paper
    - _Requirements: 6.8, 6.13_

  - [x] 7.9 Checkpoint — ensure Project 6 tests pass
    - Ensure all tests pass, ask the user if questions arise.


- [ ] 8. Top-level README and final integration
  - [x] 8.1 Write top-level `README.md`
    - Include: learning path overview (6-project progression with one-line description each), prerequisites (Python 3.11+, pip, ~8GB RAM), hardware requirements (CPU-only AMD Ryzen 9 or equivalent), quick-start instructions (`make setup`, `make test`, `make train-all`), per-project time estimates, and links to each project's README
    - _Requirements: 7.1_

  - [x] 8.2 Wire `shared/` package into all six projects
    - Add `shared/__init__.py` exporting `BaseConfig`, `JSONLogger`, `save_checkpoint`, `load_checkpoint`, `fix_all_seeds`, `cosine_with_warmup`
    - Verify each project's `train.py` (or equivalent entry point) imports from `shared` correctly
    - _Requirements: 7.1, 7.5_

  - [x] 8.3 Final integration checkpoint — run full test suite
    - Run `python -m pytest project1_mlp/tests project2_transformer/tests project3_alignment/tests project4_vit/tests project5_reasoning/tests project6_eval/tests -v` and ensure all tests pass
    - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 1.8, 2.9, 3.11, 4.11, 5.11, 6.10, 7.9, and 8.3 ensure incremental validation
- All 18 correctness properties from the design are covered by property-based tests using Hypothesis (`@settings(max_examples=100)`)
- Property tests are placed close to the implementation tasks they validate to catch errors early
- Unit/example tests cover integration points, edge cases, and error conditions
- All code uses Python 3.11+ type annotations on public functions
- All training scripts use `torch.device("cpu")` explicitly with BF16 comments at the point of use
