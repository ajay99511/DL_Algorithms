# Requirements Document

## Introduction

This spec defines a research-grade, hands-on deep learning and LLM mastery curriculum for an
intermediate-level practitioner working on a CPU-only AMD Ryzen 9 laptop. The goal is to produce
work that mirrors the standards of AI researchers at top labs (DeepMind, OpenAI, Meta AI, Google
Brain, Anthropic) — not tutorial-quality throwaway code. Every project produces a professional
artifact: a trained model with reproducible results, structured experiment logs, a research-quality
README with paper citations, and clean modular Python code. The curriculum spans 6 projects that
build on each other, covering the full arc from training loop fundamentals through transformer
pre-training, alignment (SFT + RLHF), vision transformers, reasoning and inference, and
professional benchmark evaluation. PyTorch is the primary framework throughout. All datasets are
real, publicly available, and used in published research or industry benchmarks.

---

## Glossary

- **Learner**: The practitioner following this curriculum.
- **Project**: A self-contained Python module with its own config, data pipeline, training loop,
  evaluation, and README.
- **Mini-Model**: A small-scale model (1–10M parameters) designed to train on CPU within a
  reasonable wall-clock time.
- **Training Loop**: The full cycle of forward pass → loss computation → backward pass →
  optimizer step, with LR scheduling and gradient accumulation.
- **Pre-training**: Training a model from randomly initialized weights on a large unsupervised
  corpus using a self-supervised objective (e.g., next-token prediction).
- **SFT**: Supervised Fine-Tuning — adapting a pre-trained model to follow instructions using
  labeled (prompt, response) pairs.
- **RLHF**: Reinforcement Learning from Human Feedback — aligning model outputs using a reward
  model trained on human preference pairs.
- **RLAIF**: Reinforcement Learning from AI Feedback — using an AI-generated preference signal
  in place of human labels.
- **Reasoning Model**: A model trained or prompted to produce explicit chain-of-thought steps
  before emitting a final answer.
- **KV Cache**: Key-Value cache used during autoregressive inference to avoid recomputing
  attention over previously seen tokens.
- **Interpretability**: The practice of understanding what a model has learned by inspecting
  weights, activations, and attention patterns.
- **Benchmark**: A standardized evaluation suite used to compare model capabilities across
  tasks (e.g., ARC, HellaSwag, MMLU, TruthfulQA).
- **ViT**: Vision Transformer — a transformer architecture applied to fixed-size image patches.
- **Tokenizer**: A component that converts raw text into integer token IDs consumable by a
  language model.
- **Gradient Accumulation**: Accumulating gradients over multiple micro-batches before
  performing an optimizer step, simulating a larger effective batch size.
- **LR Schedule**: A policy for adjusting the learning rate during training (e.g., cosine
  decay with linear warmup).
- **Experiment Log**: A structured record (JSON, CSV, or W&B run) of hyperparameters, metrics,
  and artifacts for a single training run.
- **Checkpoint**: A saved snapshot of model weights and optimizer state that enables training
  to be resumed from a specific step.
- **Config**: A YAML file or Python dataclass that fully specifies all hyperparameters for a
  training run, enabling exact reproduction.
- **Hugging Face**: A platform hosting pre-trained models, datasets, and evaluation tooling
  widely used in industry and academia.
- **lm-evaluation-harness**: EleutherAI's open-source framework for evaluating language models
  on standardized benchmarks, used by the Hugging Face Open LLM Leaderboard.
- **ImageNette**: A 10-class subset of ImageNet curated by fast.ai, used as a real vision
  benchmark that is tractable on consumer hardware.
- **GSM8K**: Grade School Math 8K — a dataset of 8,500 grade-school math word problems used
  to evaluate multi-step arithmetic reasoning in language models.
- **TinyStories**: A synthetic dataset of short children's stories (Eldan & Li, 2023) designed
  to train small language models that produce coherent English text.
- **Alpaca**: Stanford's instruction-following dataset of 52K (prompt, response) pairs
  generated using self-instruct from GPT-3.5, used to fine-tune LLaMA into an instruction
  follower.
- **HH-RLHF**: Anthropic's Helpful and Harmless RLHF dataset of human preference pairs used
  to train reward models for alignment research.
- **Optimizer**: An algorithm (e.g., AdamW) that updates model weights using gradients and
  momentum estimates.
- **Mixed Precision**: Training with FP16 or BF16 activations alongside FP32 master weights
  to reduce memory and increase throughput (documented for awareness even on CPU).

---

## Requirements

---

### Requirement 1: Project 1 — Deep Learning Foundations and the Training Loop

**User Story:** As a Learner, I want to build and train a small neural network on a real,
domain-meaningful dataset using research-grade practices, so that I internalize every component
of the training loop — including LR scheduling, checkpointing, and experiment logging — before
touching transformers.

#### Acceptance Criteria

1. THE Project_1 SHALL implement a multi-layer perceptron (MLP) in pure PyTorch (no high-level
   wrappers) covering: weight initialization, forward pass, loss computation, backward pass,
   gradient clipping, and AdamW optimizer step.

2. THE Project_1 SHALL train on the California Housing dataset (scikit-learn built-in, no manual
   download) for regression, using a proper 80/10/10 train/validation/test split with a fixed
   random seed, and SHALL report RMSE and MAE on the held-out test set.

3. THE Project_1 SHALL implement cosine annealing with linear warmup as the learning rate
   schedule and SHALL log the LR value at every step alongside training loss.

4. THE Project_1 SHALL implement gradient accumulation over a configurable number of
   micro-batches and SHALL demonstrate that effective batch size scales proportionally.

5. WHEN a training run completes, THE Project_1 SHALL save a checkpoint containing model
   weights, optimizer state, scheduler state, current epoch, and best validation RMSE, and
   SHALL support resuming training from any saved checkpoint.

6. THE Project_1 SHALL log all hyperparameters and per-epoch metrics (train loss, val loss,
   val RMSE, val MAE, gradient norm, LR) to a structured JSON experiment log file, with one
   entry per epoch.

7. THE Project_1 SHALL demonstrate three weight initialization strategies (random normal,
   Xavier/Glorot, Kaiming/He) and SHALL produce a saved plot comparing validation loss curves
   across all three strategies on the same dataset split.

8. WHEN the Learner inspects intermediate layer outputs, THE Project_1 SHALL provide a utility
   that prints activation statistics (mean, std, fraction of dead neurons) for each layer
   during a forward pass, enabling diagnosis of vanishing/exploding activations.

9. THE Project_1 SHALL be organized as a Python module with the following separation of
   concerns: `model.py` (MLP definition), `data.py` (dataset loading and splitting),
   `train.py` (training loop), `evaluate.py` (test-set evaluation), `config.py` (dataclass
   config), and `visualize.py` (plot generation).

10. THE Project_1 SHALL include type annotations on all public functions and SHALL include
    unit tests for the data pipeline (correct split sizes, no data leakage between splits)
    and the MLP forward pass (output shape correctness).

11. THE Project_1 SHALL include a research-quality README with: motivation, dataset description
    with citation (Harrison & Rubinfeld, 1978), methodology, hyperparameter table, results
    table, and at least two paper citations relevant to the techniques used (e.g., AdamW,
    Kaiming initialization).

12. WHEN a training run completes, THE Project_1 SHALL complete end-to-end in under 5 minutes
    on a CPU-only AMD Ryzen 9 machine.

---

### Requirement 2: Project 2 — Transformer Architecture and Language Model Pre-training

**User Story:** As a Learner, I want to build a GPT-style transformer from scratch and pre-train
it on a real language modeling dataset used in published research, so that I understand every
architectural decision and can observe how language modeling loss and perplexity evolve during
training.

#### Acceptance Criteria

1. THE Project_2 SHALL implement a decoder-only transformer (GPT-style) from scratch in PyTorch,
   including: token embeddings, learned positional encodings, multi-head causal self-attention
   with masking, feed-forward layers with GELU activation, pre-norm layer normalization, weight
   tying between embedding and LM head, and a language modeling head.

2. THE Project_2 SHALL train on the TinyStories dataset (Eldan & Li, 2023 — available via
   Hugging Face Datasets as `roneneldan/TinyStories`) using a 95/5 train/validation split,
   and SHALL use a subset of up to 50,000 stories to keep CPU training tractable.

3. THE Project_2 SHALL implement a BPE tokenizer using the `tokenizers` library (Hugging Face)
   trained on the TinyStories training split, and SHALL save the trained tokenizer vocabulary
   and merges to disk for reproducibility.

4. WHEN the Learner runs pre-training, THE Project_2 SHALL log training loss, validation
   perplexity, gradient norm, and learning rate at configurable intervals and SHALL save all
   metrics to a structured JSON experiment log.

5. THE Project_2 SHALL implement cosine decay with linear warmup as the LR schedule and SHALL
   implement gradient accumulation to simulate an effective batch size of at least 256 tokens
   on hardware with limited memory.

6. WHEN a pre-training run completes or is interrupted, THE Project_2 SHALL save a checkpoint
   containing model weights, optimizer state, scheduler state, current step, and best
   validation perplexity, and SHALL support resuming from any checkpoint.

7. WHEN a forward pass is executed, THE Project_2 SHALL provide a hook-based inspection
   utility that captures attention weights for each head in each layer and saves them as
   heatmap visualizations to disk.

8. WHEN the Learner queries the trained model, THE Project_2 SHALL support greedy decoding
   and temperature-scaled top-p (nucleus) sampling as inference modes, with a fixed seed
   option for reproducible generation.

9. THE Project_2 SHALL include a weight analysis utility that computes and displays the
   distribution of weights in each layer (embedding, attention projections, FFN) after
   training, including spectral norms of weight matrices.

10. FOR ALL valid token sequences up to the model's context length, THE Project_2 SHALL
    produce output logits of shape `[batch, seq_len, vocab_size]` without raising runtime
    errors (shape correctness property).

11. THE Project_2 SHALL be organized as a Python module with the following separation of
    concerns: `model.py`, `tokenizer.py`, `data.py`, `train.py`, `evaluate.py`, `config.py`,
    `generate.py`, and `visualize.py`.

12. THE Project_2 SHALL include type annotations on all public functions and SHALL include
    unit tests for: tokenizer round-trip (encode then decode returns original text), attention
    mask correctness (no future token leakage), and data loader output shapes.

13. THE Project_2 SHALL include a research-quality README with: motivation, dataset description
    with citation (Eldan & Li, 2023), model architecture table (layers, heads, d_model,
    parameters), training curve plots, sample generated stories, and citations for the
    foundational papers (Vaswani et al., 2017; Radford et al., 2019).

14. WHEN pre-training completes, THE Project_2 SHALL achieve a validation perplexity
    demonstrably lower than an untrained baseline (random weights), confirming the model
    has learned language structure.

---

### Requirement 3: Project 3 — Supervised Fine-Tuning, Instruction Tuning, and RLHF

**User Story:** As a Learner, I want to fine-tune a small pre-trained language model using
the exact datasets and methods used in alignment research — Alpaca for instruction tuning and
Anthropic HH-RLHF for preference learning — so that I understand how SFT, reward modeling,
and RLHF work at the implementation level.

#### Acceptance Criteria

1. THE Project_3 SHALL load a small pre-trained model (GPT-2 small from Hugging Face, or the
   model trained in Project 2) and SHALL perform supervised fine-tuning (SFT) on the Stanford
   Alpaca dataset (`tatsu-lab/alpaca` via Hugging Face Datasets, 52K instruction-response pairs)
   using a 90/10 train/validation split with a fixed seed.

2. WHEN SFT runs, THE Project_3 SHALL log per-layer gradient magnitudes at configurable
   intervals so the Learner can observe which layers change most during fine-tuning versus
   the pre-training baseline.

3. THE Project_3 SHALL implement instruction fine-tuning using the prompt template from the
   original Alpaca paper (Taori et al., 2023) and SHALL demonstrate the difference in model
   outputs on a fixed evaluation prompt set before and after SFT.

4. THE Project_3 SHALL train a reward model on the Anthropic HH-RLHF dataset
   (`Anthropic/hh-rlhf` via Hugging Face Datasets) using chosen/rejected preference pairs,
   and SHALL evaluate the reward model's preference accuracy on a held-out validation split.

5. THE Project_3 SHALL implement a minimal RLHF pipeline consisting of: the trained reward
   model, a policy model initialized from the SFT checkpoint, and a PPO-style update loop
   with KL penalty against the SFT reference policy.

6. WHEN the RLHF training loop runs, THE Project_3 SHALL log reward scores, KL divergence
   from the reference policy, policy loss, and value loss at each update step to a structured
   JSON experiment log.

7. THE Project_3 SHALL implement an RLAIF variant where a FLAN-T5-small model (available via
   Hugging Face) generates preference scores in place of the human-labeled reward model, and
   SHALL log a comparison of reward distributions between RLHF and RLAIF configurations.

8. THE Project_3 SHALL include a before/after comparison utility that generates model outputs
   on the same 20 fixed evaluation prompts at each stage (base, SFT, RLHF, RLAIF) and saves
   the outputs to a structured comparison file for qualitative analysis.

9. IF the reward model produces a score outside a configurable bound during PPO updates,
   THEN THE Project_3 SHALL log a warning with the step number and reward value and SHALL
   clip the reward to the configured bound before the policy update.

10. THE Project_3 SHALL implement cosine decay with warmup for all fine-tuning stages and
    SHALL save checkpoints at configurable intervals with resume capability.

11. THE Project_3 SHALL be organized as a Python module with the following separation of
    concerns: `sft.py`, `reward_model.py`, `rlhf.py`, `rlaif.py`, `data.py`, `evaluate.py`,
    `config.py`, and `compare.py`.

12. THE Project_3 SHALL include type annotations on all public functions and SHALL include
    unit tests for: reward model output shape, KL divergence computation correctness, and
    data pipeline (no overlap between train and validation splits).

13. THE Project_3 SHALL include a research-quality README with: motivation, dataset descriptions
    with citations (Taori et al., 2023 for Alpaca; Bai et al., 2022 for HH-RLHF; Ouyang et al.,
    2022 for InstructGPT/RLHF), methodology, results table comparing outputs across stages,
    and a discussion of alignment tax observations.

14. THE Project_3 SHALL complete all fine-tuning stages on CPU in under 90 minutes using
    appropriately small model and dataset subset sizes.

---

### Requirement 4: Project 4 — Vision Transformers and Image Classification

**User Story:** As a Learner, I want to build and train a Vision Transformer from scratch on
a real vision benchmark used in published research, so that I understand how image patches,
positional embeddings, and attention work in the vision domain and can compare ViT against
a CNN baseline on the same data.

#### Acceptance Criteria

1. THE Project_4 SHALL implement a Vision Transformer (ViT) from scratch in PyTorch following
   Dosovitskiy et al. (2020), including: patch embedding via a learned linear projection,
   class token, learned 1D positional embeddings, transformer encoder blocks with pre-norm,
   and a classification MLP head.

2. THE Project_4 SHALL train on CIFAR-10 (available via `torchvision.datasets.CIFAR10`,
   no manual download) using the standard 50,000/10,000 train/test split, with a 10% held-out
   validation split taken from the training set using a fixed seed.

3. WHERE the Learner wants to train on a harder benchmark, THE Project_4 SHALL support
   ImageNette (fast.ai's 10-class ImageNet subset, available via Hugging Face Datasets as
   `frgfm/imagenette`) as a drop-in dataset replacement via a config flag.

4. THE Project_4 SHALL implement standard image augmentation for training (random horizontal
   flip, random crop with padding, normalization using dataset statistics) and SHALL apply
   only normalization at validation/test time.

5. WHEN a forward pass is executed on an image, THE Project_4 SHALL provide a visualization
   utility that overlays attention weights from the class token onto the original image as a
   heatmap, saved to disk, showing which patches the model attends to per class.

6. THE Project_4 SHALL compare two patch size configurations (4×4 and 8×8 patches on CIFAR-10)
   and SHALL log how patch size affects model parameter count, training speed (steps/sec),
   and validation accuracy.

7. THE Project_4 SHALL implement attention rollout (Abnar & Zuidema, 2020) to produce
   saliency maps saved to disk for a representative sample of test images per class.

8. THE Project_4 SHALL include a CNN baseline (a small ResNet-18 or equivalent ConvNet)
   trained on the same dataset with the same augmentation and LR schedule, and SHALL produce
   a comparison table of validation accuracy, parameter count, and training time.

9. WHEN a training run completes or is interrupted, THE Project_4 SHALL save a checkpoint
   containing model weights, optimizer state, scheduler state, current epoch, and best
   validation accuracy, and SHALL support resuming from any checkpoint.

10. THE Project_4 SHALL implement cosine annealing with linear warmup as the LR schedule
    and SHALL log training loss, validation accuracy (top-1), gradient norm, and LR at
    configurable intervals to a structured JSON experiment log.

11. THE Project_4 SHALL be organized as a Python module with the following separation of
    concerns: `model.py` (ViT), `baseline.py` (CNN), `data.py`, `train.py`, `evaluate.py`,
    `config.py`, `visualize.py`, and `attention_viz.py`.

12. THE Project_4 SHALL include type annotations on all public functions and SHALL include
    unit tests for: patch embedding output shape, positional embedding shape, attention mask
    correctness, and data loader output shapes and value ranges.

13. THE Project_4 SHALL include a research-quality README with: motivation, dataset description,
    architecture table, training curve plots, attention visualization examples, comparison table
    (ViT vs CNN), and citations for foundational papers (Dosovitskiy et al., 2020; He et al.,
    2016 for ResNet; Abnar & Zuidema, 2020 for attention rollout).

14. THE Project_4 SHALL complete training in under 45 minutes on a CPU-only AMD Ryzen 9
    machine for the CIFAR-10 configuration.

---

### Requirement 5: Project 5 — Reasoning, Inference Strategies, and KV Cache

**User Story:** As a Learner, I want to implement chain-of-thought reasoning on a real
multi-step reasoning benchmark, explore all major inference strategies, and build a KV cache
from scratch, so that I understand how reasoning emerges and how inference is made efficient
at the implementation level.

#### Acceptance Criteria

1. THE Project_5 SHALL implement greedy decoding, beam search, top-k sampling, top-p (nucleus)
   sampling, and temperature scaling as separate, inspectable inference functions with full
   type annotations.

2. THE Project_5 SHALL evaluate inference strategies on the GSM8K dataset (Cobbe et al., 2021
   — available via Hugging Face Datasets as `gsm8k`) using the standard test split of 1,319
   grade-school math problems, reporting exact-match accuracy per strategy.

3. WHEN the Learner runs each inference strategy on the same GSM8K prompt, THE Project_5
   SHALL log the token probabilities at each decoding step so the Learner can observe how
   each strategy selects the next token differently.

4. THE Project_5 SHALL implement a KV cache from scratch and SHALL measure and log the
   speedup in tokens-per-second compared to inference without caching on the same model and
   prompt set, with results saved to a structured benchmark file.

5. THE Project_5 SHALL implement chain-of-thought (CoT) prompting (Wei et al., 2022) and
   SHALL compare zero-shot, few-shot, and chain-of-thought prompting on a fixed subset of
   GSM8K problems, logging accuracy and average response length per condition.

6. THE Project_5 SHALL implement a minimal scratchpad reasoning loop where the model generates
   intermediate reasoning steps before producing a final answer, and SHALL evaluate this on
   a subset of BIG-Bench Hard tasks (available via Hugging Face Datasets as
   `maveriq/bigbenchhard`) covering logical deduction and multi-step arithmetic.

7. WHEN beam search is executed, THE Project_5 SHALL display the top-k beam candidates and
   their cumulative log-probabilities at each step, saved to a structured log file.

8. FOR ALL inference strategies, THE Project_5 SHALL produce identical outputs given the same
   model, prompt, and random seed (reproducibility property), verified by a unit test that
   runs each strategy twice and asserts output equality.

9. THE Project_5 SHALL include a comparison table saved to disk showing exact-match accuracy,
   output diversity (distinct-n), and tokens-per-second across all inference strategies on
   a fixed 50-problem subset of GSM8K.

10. THE Project_5 SHALL be organized as a Python module with the following separation of
    concerns: `inference.py` (all decoding strategies), `kv_cache.py`, `reasoning.py`
    (CoT and scratchpad), `data.py`, `evaluate.py`, `config.py`, and `benchmark.py`.

11. THE Project_5 SHALL include type annotations on all public functions and SHALL include
    unit tests for: KV cache correctness (cached vs uncached outputs are identical), beam
    search beam count invariant, and reproducibility of all sampling strategies under fixed seed.

12. THE Project_5 SHALL include a research-quality README with: motivation, dataset descriptions
    with citations (Cobbe et al., 2021 for GSM8K; Wei et al., 2022 for CoT; Srivastava et al.,
    2022 for BIG-Bench), methodology, results table, and a discussion of when each inference
    strategy is appropriate.

---

### Requirement 6: Project 6 — Professional Benchmark Evaluation and Model Analysis

**User Story:** As a Learner, I want to evaluate language models using the exact benchmarks
and tooling used on the Hugging Face Open LLM Leaderboard, so that I can produce results
that are directly comparable to published model evaluations and gain career-relevant skills
in rigorous model assessment.

#### Acceptance Criteria

1. THE Project_6 SHALL implement evaluation pipelines for the four benchmarks used on the
   Hugging Face Open LLM Leaderboard: ARC-Challenge (Clark et al., 2018), HellaSwag
   (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), and TruthfulQA (Lin et al., 2022).

2. WHEN a benchmark evaluation runs, THE Project_6 SHALL use the `lm-evaluation-harness`
   library (EleutherAI) as the evaluation backend, using the same task names and shot
   configurations as the Hugging Face leaderboard (ARC: 25-shot, HellaSwag: 10-shot,
   MMLU: 5-shot, TruthfulQA: 0-shot MC).

3. THE Project_6 SHALL evaluate at least two publicly available small models (GPT-2 and
   Pythia-160M from EleutherAI, both available via Hugging Face) and SHALL produce a
   comparison report in both CSV and Markdown format matching the column structure of the
   Hugging Face leaderboard.

4. WHEN evaluation completes, THE Project_6 SHALL display per-task accuracy, normalized
   scores, and an aggregate score, and SHALL include inline comments explaining what
   capability each benchmark measures and why it is a meaningful signal.

5. THE Project_6 SHALL include a weight analysis module that computes and visualizes:
   weight norms per layer, singular value distributions (via SVD) of weight matrices,
   and dead neuron ratios in feed-forward layers for each evaluated model.

6. THE Project_6 SHALL include an activation analysis module that records and plots
   activation distributions across layers for a representative set of inputs from each
   benchmark, showing how representations evolve through depth.

7. THE Project_6 SHALL include a dataset exploration module that loads and analyzes three
   datasets commonly used in LLM training: a 1GB streaming subset of The Pile
   (`EleutherAI/pile` via streaming mode), a 1GB streaming subset of C4
   (`allenai/c4` via streaming mode), and OpenWebText (`Skylion007/openwebtext` via
   streaming mode), reporting token count estimates, vocabulary statistics, and sample
   inspection without requiring full dataset downloads.

8. THE Project_6 SHALL include a research paper reference section in its README listing
   the foundational papers for each benchmark and each evaluated model, with a one-line
   summary of why each paper is essential.

9. WHEN the Learner runs the full evaluation suite, THE Project_6 SHALL complete all
   benchmark evaluations on CPU in under 2 hours using the small model variants specified.

10. IF a model fails to load or a benchmark task fails during evaluation, THEN THE Project_6
    SHALL log the error with the model name, task name, and stack trace summary, and SHALL
    continue evaluating remaining tasks without crashing.

11. THE Project_6 SHALL be organized as a Python module with the following separation of
    concerns: `evaluate.py` (harness integration), `weight_analysis.py`, `activation_analysis.py`,
    `dataset_explorer.py`, `report.py` (CSV/Markdown generation), and `config.py`.

12. THE Project_6 SHALL include type annotations on all public functions and SHALL include
    unit tests for: report generation (correct column structure), error handling (graceful
    continuation on task failure), and dataset streaming (correct sample format).

13. THE Project_6 SHALL include a research-quality README with: motivation, benchmark
    descriptions with citations, model descriptions with citations (Radford et al., 2019
    for GPT-2; Biderman et al., 2023 for Pythia), results tables, weight and activation
    analysis plots, and a discussion of what the results reveal about each model's strengths
    and weaknesses.

---

### Requirement 7: Cross-Project Research Standards

**User Story:** As a Learner, I want every project to follow the code quality, reproducibility,
and documentation standards used at top AI research labs, so that the codebase is a
professional portfolio artifact and a reference I can build on.

#### Acceptance Criteria

1. THE Curriculum SHALL organize all six projects under a single repository with a top-level
   README that describes the learning path, prerequisites, hardware requirements, and how to
   run each project end-to-end.

2. THE Curriculum SHALL use config-driven training throughout: every project SHALL define all
   hyperparameters in a YAML file or Python dataclass config, with no hardcoded hyperparameters
   in training scripts.

3. WHEN any training or evaluation run starts, THE Project SHALL log the full config (all
   hyperparameters, dataset paths, random seeds, library versions) to the experiment log
   file as the first entry, enabling exact reproduction of any run.

4. THE Curriculum SHALL fix all random seeds (Python, NumPy, PyTorch) at the start of every
   training run using a configurable seed value, and SHALL document the seed in the experiment
   log.

5. WHEN the Learner sets up any project, THE Project SHALL install all dependencies via a
   single `pip install -r requirements.txt` command with pinned versions, and SHALL include
   a `setup.py` or `pyproject.toml` for installable package structure.

6. THE Curriculum SHALL ensure every project runs end-to-end on a CPU-only machine with no
   CUDA dependency, using PyTorch CPU builds, and SHALL document mixed precision awareness
   (why FP16/BF16 matters and how it would be enabled on GPU) in code comments.

7. THE Curriculum SHALL include a `data/` module in each project that downloads or streams
   the required dataset automatically on first run using Hugging Face Datasets or
   torchvision, with no manual download steps required from the Learner.

8. WHILE a training or evaluation run is in progress, THE Project SHALL display a progress
   bar via `tqdm` showing current step, estimated time remaining, and live metric values
   (loss, accuracy, or perplexity as appropriate).

9. THE Curriculum SHALL include a `.gitignore` in the repository root that excludes
   checkpoints, downloaded datasets, experiment logs, and compiled Python files, ensuring
   no large binary files are committed to version control.

10. THE Curriculum SHALL reference at least two foundational research papers per project
    in code comments at the point of implementation, using the format:
    `# Ref: Author et al., Year — "Paper Title" — reason this is relevant`.

11. THE Curriculum SHALL NOT require any paid API, cloud service, or GPU rental — all
    computation SHALL run locally on the Learner's CPU-only machine.

12. THE Curriculum SHALL include a top-level `Makefile` with targets for: `make setup`
    (install dependencies), `make test` (run all unit tests), `make train-all` (run all
    six projects sequentially), and `make clean` (remove checkpoints and logs).
