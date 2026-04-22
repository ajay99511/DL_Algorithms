# Deep Learning & LLM Mastery Curriculum

A research-grade, hands-on deep learning and LLM curriculum that runs entirely on a
**CPU-only AMD Ryzen 9 laptop** using PyTorch. Six self-contained projects that build on
each other — from training loop fundamentals through transformer pre-training, alignment,
vision transformers, reasoning, and professional benchmark evaluation.

Every project mirrors the standards of AI researchers at top labs: config-driven training,
reproducible experiments, structured logging, modular code, property-based tests, and
research-quality READMEs with paper citations.

---

## Learning Path

| # | Project | What You Build | Key Dataset | CPU Time |
|---|---------|---------------|-------------|----------|
| 1 | [MLP Training Loop](backprop/README.md) | MLP from scratch, weight init strategies, gradient clipping, activation stats | California Housing | < 5 min |
| 2 | [Transformer Pre-training](pretrain/README.md) | GPT-style decoder transformer, BPE tokenizer, attention heatmaps, weight analysis | TinyStories (Eldan & Li, 2023) | < 30 min |
| 3 | [Alignment: SFT + RLHF](finetune/README.md) | Supervised fine-tuning, reward model, PPO loop, RLAIF with FLAN-T5 | Stanford Alpaca + Anthropic HH-RLHF | < 90 min |
| 4 | [Vision Transformer](visiontx/README.md) | ViT from scratch, attention rollout, saliency maps, CNN vs ViT comparison | CIFAR-10 / ImageNette | < 45 min |
| 5 | [Reasoning & Inference](infer/README.md) | Greedy, beam search, top-k/p sampling, KV cache from scratch, chain-of-thought | GSM8K + BIG-Bench Hard | — |
| 6 | [Benchmark Evaluation](evaluate/README.md) | lm-evaluation-harness, ARC/HellaSwag/MMLU/TruthfulQA, weight & activation analysis | HF Open LLM Leaderboard benchmarks | < 2 hrs |

---

## Prerequisites

- Python 3.11+
- pip
- ~8 GB RAM (16 GB recommended for Project 3)
- No GPU required — all computation runs on CPU

---

## Hardware Requirements

Designed and tested on **AMD Ryzen 9 (CPU-only)**. All models are sized to complete
training within the time budgets above on a modern laptop CPU. No cloud, no GPU rental,
no paid APIs.

---

## Quickstart

```bash
# 1. Clone the repo
git clone <repo-url>
cd deep-learning-llm-mastery

# 2. Install all dependencies
make setup

# 3. Run the full test suite
make test

# 4. Train all projects sequentially
make train-all

# 5. Clean checkpoints and logs
make clean
```

---

## Project Structure

```
deep-learning-llm-mastery/
├── README.md                    ← You are here
├── Makefile                     ← setup / test / train-all / clean
├── pyproject.toml               ← installable package config
├── requirements.txt             ← pinned dependencies
├── .gitignore
├── shared/                      ← cross-project utilities
│   ├── config.py                ← BaseConfig dataclass + YAML loader
│   ├── logging_utils.py         ← JSONLogger (JSONL experiment logs)
│   ├── checkpointing.py         ← save/load checkpoint with resume
│   ├── seed.py                  ← fix_all_seeds() for reproducibility
│   └── lr_schedule.py           ← cosine decay with linear warmup
├── backprop/                ← MLP training loop fundamentals
├── pretrain/        ← GPT-style transformer pre-training
├── finetune/          ← SFT + RLHF + RLAIF
├── visiontx/                ← Vision Transformer on CIFAR-10
├── infer/          ← Inference strategies + KV cache
├── evaluate/               ← Professional benchmark evaluation
└── outputs/                     ← Git-ignored: checkpoints, logs, plots
```

---

## Running Individual Projects

Each project is a self-contained Python module:

```bash
# Project 1 — MLP
python -m backprop.train

# Project 2 — Transformer pre-training (~30 min)
python -m pretrain.train

# Project 3 — SFT
python -m finetune.sft

# Project 3 — RLHF (after SFT)
python -m finetune.rlhf

# Project 4 — ViT on CIFAR-10
python -m visiontx.train --model vit

# Project 4 — ResNet baseline
python -m visiontx.train --model resnet

# Project 5 — Inference benchmark
python -m infer.benchmark

# Project 6 — Benchmark evaluation
python -m evaluate.evaluate
```

---

## Running Tests

```bash
# All projects
make test

# Individual project
python -m pytest backprop/tests/ -v
python -m pytest pretrain/tests/ -v
python -m pytest finetune/tests/ -v
python -m pytest visiontx/tests/ -v
python -m pytest infer/tests/ -v
python -m pytest evaluate/tests/ -v
```

All tests use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing.
18 correctness properties are verified across all projects.

---

## Cross-Project Standards

Every project follows these research-lab standards:

- **Config-driven**: all hyperparameters in `config.yaml` / dataclass — zero hardcoded values
- **Reproducible**: fixed seeds logged at run start; exact reproduction from any experiment log
- **Checkpointing**: save/resume at any epoch or step
- **Structured logging**: JSONL experiment logs with full config as first entry
- **LR scheduling**: cosine decay with linear warmup throughout
- **Gradient accumulation**: simulate larger batch sizes on limited hardware
- **Type annotations**: all public functions fully typed
- **No cloud**: 100% local, CPU-only, no paid APIs

---

## Foundational Papers

| Paper | Relevance |
|-------|-----------|
| Vaswani et al., 2017 — *Attention Is All You Need* | The transformer architecture used in Projects 2–6 |
| Radford et al., 2019 — *Language Models are Unsupervised Multitask Learners* (GPT-2) | Decoder-only LM, weight tying, pre-norm |
| Eldan & Li, 2023 — *TinyStories* | Pre-training dataset for Project 2 |
| Taori et al., 2023 — *Alpaca* | Instruction fine-tuning dataset for Project 3 |
| Ouyang et al., 2022 — *InstructGPT* | RLHF methodology for Project 3 |
| Bai et al., 2022 — *Anthropic HH-RLHF* | Preference dataset for reward model in Project 3 |
| Dosovitskiy et al., 2020 — *An Image is Worth 16x16 Words* | ViT architecture for Project 4 |
| He et al., 2016 — *Deep Residual Learning* | ResNet baseline for Project 4 |
| Abnar & Zuidema, 2020 — *Quantifying Attention Flow* | Attention rollout for Project 4 |
| Wei et al., 2022 — *Chain-of-Thought Prompting* | CoT reasoning for Project 5 |
| Cobbe et al., 2021 — *GSM8K* | Reasoning benchmark for Project 5 |
| Clark et al., 2018 — *ARC* | Science reasoning benchmark for Project 6 |
| Zellers et al., 2019 — *HellaSwag* | Commonsense benchmark for Project 6 |
| Hendrycks et al., 2021 — *MMLU* | Knowledge benchmark for Project 6 |
| Lin et al., 2022 — *TruthfulQA* | Factual accuracy benchmark for Project 6 |
| Loshchilov & Hutter, 2019 — *AdamW* | Optimizer used throughout |
| Loshchilov & Hutter, 2017 — *SGDR* | Cosine LR schedule used throughout |
