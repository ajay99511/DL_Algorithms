# Project 3: Supervised Fine-Tuning, Instruction Tuning, and RLHF

## Motivation

Pre-trained language models like GPT-2 are powerful next-token predictors, but they are not inherently helpful, harmless, or honest. Alignment research addresses this gap by teaching models to follow instructions and produce outputs that humans prefer. This project implements the full alignment pipeline:

1. **Supervised Fine-Tuning (SFT)** — fine-tune GPT-2 on human-written instruction-response pairs
2. **Reward Model Training** — learn a scalar reward from human preference data (chosen vs. rejected)
3. **RLHF (PPO)** — optimize the policy against the reward model while penalizing KL divergence from the SFT reference
4. **RLAIF** — replace the human reward model with an AI judge (FLAN-T5-small) and compare distributions

This pipeline mirrors the approach described in InstructGPT (Ouyang et al., 2022) and Anthropic's Constitutional AI work (Bai et al., 2022), scaled down to run on a CPU-only machine.

---

## Datasets

### Stanford Alpaca
- **Source**: `tatsu-lab/alpaca` on HuggingFace Datasets
- **Size**: ~52,000 instruction-following examples
- **Format**: `{instruction, input, output}` triples
- **Use**: SFT fine-tuning of GPT-2
- **Citation**: Taori et al., 2023 — *Alpaca: A Strong, Replicable Instruction-Following Model*. Stanford Center for Research on Foundation Models.

### Anthropic HH-RLHF
- **Source**: `Anthropic/hh-rlhf` on HuggingFace Datasets
- **Size**: ~160,000 preference pairs
- **Format**: `{chosen, rejected}` conversation strings
- **Use**: Reward model training via Bradley-Terry pairwise ranking loss
- **Citation**: Bai et al., 2022 — *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. Anthropic.

---

## Methodology

### Stage 1: Supervised Fine-Tuning (SFT)

GPT-2 small (117M parameters) is fine-tuned on Alpaca using the standard instruction template:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The input section is omitted when empty. Training uses AdamW with cosine learning rate warmup, gradient accumulation, and per-layer gradient norm logging.

### Stage 2: Reward Model Training

A scalar reward head (linear layer) is attached to a GPT-2 backbone. The model is trained on HH-RLHF preference pairs using the Bradley-Terry loss:

```
L = -log σ(r_chosen - r_rejected)
```

This loss encourages the model to assign higher rewards to preferred responses.

### Stage 3: RLHF via PPO (REINFORCE-style)

The SFT model serves as both the policy being optimized and the frozen reference policy. At each step:

1. Generate a response from the policy (greedy decoding, max 64 tokens)
2. Score the response with the reward model; clip to `±reward_clip_bound`
3. Compute token-level KL divergence: `KL(policy || ref_policy)`
4. Compute adjusted reward: `r_adj = r - kl_coeff × KL`
5. Update policy via REINFORCE gradient

The KL penalty prevents the policy from drifting too far from the SFT baseline (the "alignment tax" tradeoff).

### Stage 4: RLAIF (AI Feedback)

FLAN-T5-small acts as an AI judge, prompted with:
```
Rate this response 1-10: {response}
```

The parsed scores are compared against the trained reward model's scores, providing insight into how well an AI judge correlates with human preferences.

---

## Module Structure

| Module | Description |
|---|---|
| `config.py` | `AlignmentConfig` dataclass with all hyperparameters |
| `data.py` | Alpaca and HH-RLHF data loaders |
| `sft.py` | Supervised fine-tuning loop |
| `reward_model.py` | `RewardModel` class and training loop |
| `rlhf.py` | PPO step and RLHF training orchestration |
| `rlaif.py` | FLAN-T5 AI judge and score comparison |
| `evaluate.py` | Gradient magnitude logging, reward accuracy |
| `compare.py` | Stage-by-stage output comparison on 20 fixed prompts |
| `tests/` | Property-based and unit tests |

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `gpt2` | GPT-2 small, 117M parameters |
| SFT learning rate | 2e-5 | AdamW |
| SFT batch size | 8 | Per micro-batch |
| SFT grad accum steps | 4 | Effective batch = 32 |
| SFT epochs | 3 | |
| RM learning rate | 1e-5 | AdamW |
| RM epochs | 2 | |
| PPO steps | 500 | |
| PPO learning rate | 1e-6 | |
| KL coefficient | 0.1 | Penalty weight |
| Reward clip bound | 5.0 | ±5.0 |
| Max sequence length | 512 | Truncated |
| RLAIF judge | `google/flan-t5-small` | |

---

## Results

| Stage | Example Output (prompt: "Explain gradient descent") |
|---|---|
| Base GPT-2 | *(raw continuation, often incoherent)* |
| SFT | *(structured response following instruction format)* |
| RLHF | *(more helpful, concise response)* |

*Note: Full results require running the training pipeline. See `outputs/project3/stage_comparison.json` after training.*

---

## Discussion: Alignment Tax

The KL penalty in PPO creates a fundamental tradeoff: higher KL coefficients keep the model closer to the SFT baseline (safer, more coherent) but limit how much the reward model can improve the policy. Lower KL coefficients allow larger policy updates but risk reward hacking — the model learns to exploit weaknesses in the reward model rather than genuinely improving.

This tradeoff is known as the **alignment tax**: the cost in capability or diversity paid to achieve safer, more aligned behavior. Empirically, InstructGPT found that RLHF models were preferred by human raters despite scoring slightly lower on some NLP benchmarks.

---

## Foundational Papers

- **Ouyang et al., 2022** — *Training language models to follow instructions with human feedback* (InstructGPT). OpenAI. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Bai et al., 2022** — *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. Anthropic. [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
- **Taori et al., 2023** — *Alpaca: A Strong, Replicable Instruction-Following Model*. Stanford CRFM.
- **Schulman et al., 2017** — *Proximal Policy Optimization Algorithms*. OpenAI. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **Christiano et al., 2017** — *Deep Reinforcement Learning from Human Preferences*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- **Radford et al., 2019** — *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI.
