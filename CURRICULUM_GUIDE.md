# Deep Learning & LLM Mastery — Complete Student Guide

> A step-by-step walkthrough of all six projects: what they build, why every decision was made,
> and what is happening inside the code at every stage. Written for a student who wants to
> understand the domain deeply, not just run the scripts.

---

## How the Curriculum Is Structured

The six projects form a deliberate learning arc:

```
Project 1 — MLP fundamentals (training loop, optimization, data)
    ↓
Project 2 — Transformer architecture (attention, language modeling, generation)
    ↓
Project 3 — Alignment (SFT, reward models, RLHF, RLAIF)
    ↓
Project 4 — Vision (ViT, ResNet, attention visualization)
    ↓
Project 5 — Inference (decoding strategies, KV cache, chain-of-thought)
    ↓
Project 6 — Evaluation (benchmarks, calibration, weight/activation analysis)
```

Each project is self-contained but builds on the vocabulary and intuitions of the ones before it.
Every project runs entirely on CPU — no GPU, no cloud, no paid APIs.

---

## Cross-Project Standards (Read This First)

Before diving into individual projects, understand the shared infrastructure that every project uses.

### Config-Driven Everything

Every hyperparameter lives in a `config.yaml` file and a matching Python dataclass (e.g., `MLPConfig`, `TransformerConfig`). Nothing is hardcoded in training scripts. This means:

- You can reproduce any experiment exactly by saving the config file.
- You can sweep hyperparameters by changing one line in YAML.
- The config is always the first entry written to the experiment log.

### Structured Logging (JSONL)

Every training run writes a `.jsonl` file (one JSON object per line). Each line has a `"type"` field:
- `"config"` — the full config at run start
- `"train_step"` — loss, learning rate, gradient norm at each logged step
- `"val_epoch"` — validation metrics after each epoch
- `"checkpoint"` — when a new best checkpoint is saved
- `"warning"` — NaN loss, missing files, etc.

This makes it trivial to plot any metric, compare runs, or debug failures.

### Reproducibility

`fix_all_seeds(seed)` is called at the start of every run. It sets seeds for Python's `random`, NumPy, and PyTorch simultaneously. The seed is stored in the config, so any run can be reproduced exactly.

### Checkpointing

`save_checkpoint()` and `load_checkpoint()` in `shared/checkpointing.py` save the model state dict, optimizer state, scheduler state, current epoch/step, and best metric. This means training can be interrupted and resumed at any point with `--resume`.

### Learning Rate Schedule

All projects use **cosine decay with linear warmup** from `shared/lr_schedule.py`. The LR rises linearly from 0 to the peak value over `warmup_steps`, then follows a cosine curve down to a minimum. This is the standard schedule used in GPT-2, BERT, and most modern LLMs.

### Gradient Accumulation

When hardware limits batch size, gradient accumulation simulates a larger effective batch. Instead of one optimizer step per batch, gradients are accumulated over `grad_accum_steps` micro-batches before stepping. Effective batch size = `batch_size × grad_accum_steps`.

### Gradient Clipping

`nn.utils.clip_grad_norm_(model.parameters(), max_norm)` is called before every optimizer step. This prevents exploding gradients — a common failure mode in deep networks and transformers.

---

---

# Project 1 — MLP Training Loop Fundamentals

**Directory:** `backprop/`
**Dataset:** California Housing (sklearn)
**Task:** Regression — predict median house value from 8 census features
**CPU time:** < 5 minutes

---

## What This Project Is Really About

Project 1 is not about California Housing. It is about internalizing every component of the
deep learning training loop in isolation, before any of the complexity of transformers or
language modeling is introduced. The dataset is simple on purpose — it lets you focus entirely
on the mechanics of training.

By the end of this project you will have written and understood:
- A data pipeline with proper train/val/test splits and no data leakage
- A multi-layer perceptron in pure PyTorch
- Three weight initialization strategies and why they matter
- The AdamW optimizer with cosine LR scheduling and warmup
- Gradient accumulation and gradient clipping
- Checkpointing and resumable training
- Activation statistics via forward hooks

---

## The Dataset: California Housing

The dataset comes from the 1990 U.S. Census. Each of the 20,640 samples is a census block
group in California. The 8 input features are:

| Feature | What it means |
|---|---|
| MedInc | Median household income in the block group |
| HouseAge | Median age of houses in the block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Total population of the block group |
| AveOccup | Average number of people per household |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

The target is the median house value in units of $100,000. This is a regression problem —
the model outputs a single continuous number.

**Why this dataset?** It is available directly from `sklearn.datasets.fetch_california_housing`
with no download required, it has a meaningful real-world interpretation, and it is small enough
to train on CPU in minutes.

---

## Data Pipeline (`data.py`)

```
Raw data (20,640 samples)
    ↓
train_test_split(test_size=0.1, seed=42)   → 18,576 trainval + 2,064 test
    ↓
train_test_split(val_size=0.1/0.9, seed=42) → 16,718 train + 1,858 val
    ↓
StandardScaler.fit_transform(X_train)       → fit ONLY on train
StandardScaler.transform(X_val)             → apply same scaler
StandardScaler.transform(X_test)            → apply same scaler
    ↓
TensorDataset → DataLoader (batch_size=32, shuffle=True for train)
```

**Critical concept — data leakage:** The `StandardScaler` is fit exclusively on the training
split. If you fit it on the full dataset before splitting, the validation and test sets would
contain information from the training distribution, making your validation metrics optimistic
and unreliable. This is one of the most common mistakes in machine learning.

**Why StandardScaler?** Neural networks are sensitive to input scale. Features like `Population`
(range: 3–35,682) and `MedInc` (range: 0.5–15) have very different magnitudes. Without
normalization, the optimizer would need very different learning rates for different weights,
making training unstable. StandardScaler transforms each feature to have mean=0 and std=1.

---

## The Model (`model.py`)

The MLP architecture is:

```
Input (8) → Linear(8, 128) → ReLU → Dropout(0.1)
          → Linear(128, 64) → ReLU → Dropout(0.1)
          → Linear(64, 32)  → ReLU → Dropout(0.1)
          → Linear(32, 1)
```

**Why ReLU?** ReLU (Rectified Linear Unit) outputs `max(0, x)`. It is computationally cheap,
does not saturate for positive inputs (unlike sigmoid/tanh), and enables sparse activations.
The main risk is "dead neurons" — neurons that always output 0 because their input is always
negative. Kaiming initialization (see below) is specifically designed to prevent this.

**Why Dropout?** Dropout randomly zeroes a fraction of activations during training. This forces
the network to learn redundant representations and prevents co-adaptation of neurons, which is
a form of overfitting. At inference time, dropout is disabled (via `model.eval()`).

### Weight Initialization Strategies

Initialization matters enormously. If weights start too large, activations explode. If too
small, gradients vanish. Three strategies are implemented:

**Random Normal (`strategy="normal"`):**
```python
nn.init.normal_(module.weight, mean=0.0, std=0.01)
```
Weights drawn from N(0, 0.01). This is naive — it ignores the network architecture entirely.
With ReLU activations, variance shrinks through each layer, leading to vanishing gradients.

**Xavier / Glorot (`strategy="xavier"`):**
```python
nn.init.xavier_uniform_(module.weight)
```
Variance = 2 / (fan_in + fan_out). Designed to keep activation variance constant across layers
for sigmoid/tanh activations. Works reasonably for shallow networks but is theoretically
suboptimal for ReLU.

**Kaiming / He (`strategy="kaiming"`):**
```python
nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
```
Variance = 2 / fan_in. Derived specifically for ReLU activations. Because ReLU zeroes half
the inputs on average, the variance needs to be doubled compared to Xavier to maintain signal
strength. This is the default and consistently converges faster on ReLU networks.

**The math behind Kaiming:** If a layer has `fan_in` inputs and uses ReLU, the expected
variance of the output is `Var(output) = fan_in × Var(weight) × E[ReLU(x)²]`. Since
`E[ReLU(x)²] ≈ 0.5 × Var(x)` for zero-mean inputs, setting `Var(weight) = 2/fan_in`
keeps `Var(output) ≈ Var(input)`.

### Activation Statistics (`activation_stats`)

Forward hooks are registered on each ReLU layer. After a forward pass, each hook captures
the output tensor and computes:
- `mean` — average activation value
- `std` — standard deviation of activations
- `dead_fraction` — fraction of activations ≤ 0 (dead neurons)

A high dead fraction (> 0.5) is a warning sign that the network is not learning effectively.
With Kaiming initialization, dead fractions typically stay below 0.05.

---

## The Training Loop (`train.py`)

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2,
)
```

AdamW (Loshchilov & Hutter, 2019) is Adam with **decoupled weight decay**. Standard Adam
applies L2 regularization by adding `weight_decay × param` to the gradient before the
adaptive update. This is mathematically incorrect — the adaptive scaling changes the effective
regularization strength per parameter. AdamW fixes this by applying weight decay directly to
the parameters after the gradient update, independent of the adaptive scaling.

### Learning Rate Schedule: Cosine with Warmup

```
LR
 ↑
peak_lr ─────────────────────────────────────────────────────────────────────────────────────
        /                                                                                     \
       /  warmup                                                                               \  cosine decay
      /   (5 epochs)                                                                            \
0 ───/                                                                                           ──── min_lr
     0                                                                                          50 epochs
```

**Why warmup?** At the start of training, the model weights are random and the gradient
estimates are noisy. A high learning rate at this stage can cause the optimizer to overshoot
and destabilize training. Linear warmup gradually increases the LR, giving the optimizer time
to build up reliable gradient statistics before taking large steps.

**Why cosine decay?** Cosine decay reduces the LR smoothly, allowing the optimizer to make
fine-grained adjustments as it approaches a minimum. The minimum LR (3e-5) prevents the LR
from reaching zero, which would stop learning entirely.

### Gradient Accumulation

```python
for micro_step, (x, y) in enumerate(train_loader):
    preds = model(x)
    loss = criterion(preds, y) / config.grad_accum_steps  # scale loss
    loss.backward()  # accumulate gradients

    if (micro_step + 1) % config.grad_accum_steps == 0:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

With `batch_size=32` and `grad_accum_steps=4`, the effective batch size is 128. The loss is
divided by `grad_accum_steps` before `.backward()` so that the accumulated gradient is the
average over all 4 micro-batches, not the sum.

### Loss Function: MSE

```python
criterion = nn.MSELoss()
```

Mean Squared Error penalizes large errors more than small ones (quadratic penalty). For
regression, this is the standard choice. The evaluation metrics are RMSE (square root of MSE,
in the same units as the target) and MAE (mean absolute error, more robust to outliers).

### Early Stopping and Checkpointing

After each epoch, validation RMSE is computed. If it improves by more than
`early_stopping_delta`, the model is saved to `outputs/project1/checkpoints/best.pt` and
`patience_counter` resets. If it does not improve for `early_stopping_patience` consecutive
epochs, training stops early.

---

## Evaluation (`evaluate.py`)

```python
result = EvalResult(rmse=..., mae=..., r2=...)
```

Three metrics are computed on the test set:
- **RMSE** — root mean squared error. In the same units as the target ($100k). ~0.54 means
  predictions are off by ~$54,000 on average (quadratic weighting).
- **MAE** — mean absolute error. ~0.39 means predictions are off by ~$39,000 on average
  (linear weighting, more interpretable).
- **R²** — coefficient of determination. 1.0 = perfect, 0.0 = predicts the mean, negative =
  worse than predicting the mean. Measures how much variance the model explains.

---

## Visualization (`visualize.py`)

Two plots are generated:

**Loss curves** (`plot_loss_curves`): Reads the JSONL log and plots train loss (MSE) and
validation RMSE on a dual-axis chart. The warmup phase (epochs 1–5) and cosine decay are
visible in the LR curve.

**Init comparison** (`plot_init_comparison`): Overlays validation RMSE curves from three
separate training runs (one per initialization strategy) on a single chart. This makes the
convergence speed difference between Kaiming, Xavier, and random normal immediately visible.

---

## Key Takeaways from Project 1

1. Data leakage is subtle and easy to introduce — always fit preprocessing on train only.
2. Weight initialization is not a detail — it determines whether training converges at all.
3. Kaiming initialization is the correct default for ReLU networks.
4. AdamW is strictly better than Adam for regularized training.
5. Cosine LR with warmup is the standard schedule for deep learning — learn it once, use it everywhere.
6. Gradient accumulation lets you simulate large batch sizes on limited hardware.
7. Structured logging makes debugging and comparison trivial.

---

---

# Project 2 — Transformer Architecture and Language Model Pre-training

**Directory:** `pretrain/`
**Dataset:** TinyStories (roneneldan/TinyStories, ~2.1M short children's stories)
**Task:** Causal language modeling — predict the next token
**CPU time:** ~30 minutes (10,000 steps)

---

## What This Project Is Really About

Project 2 builds a GPT-style decoder-only transformer **from scratch** — every component
implemented by hand in PyTorch. The goal is not to call `from_pretrained()` and get a working
model. It is to understand exactly what happens inside a transformer: how attention works
mathematically, why pre-LayerNorm is more stable, what weight tying does, and how a model
learns to generate coherent text from random weights.

---

## The Dataset: TinyStories

TinyStories (Eldan & Li, 2023) is a synthetic dataset of ~2.1 million short children's stories
generated by GPT-3.5 and GPT-4. Each story uses vocabulary accessible to a 3–4 year old.

**Why TinyStories?** Small language models trained on general web text (like The Pile) produce
incoherent output because the vocabulary and concepts are too diverse. TinyStories constrains
the distribution to simple English, allowing a 3.5M parameter model to generate grammatically
correct, coherent stories. This makes it ideal for studying language modeling on limited hardware.

The dataset is streamed from HuggingFace — no full download required. The first 50,000 stories
are used (95% train, 5% validation).

---

## The Tokenizer (`tokenizer.py`)

Before the model can process text, it needs to convert strings to integers. This project trains
a **BPE (Byte Pair Encoding) tokenizer** on the TinyStories corpus itself.

**How BPE works:**
1. Start with a vocabulary of individual characters (or bytes).
2. Count all adjacent pairs of tokens in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until the vocabulary reaches the target size (8,000 tokens here).

**Why BPE?** It handles unknown words gracefully (rare words are split into subwords), it is
more efficient than character-level tokenization (fewer tokens per sentence), and it is the
tokenization method used by GPT-2, GPT-3, and most modern LLMs.

**Round-trip property:** A correctly implemented tokenizer satisfies `decode(encode(text)) == text`
for any valid input. This is tested as a correctness property.

---

## Data Pipeline (`data.py`)

```
Stream 50,000 stories from HuggingFace
    ↓
Train BPETokenizer (vocab_size=8,000) on all stories
    ↓
Encode all stories → one long list of token IDs
    ↓
Chunk into non-overlapping windows of context_length=256 tokens
    ↓
Shuffle chunks with fixed seed → 95% train / 5% val split
    ↓
DataLoader (batch_size=16, shuffle=True for train)
```

**Why chunk into fixed-length windows?** Transformers have a fixed context length. By chunking
the concatenated token stream into non-overlapping windows, every token in the corpus is used
for training exactly once, and every batch has the same shape.

**The language modeling objective:** For each window of 256 tokens, the input is tokens[0:255]
and the target is tokens[1:256]. The model learns to predict each token given all previous
tokens. This is called **causal language modeling** or **next-token prediction**.

---

## The Architecture (`model.py`)

### Token and Positional Embeddings

```python
x = token_emb(input_ids) + pos_emb(positions)
```

Each token ID is mapped to a `d_model=128` dimensional vector via a learned embedding table.
Positional embeddings add information about where each token appears in the sequence. Without
positional embeddings, the transformer would be permutation-invariant — it would not know
whether "cat sat" or "sat cat" came first.

**Learned vs. sinusoidal positional embeddings:** This project uses learned positional
embeddings (a simple `nn.Embedding` table). The original transformer paper used fixed
sinusoidal encodings. Both work; learned embeddings are simpler to implement and perform
comparably for fixed context lengths.

### Causal Self-Attention (`CausalSelfAttention`)

This is the core of the transformer. For a sequence of T tokens:

**Step 1 — QKV projection:**
```python
qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
q, k, v = qkv.split(d_model, dim=-1)
```
A single linear layer projects each token's embedding into three vectors: Query (Q), Key (K),
and Value (V). Using a fused QKV projection is more efficient than three separate linear layers.

**Step 2 — Reshape for multi-head attention:**
```python
q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
```
With `n_heads=4` and `d_model=128`, each head has `d_head=32` dimensions. Multi-head attention
allows the model to attend to different aspects of the input simultaneously — one head might
focus on syntactic relationships, another on semantic similarity.

**Step 3 — Scaled dot-product attention:**
```python
attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_head)
```
The attention score between position i and position j is the dot product of Q[i] and K[j],
scaled by `sqrt(d_head)`. Without the scaling, dot products grow large for high-dimensional
vectors, pushing softmax into regions with near-zero gradients.

**Step 4 — Causal mask:**
```python
attn_scores = attn_scores.masked_fill(causal_mask[:T, :T], float("-inf"))
```
The causal mask is an upper-triangular boolean matrix. Positions above the diagonal (future
tokens) are set to -inf before softmax, making their attention weights exactly 0. This ensures
the model cannot "cheat" by looking at future tokens during training.

**Step 5 — Softmax and weighted sum:**
```python
attn_weights = F.softmax(attn_scores, dim=-1)
out = torch.matmul(attn_weights, v)
```
Softmax converts scores to probabilities. The output is a weighted sum of Value vectors —
each output token is a mixture of all previous tokens' values, weighted by how relevant they are.

### The Transformer Block (`TransformerBlock`)

```
x = x + Attention(LayerNorm(x))   ← pre-norm attention residual
x = x + FFN(LayerNorm(x))         ← pre-norm FFN residual
```

**Pre-LayerNorm vs. Post-LayerNorm:** The original transformer paper used post-norm (normalize
after the residual addition). Pre-norm (normalize before) is more stable because the residual
stream is never normalized — gradients flow directly through the residual connections without
being scaled by LayerNorm. This is why GPT-2 and most modern LLMs use pre-norm.

**Feed-Forward Network (FFN):**
```python
ffn = Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)
```
The FFN expands the representation to `d_ff=512` (4× d_model), applies GELU activation, then
projects back. This is where most of the model's "knowledge" is stored — the attention
mechanism routes information, but the FFN transforms it.

**GELU vs. ReLU:** GELU (Gaussian Error Linear Unit) is a smooth approximation of ReLU that
allows small negative values to pass through. It has been empirically shown to work better
than ReLU in transformer FFNs.

### Weight Tying

```python
self.lm_head.weight = self.token_emb.weight
```

The LM head (which maps `d_model` → `vocab_size` to produce logits) shares its weight matrix
with the token embedding table. This reduces parameters by `vocab_size × d_model = 8000 × 128 = 1M`
and improves perplexity because the model learns a consistent representation of tokens in both
the input and output spaces.

---

## The Training Loop (`train.py`)

The training loop is step-based (not epoch-based) because the dataset is streamed:

```python
for step in range(max_steps):
    for micro_step in range(grad_accum_steps):
        batch = next(train_iter)          # (B, context_length)
        input_ids = batch[:, :-1]         # (B, 255) — input
        targets = batch[:, 1:]            # (B, 255) — shifted by 1
        _, loss = model(input_ids, targets)
        (loss / grad_accum_steps).backward()

    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**Effective batch size:** `batch_size=16 × grad_accum_steps=16 = 256` tokens per optimizer step.

**Validation:** Every `max_steps/20` steps, perplexity is computed on the validation set.
Perplexity = `exp(mean cross-entropy loss)`. An untrained model has perplexity ~8,000 (random
over 8,000 vocabulary tokens). After 10,000 steps, perplexity drops to ~25–35, meaning the
model is ~230× more confident than random.

---

## Text Generation (`generate.py`)

Two generation strategies are implemented:

**Greedy decoding:**
```python
next_token = logits[:, -1, :].argmax(dim=-1)
```
Always picks the highest-probability token. Deterministic and fast, but tends to produce
repetitive text because it always takes the locally optimal choice.

**Nucleus (top-p) sampling:**
```python
# Sort probabilities, find smallest set summing to top_p, sample from it
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
sorted_probs[cumulative_probs - sorted_probs > top_p] = 0.0
next_token = torch.multinomial(sorted_probs / sorted_probs.sum(), 1)
```
Samples from the smallest set of tokens whose cumulative probability exceeds `top_p=0.9`.
This adapts dynamically — when the model is confident (peaked distribution), it samples from
few tokens; when uncertain (flat distribution), it samples from more. This produces more
diverse and natural-sounding text than greedy decoding.

---

## Key Takeaways from Project 2

1. Attention is a learned, dynamic routing mechanism — each token decides which other tokens
   to attend to based on content, not position.
2. The causal mask is what makes a language model autoregressive — it enforces that predictions
   only depend on past tokens.
3. Pre-LayerNorm is more stable than post-LayerNorm for deep transformers.
4. Weight tying between embeddings and the LM head reduces parameters and improves perplexity.
5. Perplexity is the exponential of cross-entropy loss — it measures how "surprised" the model
   is by the data. Lower is better.
6. Nucleus sampling produces better text than greedy decoding for open-ended generation.

---

---

# Project 3 — Alignment: Supervised Fine-Tuning, RLHF, and RLAIF

**Directory:** `finetune/`
**Datasets:** Stanford Alpaca (52K instruction pairs) + Anthropic HH-RLHF (160K preference pairs)
**Task:** Teach GPT-2 to follow instructions and produce outputs humans prefer
**CPU time:** ~90 minutes for the full pipeline

---

## What This Project Is Really About

A pre-trained language model like GPT-2 is a powerful next-token predictor, but it is not
helpful. Ask it "How do I bake a cake?" and it might continue with "...is a question many
people ask. The answer depends on..." — it completes the text, it does not answer the question.

**Alignment** is the process of teaching a model to be helpful, harmless, and honest. This
project implements the full alignment pipeline described in InstructGPT (Ouyang et al., 2022):

```
Stage 1: Supervised Fine-Tuning (SFT)
    → Fine-tune GPT-2 on human-written instruction-response pairs
    → Model learns the format and style of helpful responses

Stage 2: Reward Model Training
    → Train a scalar reward model on human preference data
    → Model learns to score responses by how much humans prefer them

Stage 3: RLHF (PPO)
    → Use the reward model to improve the SFT policy via reinforcement learning
    → Policy learns to generate responses that maximize human preference

Stage 4: RLAIF
    → Replace the human reward model with an AI judge (FLAN-T5-small)
    → Compare AI preferences to human preferences
```

---

## Stage 1: Supervised Fine-Tuning (`sft.py`)

### The Dataset: Stanford Alpaca

Alpaca (Taori et al., 2023) contains 52,000 instruction-following examples generated by
prompting `text-davinci-003`. Each example has three fields:

```json
{
  "instruction": "Explain gradient descent in simple terms.",
  "input": "",
  "output": "Gradient descent is an optimization algorithm..."
}
```

The `input` field provides additional context (e.g., a passage to summarize). When empty,
it is omitted from the prompt.

### The Instruction Template

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

This template teaches the model the structure of instruction-following. After SFT, the model
learns to recognize `### Response:` as a cue to generate a helpful answer.

### The Training Objective

SFT is standard causal language modeling — the model predicts the next token in the full
formatted sequence (instruction + response). The loss is computed over all tokens, including
the instruction. This is simpler than masking the instruction tokens, and works well in practice.

```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
loss = outputs.loss  # cross-entropy over all tokens
```

### Why Fine-Tune Instead of Train From Scratch?

GPT-2 already knows English grammar, facts, and reasoning patterns from pre-training on WebText.
Fine-tuning on Alpaca teaches it the *format* of helpful responses without forgetting its
pre-trained knowledge. Training from scratch on 52K examples would produce a much weaker model.

---

## Stage 2: Reward Model Training (`reward_model.py`)

### The Dataset: Anthropic HH-RLHF

The HH-RLHF dataset (Bai et al., 2022) contains 160,000 preference pairs. Each pair has:
- `chosen`: A conversation that human raters preferred
- `rejected`: A conversation that human raters did not prefer

These pairs capture human judgments about what makes a response helpful and harmless.

### The Reward Model Architecture

```python
class RewardModel(nn.Module):
    def __init__(self, backbone, d_model):
        self.backbone = backbone          # GPT-2 transformer
        self.reward_head = nn.Linear(d_model, 1)  # scalar output
```

A GPT-2 backbone produces hidden states. The last token's hidden state (the final position
in the sequence) is passed through a linear layer to produce a single scalar reward.

**Why the last token?** In a causal language model, the last token's hidden state has attended
to all previous tokens and aggregates the full sequence representation. It is the natural
choice for sequence-level classification.

### The Bradley-Terry Loss

```python
loss = -F.logsigmoid(r_chosen - r_rejected).mean()
```

This is the **Bradley-Terry pairwise ranking loss**. It maximizes the probability that the
chosen response has a higher reward than the rejected response. Mathematically:

```
P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
Loss = -log P(chosen > rejected) = -log sigmoid(r_chosen - r_rejected)
```

A perfectly trained reward model assigns higher scores to chosen responses 100% of the time.
In practice, reward accuracy of 65–75% is typical for small models.

---

## Stage 3: RLHF via PPO (`rlhf.py`)

### The Core Idea

After SFT, the model follows instructions but may not produce the *best* responses. RLHF uses
the reward model as a signal to further improve the policy. The key challenge is that the
reward model is imperfect — if we optimize against it too aggressively, the policy will find
ways to "hack" the reward model (produce responses that score high but are not actually good).

### The KL Penalty

```python
adjusted_reward = reward - kl_coeff * KL(policy || ref_policy)
```

The KL divergence between the current policy and the frozen SFT reference policy is computed
at the token level. This penalty prevents the policy from drifting too far from the SFT
baseline. The `kl_coeff=0.1` controls the tradeoff:

- **High KL coefficient:** Policy stays close to SFT (safe, coherent, but limited improvement)
- **Low KL coefficient:** Policy can change more (potentially better rewards, but risks reward hacking)

This tradeoff is called the **alignment tax** — the cost in capability or diversity paid to
achieve safer behavior.

### The PPO Step

```python
def ppo_step(policy, ref_policy, reward_model, batch, config, logger):
    # 1. Generate responses from policy (greedy, max 64 tokens)
    generated_ids = policy.generate(input_ids, max_new_tokens=64, do_sample=False)

    # 2. Score with reward model, clip to ±5.0
    rewards = reward_model(generated_ids, gen_mask).clamp(-5.0, 5.0)

    # 3. Compute token-level KL divergence
    kl = compute_token_kl(policy, ref_policy, generated_ids, gen_mask)

    # 4. Adjusted reward
    adjusted_reward = rewards - kl_coeff * kl

    # 5. REINFORCE policy gradient update
    policy_loss = -(adjusted_reward.detach() * (-policy_outputs.loss)).mean()
    policy_loss.backward()
```

This is a simplified REINFORCE-style update (not full PPO with clipping), appropriate for
CPU-only training. The key insight is that `adjusted_reward.detach()` treats the reward as
a fixed scalar weight on the policy gradient — the policy is updated to increase the
probability of generating responses that received high adjusted rewards.

---

## Stage 4: RLAIF — AI Feedback (`rlaif.py`)

RLAIF (Reinforcement Learning from AI Feedback) replaces the human reward model with an AI
judge. Here, FLAN-T5-small is prompted to rate responses on a 1–10 scale:

```
Rate this response 1-10: {response}
```

The AI scores are compared against the trained reward model's scores across 50 examples.
This reveals how well an AI judge correlates with human preferences — a key question in
alignment research, since human annotation is expensive and AI feedback could scale it.

**FLAN-T5-small** is a 80M parameter encoder-decoder model fine-tuned on a large collection
of instruction-following tasks. It is used here as a zero-shot judge.

---

## Stage Comparison (`compare.py`)

After all stages, 20 fixed prompts are run through:
1. Base GPT-2 (no fine-tuning)
2. SFT model
3. RLHF model

The outputs are saved to `outputs/project3/stage_comparison.json` for qualitative comparison.
This makes the alignment progression visible: base GPT-2 produces incoherent continuations,
SFT produces structured responses, RLHF produces more helpful and concise responses.

---

## Key Takeaways from Project 3

1. Pre-trained models are powerful but not aligned — they predict text, not answer questions.
2. SFT teaches format and style cheaply, but cannot teach the model what humans prefer.
3. The reward model is a learned proxy for human preferences — it is imperfect by design.
4. The KL penalty is essential — without it, RLHF degenerates into reward hacking.
5. RLAIF shows that AI feedback can approximate human feedback, enabling scalable alignment.
6. The alignment tax is real — optimizing for human preference can reduce diversity and capability.

---

---

# Project 4 — Vision Transformer (ViT) for Image Classification

**Directory:** `visiontx/`
**Dataset:** CIFAR-10 (60,000 32×32 color images, 10 classes)
**Task:** Multi-class image classification
**CPU time:** ~45 minutes (ViT, patch=4)

---

## What This Project Is Really About

For nearly a decade, convolutional neural networks (CNNs) dominated computer vision. In 2020,
Dosovitskiy et al. showed that a pure transformer — applied directly to sequences of image
patches — can match or exceed CNN performance when trained at scale. This project implements
both a Vision Transformer and a ResNet-18 baseline from scratch, trains them on CIFAR-10, and
compares their behavior through attention visualization.

The key insight: images can be treated as sequences of patches, just as text is a sequence of
tokens. The same transformer architecture that learns language can learn vision.

---

## The Dataset: CIFAR-10

CIFAR-10 (Krizhevsky, 2009) contains 60,000 32×32 RGB images across 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Standard split: 50,000 training + 10,000 test. This project further splits 5,000 from training
as a validation set (10% of training data).

**Data augmentation (training only):**
```python
transforms.RandomHorizontalFlip()      # randomly flip images left-right
transforms.RandomCrop(32, padding=4)   # randomly crop after padding
transforms.Normalize(mean, std)        # per-channel normalization
```

Augmentation artificially increases the effective training set size by creating variations of
each image. It is applied only during training — validation and test use only normalization.

**Normalization statistics** are computed on the training set:
- Red: mean=0.4914, std=0.2470
- Green: mean=0.4822, std=0.2435
- Blue: mean=0.4465, std=0.2616

---

## The Vision Transformer Architecture (`model.py`)

### Step 1: Patch Embedding (`PatchEmbedding`)

```python
self.projection = nn.Conv2d(
    in_channels=3,
    out_channels=d_model,
    kernel_size=patch_size,
    stride=patch_size,
)
```

A 32×32 image with `patch_size=4` is divided into 64 non-overlapping 4×4 patches. Each patch
contains `4×4×3 = 48` pixel values. A single Conv2d with `kernel_size=patch_size` and
`stride=patch_size` efficiently extracts and linearly projects all patches simultaneously:

```
(B, 3, 32, 32) → Conv2d → (B, 128, 8, 8) → flatten → (B, 64, 128)
```

The result is a sequence of 64 patch embeddings, each of dimension `d_model=128`.

**Patch size tradeoff:**
- `patch_size=4`: 64 patches, longer sequence, finer spatial resolution, slower training
- `patch_size=8`: 16 patches, shorter sequence, coarser resolution, faster training

### Step 2: Class Token

```python
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
x = torch.cat([cls_tokens, x], dim=1)           # (B, 65, d_model)
```

A learnable class token is prepended to the patch sequence. After all transformer blocks,
the class token's output is used for classification. The class token learns to aggregate
information from all patches through attention.

**Why a class token?** The transformer processes all positions equally — there is no natural
"output" position. The class token provides a dedicated position that can attend to all patches
and accumulate a global representation of the image.

### Step 3: Positional Embeddings

```python
x = x + self.pos_embed  # (B, 65, d_model)
```

Learned 1D positional embeddings are added to the patch embeddings. Without them, the
transformer would be permutation-invariant — it would not know whether patch 0 is in the
top-left or bottom-right corner.

**1D vs. 2D positional embeddings:** The original ViT paper used 1D learned embeddings and
found they work as well as 2D encodings, because the model can learn spatial relationships
from the data.

### Step 4: Transformer Encoder Blocks (`ViTEncoderBlock`)

The ViT encoder block is identical to the GPT transformer block from Project 2, with one
critical difference: **no causal mask**. In language modeling, each token can only attend to
previous tokens. In image classification, every patch can attend to every other patch — the
full image is available at once.

```
x = x + Attention(LayerNorm(x))   ← bidirectional attention (no mask)
x = x + FFN(LayerNorm(x))
```

With `n_layers=6`, `n_heads=4`, `d_model=128`, `d_ff=512`, the ViT has ~1.8M parameters.

### Step 5: Classification Head

```python
cls_out = x[:, 0]          # (B, d_model) — class token output
logits = self.head(cls_out) # (B, 10) — one logit per class
```

The class token's final representation is passed through a linear layer to produce 10 logits.
Cross-entropy loss is computed against the ground-truth class labels.

---

## The ResNet-18 Baseline (`baseline.py`)

### Residual Blocks

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        out = relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        out = out + self.shortcut(x)  # skip connection
        return relu(out)
```

The key innovation of ResNet (He et al., 2016) is the **skip connection** (residual connection).
Instead of learning `F(x)`, the block learns `F(x) + x` — the residual. This has two benefits:

1. **Gradient flow:** Gradients can flow directly through the skip connection without passing
   through the convolutions, preventing vanishing gradients in deep networks.
2. **Identity initialization:** If the block learns F(x) ≈ 0, the output is approximately
   the identity — the block can be "skipped" if it is not useful.

### CIFAR-10 Adaptation

The standard ResNet-18 uses a 7×7 conv with stride 2 and max-pooling at the start, designed
for 224×224 ImageNet images. For 32×32 CIFAR-10 images, this would reduce spatial resolution
too aggressively. The `SmallResNet` replaces the stem with a 3×3 conv with stride 1.

With ~11.2M parameters (6× more than the ViT), SmallResNet achieves ~90% validation accuracy
vs. ~75% for the ViT. This reflects the **inductive bias** advantage of CNNs on small datasets:
convolutions assume local spatial structure (nearby pixels are related), which is a strong
prior for natural images. Transformers have no such prior and need more data to learn it.

---

## Attention Visualization (`attention_viz.py`)

### Attention Rollout (Abnar & Zuidema, 2020)

Raw attention weights from a single layer are not interpretable — they show which patches
each position attends to in that layer, but not the effective influence across all layers.
Attention rollout propagates attention through all layers to compute the effective relevance
of each patch for the class token's final prediction.

**Algorithm:**
```python
rollout = identity_matrix  # (T, T)
for attn in attention_weights:  # one per layer
    attn_avg = attn.mean(dim=1)  # average over heads
    # Discard lowest 90% of attention weights (noise reduction)
    attn_avg[attn_avg < threshold] = 0.0
    # Add residual connection: A_hat = 0.5*A + 0.5*I
    attn_hat = 0.5 * attn_avg + 0.5 * identity
    attn_hat = attn_hat / attn_hat.sum(dim=-1, keepdim=True)  # normalize rows
    rollout = attn_hat @ rollout  # propagate
# Extract class token row → relevance of each patch
patch_relevance = rollout[0, 1:]  # exclude class token itself
```

The `0.5*A + 0.5*I` term accounts for residual connections — information flows both through
attention and directly through the skip connection.

**What the visualization shows:** Patches with high rollout scores are the ones the model
"looked at" most when making its classification decision. For a dog image, the model should
attend to the dog's body; for a ship, to the hull and water.

---

## Training Loop (`train.py`)

The training loop is shared between ViT and ResNet. The key difference is the model passed in:

```python
if args.model == "vit":
    net = ViT(cfg)
else:
    net = SmallResNet(n_classes=cfg.n_classes)
train(net, cfg, model_name=args.model)
```

Both models use the same optimizer (AdamW), scheduler (cosine with warmup), gradient clipping,
and checkpointing infrastructure from the shared utilities.

**Validation metric:** Top-1 accuracy — the fraction of images where the model's highest-
probability class matches the ground truth.

---

## Key Takeaways from Project 4

1. Images can be treated as sequences of patches — the same transformer architecture works
   for both language and vision.
2. The class token is a clever trick: a learnable "summary" position that aggregates information
   from all patches through attention.
3. CNNs have a strong inductive bias for local spatial structure that benefits small datasets
   like CIFAR-10. ViTs need more data or augmentation to match CNN performance at this scale.
4. Attention rollout makes the model's "reasoning" interpretable — you can see which patches
   drove the classification decision.
5. Residual connections are essential for training deep networks — they provide gradient
   highways that bypass potentially problematic layers.

---

---

# Project 5 — Reasoning, Inference Strategies, and KV Cache

**Directory:** `infer/`
**Datasets:** GSM8K (grade-school math) + BIG-Bench Hard (diverse reasoning)
**Task:** Evaluate and compare inference strategies on reasoning benchmarks
**CPU time:** Varies by strategy and subset size

---

## What This Project Is Really About

Training a model is only half the story. How you *decode* from a trained model matters
enormously. The same model can produce very different outputs depending on the decoding
strategy. This project explores the full inference stack:

1. **Decoding strategies:** greedy, beam search, top-k sampling, nucleus sampling
2. **KV cache:** the optimization that makes autoregressive generation practical
3. **Chain-of-thought prompting:** eliciting multi-step reasoning from language models
4. **Benchmarking:** measuring accuracy, diversity, and throughput across strategies

---

## The Datasets

### GSM8K — Grade School Math

GSM8K (Cobbe et al., 2021) contains 8,500 grade-school math word problems requiring 2–8
reasoning steps. Example:

```
Q: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes
   muffins for her friends every day with 4. She sells the remainder at the farmers' market
   daily for $2 per fresh duck egg. How much in dollars does she make every day at the
   farmers' market?

A: Janet sells 16 - 3 - 4 = 9 duck eggs a day.
   She makes 9 * 2 = $18 every day at the farmers' market.
   #### 18
```

The answer is delimited by `####`. The `_extract_final_answer()` function in `reasoning.py`
looks for this delimiter to extract the numeric answer for evaluation.

**Why GSM8K?** It requires genuine multi-step reasoning, not just pattern matching. A model
that has memorized math facts but cannot chain reasoning steps will fail. It is also a
standard benchmark for evaluating chain-of-thought prompting.

### BIG-Bench Hard

BIG-Bench Hard (Srivastava et al., 2022) is a curated subset of 23 challenging tasks where
prior LLMs performed near or below human baselines. Tasks include boolean expressions, causal
judgment, date understanding, and more. It tests whether models can reason about novel problems
that are unlikely to appear verbatim in training data.

---

## Decoding Strategies (`inference.py`)

All four strategies are implemented from scratch using PyTorch, working with any HuggingFace
causal LM or the custom GPTModel from Project 2.

### Greedy Decoding

```python
next_token = logits[:, -1, :].argmax(dim=-1)
```

At each step, pick the token with the highest probability. Simple, fast, deterministic.

**The problem with greedy:** It is locally optimal but globally suboptimal. Consider:
- Token A has probability 0.6, followed by a dead end (all continuations have probability 0.01)
- Token B has probability 0.4, followed by a rich continuation (many high-probability tokens)

Greedy picks A and gets stuck. Beam search would find B.

**When to use:** Factual QA, code generation where there is a single correct answer and speed
matters more than diversity.

### Beam Search

```python
# Maintain beam_width candidate sequences
beams = [(0.0, prompt)]  # (cumulative_log_prob, sequence)
for step in range(max_new_tokens):
    all_candidates = []
    for cum_lp, seq in beams:
        log_probs = F.log_softmax(model(seq)[:, -1, :], dim=-1)
        top_lps, top_tokens = log_probs.topk(beam_width)
        for lp, tok in zip(top_lps, top_tokens):
            all_candidates.append((cum_lp + lp, append(seq, tok)))
    beams = sorted(all_candidates, reverse=True)[:beam_width]
```

Beam search maintains `beam_width` candidate sequences simultaneously. At each step, each
beam is expanded by `beam_width` tokens, producing `beam_width²` candidates. The top
`beam_width` by cumulative log-probability are kept.

**Cumulative log-probability:** Using log-probabilities instead of probabilities prevents
numerical underflow (multiplying many small probabilities together). Log-probabilities are
summed instead of multiplied.

**The problem with beam search:** It tends to produce generic, safe outputs. The highest-
probability sequences are often the most common phrases, not the most informative ones.

**When to use:** Machine translation, summarization — tasks where fluency and coherence matter
more than diversity.

### Top-k Sampling

```python
top_k_logits, top_k_indices = torch.topk(scaled_logits, k)
filtered_logits = torch.full_like(scaled_logits, float("-inf"))
filtered_logits.scatter_(1, top_k_indices, top_k_logits)
probs = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

Zero out all but the top-k tokens, then sample from the remaining distribution. Temperature
scales the logits before softmax: high temperature (>1) flattens the distribution (more
random), low temperature (<1) sharpens it (more deterministic).

**The problem with fixed k:** When the model is very confident (peaked distribution), k=50
might include many low-probability tokens. When uncertain (flat distribution), k=50 might
exclude good tokens. The optimal k varies by context.

**When to use:** Creative writing, story generation — tasks where diversity is valued.

### Nucleus (Top-p) Sampling

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
sorted_probs[cumulative_probs - sorted_probs > top_p] = 0.0
sorted_probs /= sorted_probs.sum()
next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
```

Sample from the smallest set of tokens whose cumulative probability exceeds `top_p`. This
adapts dynamically to the distribution:
- Confident prediction (peaked): samples from 2–3 tokens
- Uncertain prediction (flat): samples from 50+ tokens

This is generally preferred over top-k because it adapts to the model's confidence level.
`top_p=0.9` is a common default.

**When to use:** Open-ended generation, dialogue, creative tasks — the default choice for
most generation applications.

---

## KV Cache (`kv_cache.py`)

### The Problem Without Caching

Autoregressive generation is inherently sequential: to generate token T+1, you need the
output of the model on tokens 1..T. Without caching, generating a 100-token response from
a 50-token prompt requires:
- Step 1: forward pass on 50 tokens
- Step 2: forward pass on 51 tokens
- ...
- Step 100: forward pass on 149 tokens

Total computation: O(T²) — quadratic in sequence length.

### The KV Cache Solution

In the attention mechanism, the Key and Value matrices for all previous tokens are the same
at every step — only the new token's Q, K, V change. The KV cache stores the K and V tensors
from all previous steps:

```python
class KVCache:
    def update(self, layer_idx, k, v):
        # Append new k/v to cache
        self._k_cache[layer_idx] = torch.cat([self._k_cache[layer_idx], k], dim=2)
        self._v_cache[layer_idx] = torch.cat([self._v_cache[layer_idx], v], dim=2)
        return self._k_cache[layer_idx], self._v_cache[layer_idx]
```

With caching, each generation step only computes attention for the new token against the
cached K/V tensors. Computation per step is O(T) instead of O(T²).

**Speedup:** Roughly proportional to sequence length. For a 512-token context, the KV cache
provides ~10× speedup. The `benchmark_kv_cache()` function measures this empirically by
running `model.generate()` with `use_cache=True` vs. `use_cache=False`.

---

## Chain-of-Thought Prompting (`reasoning.py`)

### Zero-Shot Prompting

```python
def zero_shot_prompt(question):
    return f"Q: {question}\nA:"
```

The model is asked to answer directly. For simple factual questions, this works well. For
multi-step reasoning, the model often jumps to an incorrect answer.

### Few-Shot Prompting

```python
def few_shot_prompt(question, examples):
    parts = [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in examples]
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)
```

Providing examples of question-answer pairs in the prompt helps the model understand the
expected format and reasoning style. This is **in-context learning** — the model adapts to
the task from examples in the prompt, without any weight updates.

### Chain-of-Thought (CoT) Prompting

```python
def chain_of_thought_prompt(question, examples):
    parts = [f"Q: {ex['question']}\nA: Let's think step by step. {ex['answer']}" for ex in examples]
    parts.append(f"Q: {question}\nA: Let's think step by step.")
    return "\n\n".join(parts)
```

Adding "Let's think step by step." to the prompt (Wei et al., 2022) dramatically improves
performance on multi-step reasoning tasks. The model generates intermediate reasoning steps
before the final answer, which:
1. Forces the model to decompose the problem
2. Provides a "scratchpad" for intermediate calculations
3. Makes errors visible and correctable

**Why does CoT work?** The model has seen many examples of step-by-step reasoning in its
training data (textbooks, worked examples, etc.). The CoT trigger phrase activates this
reasoning pattern. Larger models benefit more from CoT — small models like GPT-2 may not
have enough capacity to use it effectively.

### Scratchpad Generation (`scratchpad_generate`)

A two-pass approach:
1. Generate the full reasoning chain from a CoT prompt
2. Extract the final answer using `_extract_final_answer()` (looks for `####`, "the answer is", etc.)

---

## Benchmark Runner (`benchmark.py`)

`run_benchmark()` evaluates all four strategies on a GSM8K subset and collects:

| Metric | What it measures |
|---|---|
| `exact_match_accuracy` | Fraction of problems where the extracted answer matches the reference |
| `distinct_1` | Fraction of unique unigrams across all generated responses (diversity) |
| `distinct_2` | Fraction of unique bigrams (higher-order diversity) |
| `tokens_per_sec` | Generation throughput |
| `avg_response_length` | Mean number of generated tokens |

**Distinct-n** measures output diversity. A model that always generates the same response
has distinct-1 = 0. High distinct-n indicates the model produces varied outputs across
different inputs.

---

## Key Takeaways from Project 5

1. Greedy decoding is fast and deterministic but locally suboptimal — it misses globally
   better sequences.
2. Beam search finds higher-probability sequences but produces generic outputs.
3. Nucleus sampling is the best general-purpose strategy — it adapts to the model's confidence.
4. The KV cache is not optional for production inference — it provides quadratic speedup.
5. Chain-of-thought prompting is one of the most impactful techniques in modern LLM usage —
   it elicits reasoning that the model already knows but does not express by default.
6. Exact match accuracy is a strict metric — even correct answers in different formats fail.

---

---

# Project 6 — Professional Benchmark Evaluation and Model Analysis

**Directory:** `evaluate/`
**Models:** GPT-2 (117M) vs. Pythia-160M (160M)
**Task:** Rigorous multi-dimensional evaluation of language models
**CPU time:** ~2 hours (full pipeline with both models)

---

## What This Project Is Really About

Getting a number on a benchmark is easy. Understanding what that number means, whether the
difference between two models is statistically significant, whether the model is well-calibrated,
and *why* it performs the way it does — that is hard. Project 6 applies the full professional
evaluation toolkit used by AI research labs.

The project answers six questions:
1. How do GPT-2 and Pythia-160M compare on standard benchmarks?
2. Are the differences statistically significant?
3. How confident is each model, and does that confidence match its accuracy?
4. How sensitive is performance to the number of in-context examples?
5. What do the model weights reveal about capacity and structure?
6. What do the activation distributions reveal about how representations flow through depth?

---

## The Models Being Evaluated

### GPT-2 (117M parameters)

Trained by OpenAI on WebText — a dataset of ~40GB of web text scraped from Reddit links with
≥3 upvotes. GPT-2 was the first model to demonstrate impressive zero-shot task transfer from
large-scale web text pre-training. Its training data skews toward internet prose, which means:
- Strong at fluent text generation and commonsense completion (HellaSwag)
- Weaker at academic knowledge (MMLU) — WebText has less academic content
- Poor at TruthfulQA — trained to predict common text, including common misconceptions

### Pythia-160M (160M parameters)

Trained by EleutherAI on The Pile — a diverse 825GB dataset including books, code, academic
papers, web text, and more. Pythia is designed for reproducibility: all training checkpoints
are publicly available, making it ideal for studying how models change during training.
- Broader factual coverage due to The Pile's diversity
- May underperform GPT-2 on HellaSwag due to less curated web text
- More transparent training process

---

## The Benchmarks

### ARC-Challenge (25-shot)

The AI2 Reasoning Challenge (Clark et al., 2018) contains grade-school science questions
that require genuine reasoning, not just pattern matching. Example:

```
Q: Which of the following best explains why the moon appears to change shape over a month?
(A) The moon rotates on its axis.
(B) The moon's shadow falls on Earth.
(C) Different amounts of the moon's lit surface are visible from Earth.
(D) The moon moves closer to and farther from Earth.
```

The model must select the correct answer from four options. **25-shot** means 25 example
question-answer pairs are included in the prompt before the test question. The metric is
**normalized accuracy** (`acc_norm`) — accuracy after normalizing by answer length.

### HellaSwag (10-shot)

HellaSwag (Zellers et al., 2019) tests commonsense completion. Given a partial activity
description, the model must pick the most plausible continuation from four options. The
dataset uses adversarial filtering — wrong answers are generated by a model and filtered to
be plausible-sounding but incorrect. This makes it harder than random chance would suggest.

### MMLU (5-shot)

Massive Multitask Language Understanding (Hendrycks et al., 2021) covers 57 academic subjects
including STEM, humanities, social sciences, and professional domains. It tests whether models
have absorbed academic knowledge from their training data. 5-shot means 5 examples per subject
are provided. This is the most demanding benchmark for small models — academic knowledge
requires diverse, high-quality training data.

### TruthfulQA (0-shot)

TruthfulQA (Lin et al., 2022) tests whether models avoid generating plausible-sounding but
false answers. Questions are designed to elicit common misconceptions. Example:

```
Q: What happens if you eat watermelon seeds?
```

A truthful model says "Nothing harmful happens." A model that mimics common misconceptions
might say "They grow in your stomach." The metric is **mc2** — multiple-choice accuracy
where the model must identify all true statements from a list.

**The counterintuitive finding:** Larger models can be *less* truthful on TruthfulQA, because
they are better at generating plausible-sounding text, including plausible-sounding falsehoods.

---

## The Evaluation Pipeline (`pipeline.py`)

The `EvaluationPipeline.run()` method executes 9 steps in sequence. Each step is wrapped in
`try/except` — a failure in any step is logged and the pipeline continues. This fault isolation
is essential for long-running evaluations where one step failing should not waste hours of work.

### Step 1: Dataset Statistics (`dataset_explorer.py`)

Before evaluating models, understand the data:

**N-gram overlap** (`compute_ngram_overlap`): Measures what fraction of test n-grams appear
in the training corpus. High overlap suggests **benchmark contamination** — the model may have
seen the test questions during training, making its scores artificially high.

```python
overlap = len(test_ngrams & train_ngrams) / len(test_ngrams)
```

**Domain distribution** (`compute_domain_distribution`): For datasets with source labels
(like The Pile), shows what fraction of samples come from each domain (web, books, code, etc.).

**Length distribution** (`compute_length_distribution`): Histograms of sequence lengths.
Reveals whether the evaluation corpus has very long or very short sequences that might
affect perplexity measurements.

### Step 2: Benchmark Evaluation (`evaluate.py`)

The core evaluation wraps `lm_eval.simple_evaluate()` from EleutherAI's lm-evaluation-harness:

```python
output = lm_eval.simple_evaluate(
    model=lm,
    tasks=[task_name],
    num_fewshot=num_fewshot,
    device="cpu",
)
accuracy = (
    task_results.get("acc_norm,none")   # normalized accuracy (ARC, HellaSwag)
    or task_results.get("acc,none")     # raw accuracy
    or task_results.get("mc2,none")     # TruthfulQA multiple-choice
)
```

**Why lm-evaluation-harness?** It is the de-facto standard for open LLM evaluation, used by
the HuggingFace Open LLM Leaderboard. Using it ensures results are comparable to published
benchmarks and eliminates implementation differences as a confound.

### Step 3: Local Model Evaluation

If `local_checkpoint_path` is set in the config, the pipeline loads a locally trained model
(from Project 2's `pretrain/` directory) and evaluates it on the same benchmarks. This allows
direct comparison between a model you trained yourself and established baselines.

### Step 4: Perplexity (`perplexity.py`)

Perplexity measures how well a model predicts a held-out text corpus:

```
Perplexity = exp(mean cross-entropy loss over all tokens)
```

**Implementation:** The corpus is tokenized and split into non-overlapping windows of
`context_length` tokens. For each window, the model predicts each token given all previous
tokens in the window. The cross-entropy loss is accumulated and exponentiated.

```python
for window in windows:
    input_ids = window[:-1]   # tokens 0..T-1
    target_ids = window[1:]   # tokens 1..T (shifted by 1)
    logits = model(input_ids)
    loss = cross_entropy(logits, target_ids, reduction="sum")
    total_loss += loss
perplexity = exp(total_loss / total_tokens)
```

**Interpretation:** Perplexity of 25 means the model is as uncertain as if it had to choose
uniformly among 25 equally likely tokens at each step. Lower is better. A model trained on
the same domain as the corpus will have lower perplexity than one trained on a different domain.

### Step 5: Few-Shot Sensitivity (`few_shot.py`)

`FewShotAnalyzer.run()` evaluates each model on every (task, n_shots) combination defined in
`shot_counts`. For example, ARC-Challenge is evaluated at 0, 1, 5, and 25 shots.

**Why this matters:** A model that performs well at 25-shot but poorly at 0-shot is heavily
dependent on in-context examples — it may be doing "prompt following" rather than genuine
reasoning. A model that performs consistently across shot counts has more robust knowledge.

**Sensitivity analysis:** Plotting accuracy vs. n_shots reveals:
- Monotonically increasing: model benefits from examples (typical)
- Non-monotonic: model is confused by examples at some counts (unusual, suggests instability)
- Flat: model ignores examples (may indicate the task is too easy or too hard)

### Step 6: Calibration (`calibration.py`)

**Expected Calibration Error (ECE)** measures whether a model's confidence matches its accuracy.
A perfectly calibrated model that says "I'm 70% confident" is correct 70% of the time.

**Algorithm:**
```python
# Partition predictions into n_bins=10 equal-width confidence bins
for bin in bins:
    indices = [i for i, c in enumerate(confidences) if bin.lo <= c < bin.hi]
    mean_conf = mean(confidences[i] for i in indices)
    accuracy = mean(labels[i] for i in indices)
    ece += (len(indices) / n) * abs(mean_conf - accuracy)
```

ECE = 0 means perfect calibration. ECE > 0.1 is a concern — the model's confidence scores
are not reliable for downstream decision-making.

**Reliability diagram:** A plot of mean confidence vs. empirical accuracy per bin. A perfectly
calibrated model lies on the diagonal. Points above the diagonal indicate overconfidence;
below indicates underconfidence.

### Step 7: Weight Analysis (`weight_analysis.py`)

Three analyses on model weight matrices:

**Frobenius norms** (`compute_weight_norms`):
```python
norms[name] = float(torch.linalg.norm(param.data, ord="fro"))
```
The Frobenius norm is the square root of the sum of squared elements — a measure of the
overall "energy" in a weight matrix. Layers with high Frobenius norms contribute more to
the model's output. Comparing norms across layers reveals which layers are most active.

**Singular value spectra** (`compute_singular_values`):
```python
_, singular_values, _ = torch.linalg.svd(param.data, full_matrices=False)
```
SVD decomposes a weight matrix W into U × S × Vᵀ, where S is a diagonal matrix of singular
values. The singular values reveal the **effective rank** of the weight matrix:
- Fast-decaying spectrum: low effective rank — the matrix can be approximated by a low-rank
  factorization (potential for compression via LoRA or pruning)
- Flat spectrum: high effective rank — the matrix uses its full capacity

**Dead neuron ratios** (`compute_dead_neuron_ratio`):
Forward hooks on ReLU/GELU layers measure what fraction of neurons produce near-zero
activations on typical inputs. High dead neuron ratios indicate under-utilized capacity —
the model has more parameters than it is effectively using.

### Step 8: Activation Analysis (`activation_analysis.py`)

Forward hooks on all `nn.Linear` layers capture activation tensors across multiple batches:

```python
def _make_hook(layer_name):
    def _hook(module, input, output):
        accumulated[layer_name].append(output.detach().cpu().view(-1))
    return _hook
```

After collecting activations, histograms are plotted per layer. Healthy activations are
approximately Gaussian (mean ≈ 0, moderate variance). Pathological patterns include:
- **Saturation:** activations clustered near ±max — the layer is not discriminating
- **Skew:** heavy tail in one direction — may indicate poor initialization or training instability
- **Bimodal:** two clusters — may indicate the layer has learned two distinct modes

Comparing activation distributions across layers reveals how representations evolve through
depth — early layers tend to capture low-level features, later layers capture high-level concepts.

### Step 9: Report Generation (`report.py`)

Three report formats are generated:

**CSV report** (`generate_csv_report`): A simple table with one row per model and columns for
each benchmark score plus average. Suitable for spreadsheet analysis.

**Markdown report** (`generate_markdown_report`): The same table in Markdown format for
inclusion in documentation or GitHub READMEs.

**Narrative report** (`generate_narrative_report`): A rich Markdown report with:

*Wilson 95% confidence intervals:*
```python
def wilson_ci(p, n, z=1.96):
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0, centre - margin), min(1, centre + margin)
```
The Wilson interval is preferred over the naive `p ± z*sqrt(p(1-p)/n)` because it is
asymmetric and bounded in [0, 1], giving better coverage for proportions near 0 or 1.

*Pairwise z-test for significance:*
```python
def two_proportion_ztest(n1, k1, n2, k2):
    p_pool = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    p_value = erfc(abs(z) / sqrt(2))  # two-tailed
    return p_value
```
Tests H₀: p₁ = p₂ (the two models have the same accuracy on this task). If p < 0.05, the
difference is statistically significant — unlikely to be due to sampling variability.

*Best model annotation (★):* The model with the highest score on each task is marked with ★.

*Narrative paragraph:* Auto-generated text summarizing the overall best model, which tasks
showed significant differences, and any calibration concerns.

---

## Configuration (`config.py` / `config.yaml`)

```yaml
models:
  - gpt2
  - EleutherAI/pythia-160m
tasks:
  arc_challenge: 25    # task_name: num_fewshot
  hellaswag: 10
  mmlu: 5
  truthfulqa_mc: 0
shot_counts:
  arc_challenge: [0, 1, 5, 25]   # sweep these shot counts for sensitivity analysis
  hellaswag: [0, 1, 10]
  mmlu: [0, 5]
  truthfulqa_mc: [0]
local_checkpoint_path: null      # set to evaluate a locally trained model
perplexity_corpus: ""            # path to held-out text file for perplexity
```

To add a new model, add its HuggingFace identifier to the `models` list. To add a new
benchmark, add it to `tasks` with the appropriate few-shot count.

---

## Key Takeaways from Project 6

1. A single accuracy number is not enough — always report confidence intervals and test
   for statistical significance.
2. Calibration matters as much as accuracy — a model that is 70% accurate but 95% confident
   is dangerous in high-stakes applications.
3. Few-shot sensitivity reveals whether a model is genuinely reasoning or just following prompts.
4. Weight analysis (Frobenius norms, SVD) reveals model capacity and compression potential.
5. Activation analysis reveals how representations evolve through depth and whether capacity
   is being used effectively.
6. Benchmark contamination (n-gram overlap) is a real concern — high overlap inflates scores.
7. The choice of training data matters as much as model size — Pythia-160M's diverse training
   data gives it different strengths than GPT-2's curated web text.

---

---

# Shared Infrastructure (`shared/`)

The `shared/` directory contains utilities used by all six projects. Understanding these
once means you understand them everywhere.

---

## `shared/config.py` — BaseConfig and YAML Loading

```python
@dataclass
class BaseConfig:
    seed: int = 42
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1
```

Every project's config inherits from `BaseConfig`. The `load_config(path, ConfigClass)`
function reads a YAML file and populates the dataclass fields. Unknown YAML keys are ignored;
missing keys use the dataclass defaults.

**Why dataclasses?** They provide type annotations, default values, and `__repr__` for free.
They are also serializable to dict, making it easy to log the full config as JSON.

---

## `shared/logging_utils.py` — JSONLogger

```python
class JSONLogger:
    def log(self, data: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_config(self, config) -> None:
        self.log({"type": "config", **dataclasses.asdict(config)})
```

Every log entry is a JSON object on its own line (JSONL format). This makes logs:
- **Streamable:** you can read the log while training is running
- **Queryable:** `grep '"type": "val_epoch"'` extracts all validation entries
- **Parseable:** any JSON library can read it

---

## `shared/lr_schedule.py` — Cosine with Warmup

```python
def cosine_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * progress))
    return LambdaLR(optimizer, lr_lambda)
```

`LambdaLR` applies a multiplicative factor to the base learning rate at each step. The
`lr_lambda` function returns a value in [0, 1] that is multiplied by the optimizer's `lr`.

---

## `shared/checkpointing.py` — Save and Load

```python
def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_metric):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
    }, path)
```

Saving the optimizer and scheduler state is essential for true resumability. Without them,
resuming training would restart with a fresh optimizer (no momentum history) and a fresh
scheduler (wrong LR), producing different results than uninterrupted training.

---

## `shared/seed.py` — Reproducibility

```python
def fix_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Setting all seeds ensures that data shuffling, weight initialization, and dropout are
identical across runs with the same seed. This is the foundation of reproducible research.

---

# How to Run Everything

```bash
# Install all dependencies
make setup
# or: pip install -r requirements.txt

# Run all tests (no network calls, fast)
make test
# or: python -m pytest backprop/tests/ pretrain/tests/ finetune/tests/ visiontx/tests/ infer/tests/ evaluate/tests/ -v

# Project 1 — MLP
python -m backprop.train
python -m backprop.evaluate --checkpoint outputs/project1/checkpoints/best.pt --config backprop/config.yaml

# Project 2 — Transformer pre-training (~30 min)
python -m pretrain.train
python -m pretrain.generate --prompt "Once upon a time"

# Project 3 — Alignment
python -m finetune.sft
python -m finetune.rlhf
python -m finetune.rlaif

# Project 4 — Vision Transformer
python -m visiontx.train --model vit
python -m visiontx.train --model resnet

# Project 5 — Inference benchmark
python -m infer.benchmark

# Project 6 — Benchmark evaluation (~2 hrs)
python -m evaluate.evaluate
# or with custom config:
python -m evaluate.evaluate --config evaluate/config.yaml
```

---

# Glossary of Key Terms

| Term | Definition |
|---|---|
| **Autoregressive** | Generating one token at a time, each conditioned on all previous tokens |
| **BPE** | Byte Pair Encoding — a subword tokenization algorithm that merges frequent character pairs |
| **Causal mask** | Upper-triangular mask that prevents attention to future positions |
| **Calibration** | Whether a model's confidence scores match its empirical accuracy |
| **CoT** | Chain-of-Thought — prompting technique that elicits step-by-step reasoning |
| **Cross-entropy loss** | `-log P(correct_token)` — the standard language modeling loss |
| **d_model** | The embedding dimension — the width of the transformer's internal representation |
| **Dead neuron** | A neuron that always outputs 0 (ReLU) — contributes nothing to the model |
| **ECE** | Expected Calibration Error — weighted mean absolute difference between confidence and accuracy |
| **Effective rank** | The number of significant singular values in a weight matrix |
| **Few-shot** | Providing k examples in the prompt before the test question |
| **Frobenius norm** | Square root of sum of squared elements — measures overall weight magnitude |
| **Gradient accumulation** | Accumulating gradients over multiple micro-batches before stepping |
| **Gradient clipping** | Scaling gradients when their norm exceeds a threshold — prevents explosions |
| **Inductive bias** | Assumptions built into a model architecture (e.g., CNNs assume local spatial structure) |
| **KL divergence** | Measures how different two probability distributions are |
| **KV cache** | Storing key/value tensors from previous steps to avoid recomputation |
| **LayerNorm** | Normalizes activations to zero mean and unit variance within each sample |
| **LoRA** | Low-Rank Adaptation — fine-tuning by adding low-rank matrices to frozen weights |
| **Nucleus sampling** | Sampling from the smallest set of tokens summing to top_p probability |
| **Perplexity** | `exp(mean cross-entropy loss)` — measures model uncertainty on a corpus |
| **Pre-norm** | Applying LayerNorm before attention/FFN (more stable than post-norm) |
| **REINFORCE** | Policy gradient algorithm — update policy to increase probability of high-reward actions |
| **Residual connection** | Skip connection that adds the input to the output of a block |
| **RLHF** | Reinforcement Learning from Human Feedback — optimizing a policy against a reward model |
| **SFT** | Supervised Fine-Tuning — fine-tuning on labeled instruction-response pairs |
| **SVD** | Singular Value Decomposition — factorizes a matrix into U × S × Vᵀ |
| **Temperature** | Scaling factor for logits before softmax — controls randomness of sampling |
| **Top-k sampling** | Sampling from the k highest-probability tokens |
| **Weight tying** | Sharing weights between the token embedding and LM head |
| **Wilson CI** | Confidence interval for proportions that is bounded in [0, 1] |
| **Zero-shot** | Asking the model to perform a task with no examples in the prompt |

---

# Research Papers Referenced Across All Projects

| Paper | Project | Key Contribution |
|---|---|---|
| Vaswani et al., 2017 — *Attention Is All You Need* | 2, 4 | The transformer architecture |
| Radford et al., 2019 — *GPT-2* | 2, 3, 5, 6 | Decoder-only LM, weight tying, pre-norm |
| Eldan & Li, 2023 — *TinyStories* | 2 | Pre-training dataset for small models |
| Ouyang et al., 2022 — *InstructGPT* | 3 | RLHF methodology |
| Bai et al., 2022 — *Anthropic HH-RLHF* | 3 | Human preference dataset |
| Taori et al., 2023 — *Alpaca* | 3 | Instruction fine-tuning dataset |
| Schulman et al., 2017 — *PPO* | 3 | Proximal Policy Optimization |
| Dosovitskiy et al., 2020 — *ViT* | 4 | Vision Transformer architecture |
| He et al., 2016 — *ResNet* | 4 | Deep residual learning |
| Abnar & Zuidema, 2020 — *Attention Rollout* | 4 | Quantifying attention flow |
| Wei et al., 2022 — *Chain-of-Thought* | 5 | CoT prompting for reasoning |
| Holtzman et al., 2020 — *Nucleus Sampling* | 5 | Top-p sampling |
| Cobbe et al., 2021 — *GSM8K* | 5 | Grade-school math benchmark |
| Pope et al., 2022 — *KV Cache* | 5 | Efficient transformer inference |
| Clark et al., 2018 — *ARC* | 6 | Science reasoning benchmark |
| Zellers et al., 2019 — *HellaSwag* | 6 | Commonsense completion benchmark |
| Hendrycks et al., 2021 — *MMLU* | 6 | Multitask knowledge benchmark |
| Lin et al., 2022 — *TruthfulQA* | 6 | Factual accuracy benchmark |
| Biderman et al., 2023 — *Pythia* | 6 | Reproducible LLM suite |
| He et al., 2015 — *Kaiming Init* | 1 | Weight initialization for ReLU networks |
| Glorot & Bengio, 2010 — *Xavier Init* | 1 | Weight initialization for saturating activations |
| Loshchilov & Hutter, 2019 — *AdamW* | 1–6 | Decoupled weight decay optimizer |
| Loshchilov & Hutter, 2017 — *SGDR* | 1–6 | Cosine LR schedule with warm restarts |
