# Project 5: Reasoning, Inference Strategies, and KV Cache

## Motivation

Language models are not just trained — they are *deployed*. How you decode matters as much as how you train. This project explores the full inference stack: from the mechanics of greedy and beam search decoding, to stochastic sampling strategies, to the KV cache optimization that makes autoregressive generation practical at scale.

We also study *reasoning* — the ability of LLMs to solve multi-step problems. Chain-of-thought prompting (Wei et al., 2022) dramatically improves performance on arithmetic and commonsense tasks by eliciting intermediate reasoning steps. We evaluate these strategies on two rigorous benchmarks: GSM8K (grade-school math) and BIG-Bench Hard (diverse reasoning tasks).

---

## Datasets

### GSM8K — Grade School Math

> Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.

GSM8K contains 8,500 grade-school math word problems requiring 2–8 reasoning steps. The test split has 1,319 problems. Each problem has a natural-language question and a step-by-step solution ending with a numeric answer delimited by `####`.

**HuggingFace:** `gsm8k` (config: `main`, split: `test`)

### BIG-Bench Hard

> Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., ... & Wu, Z. (2022). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models*. arXiv:2206.04615.

BIG-Bench Hard is a curated subset of 23 challenging BIG-Bench tasks where prior LLMs performed near or below human baselines. Tasks include boolean expressions, causal judgment, date understanding, and more.

**HuggingFace:** `maveriq/bigbenchhard` (config: `boolean_expressions`, split: `train`)

---

## Module Structure

| Module | Description |
|---|---|
| `config.py` | `ReasoningConfig` dataclass — model, inference, and path settings |
| `config.yaml` | Default configuration values |
| `data.py` | `load_gsm8k()`, `load_bigbench_hard()` — HF Datasets loaders |
| `inference.py` | `greedy_decode`, `beam_search`, `top_k_sample`, `nucleus_sample` |
| `kv_cache.py` | `KVCache` class + `benchmark_kv_cache()` throughput measurement |
| `reasoning.py` | Prompt formatters: zero-shot, few-shot, chain-of-thought; `scratchpad_generate` |
| `evaluate.py` | `exact_match_accuracy`, `distinct_n` diversity metric |
| `benchmark.py` | `run_benchmark()` — end-to-end strategy comparison on GSM8K |
| `tests/` | Property and unit tests for all modules |

---

## Inference Strategy Comparison

### Greedy Decoding

At each step, select the token with the highest probability:

```
x_{t+1} = argmax P(x | x_{1:t})
```

**Pros:** Deterministic, fast, no hyperparameters.  
**Cons:** Prone to repetition; misses globally better sequences.  
**When to use:** When you need reproducibility and speed, and the task has a clear single correct answer (e.g., factual QA).

### Beam Search

Maintain `beam_width` candidate sequences, expanding each at every step and keeping the top-k by cumulative log-probability:

```
score(x_{1:t}) = sum_{i=1}^{t} log P(x_i | x_{1:i-1})
```

**Pros:** Finds higher-probability sequences than greedy; controllable via `beam_width`.  
**Cons:** Still deterministic; can produce generic/safe outputs; O(beam_width × vocab_size) per step.  
**When to use:** Machine translation, summarization — tasks where fluency and coherence matter more than diversity.

### Top-k Sampling

Sample from the k highest-probability tokens after applying temperature scaling:

```
P'(x) ∝ P(x)^{1/T}  for x in top-k tokens
```

**Pros:** Introduces diversity; temperature controls creativity.  
**Cons:** Fixed k may include low-probability tokens (when distribution is flat) or exclude good ones (when peaked).  
**When to use:** Creative text generation, story writing, dialogue systems.

### Nucleus (Top-p) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds `top_p`:

```
V_p = min{V' ⊆ V : sum_{x in V'} P(x) >= p}
```

> Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020.

**Pros:** Adapts dynamically to the distribution — uses more tokens when uncertain, fewer when confident.  
**Cons:** Slightly more complex; `top_p` requires tuning.  
**When to use:** Open-ended generation where you want diversity without sacrificing coherence. Generally preferred over top-k.

---

## KV Cache

The KV cache stores computed key and value tensors from previous positions, avoiding redundant computation during autoregressive generation. Without caching, each new token requires recomputing attention over the entire context. With caching, only the new token's attention is computed.

> Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., & Dean, J. (2022). *Efficiently Scaling Transformer Inference*. arXiv:2211.05102.

**Speedup:** Roughly proportional to sequence length — longer contexts benefit more.

---

## Chain-of-Thought Prompting

> Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.

CoT prompting adds "Let's think step by step" to the prompt, encouraging the model to generate intermediate reasoning before the final answer. This dramatically improves performance on multi-step arithmetic and commonsense reasoning tasks.

---

## Quickstart

```bash
# Run all tests
python -m pytest infer/tests/ -v

# Run benchmark (requires internet for model/dataset download)
python -c "
from infer.config import ReasoningConfig
from infer.benchmark import run_benchmark
config = ReasoningConfig(gsm8k_subset_size=10, max_new_tokens=50)
results = run_benchmark(config)
for r in results:
    print(r)
"

# Quick inference demo
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from infer.inference import greedy_decode, nucleus_sample
from infer.reasoning import chain_of_thought_prompt

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

prompt = chain_of_thought_prompt('What is 15 + 27?', [])
inputs = tokenizer(prompt, return_tensors='pt')
output, _ = greedy_decode(model, inputs['input_ids'], max_new_tokens=50, tokenizer=tokenizer)
print(tokenizer.decode(output[0], skip_special_tokens=True))
"
```

---

## Results

| Strategy | Exact Match Acc. | Distinct-1 | Distinct-2 | Tokens/sec | Avg Length |
|---|---|---|---|---|---|
| Greedy | — | — | — | — | — |
| Beam Search (w=4) | — | — | — | — | — |
| Top-k (k=50) | — | — | — | — | — |
| Nucleus (p=0.9) | — | — | — | — | — |

*Run `run_benchmark(config)` to populate this table.*

---

## Discussion: When to Use Each Strategy

| Task Type | Recommended Strategy | Reason |
|---|---|---|
| Factual QA | Greedy | Single correct answer; speed matters |
| Math reasoning | Beam search + CoT | Structured output; beam finds better solutions |
| Creative writing | Nucleus (p=0.9) | Diversity without incoherence |
| Code generation | Top-k (k=50) | Controlled diversity; syntax constraints |
| Dialogue | Nucleus (p=0.9) | Natural variation in responses |
| Summarization | Beam search | Coherence and fluency priority |

---

## Foundational Papers

1. **Greedy / Beam Search:** Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS 2014.

2. **Top-k Sampling:** Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. ACL 2018.

3. **Nucleus Sampling:** Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020.

4. **Chain-of-Thought:** Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.

5. **Zero-Shot CoT:** Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). *Large Language Models are Zero-Shot Reasoners*. NeurIPS 2022.

6. **GSM8K:** Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.

7. **BIG-Bench:** Srivastava, A. et al. (2022). *Beyond the Imitation Game*. arXiv:2206.04615.

8. **KV Cache:** Pope, R. et al. (2022). *Efficiently Scaling Transformer Inference*. arXiv:2211.05102.

9. **GPT-2:** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI Blog.
