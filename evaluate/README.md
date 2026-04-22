# Project 6 — Professional Benchmark Evaluation and Model Analysis

## Motivation

Evaluating language models rigorously is as important as training them. This project applies
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the de-facto
standard for open LLM evaluation — to compare GPT-2 and Pythia-160M across four canonical
benchmarks. Beyond accuracy numbers, we inspect *why* models behave as they do through weight
norm analysis, singular value decomposition, and activation distribution profiling.

---

## Benchmarks

| Benchmark | Task | Few-shot | What it measures |
|---|---|---|---|
| **ARC-Challenge** | `arc_challenge` | 25 | Science reasoning via multiple-choice questions drawn from grade-school exams. Models must select the correct answer from four options. (Clark et al., 2018) |
| **HellaSwag** | `hellaswag` | 10 | Commonsense completion: given a partial activity description, pick the most plausible continuation. (Zellers et al., 2019) |
| **MMLU** | `mmlu` | 5 | World knowledge across 57 academic subjects (STEM, humanities, social sciences). (Hendrycks et al., 2021) |
| **TruthfulQA** | `truthfulqa_mc` | 0 | Factual accuracy and resistance to hallucination: models must avoid plausible-sounding but false answers. (Lin et al., 2022) |

---

## Models

| Model | Parameters | Pre-training data | Citation |
|---|---|---|---|
| **GPT-2** | 117 M | WebText (~40 GB web text) | Radford et al., 2019 |
| **Pythia-160M** | 160 M | The Pile (825 GB diverse text) | Biderman et al., 2023 |

GPT-2 is a landmark autoregressive language model trained on curated web text. Pythia-160M is
part of a suite of models trained on The Pile with full reproducibility in mind, making it ideal
for controlled comparisons.

---

## Module Structure

| Module | Purpose |
|---|---|
| `config.py` | `EvalConfig` dataclass — models, tasks, output paths |
| `evaluate.py` | `run_evaluation()` — wraps `lm_eval.simple_evaluate()` with error handling |
| `weight_analysis.py` | Frobenius norms, SVD singular values, dead neuron ratios |
| `activation_analysis.py` | Forward-hook activation capture and histogram plots |
| `dataset_explorer.py` | Streaming dataset statistics (no full download) |
| `report.py` | CSV and Markdown report generation |
| `tests/` | Property-based and unit tests (Hypothesis) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt
pip install lm-eval datasets

# Run evaluation (downloads models on first run)
python -m evaluate.evaluate

# Run tests only (no network calls)
python -m pytest evaluate/tests/ -v --tb=short
```

---

## Results

| Model | ARC-Challenge | HellaSwag | MMLU | TruthfulQA | Average |
|---|---|---|---|---|---|
| gpt2 | — | — | — | — | — |
| EleutherAI/pythia-160m | — | — | — | — | — |

*Run `python -m evaluate.evaluate` to populate this table.*

---

## Weight and Activation Analysis

### Weight Norms (`weight_analysis.py`)

- **Frobenius norms** reveal which layers have the largest weight magnitudes — a proxy for
  how much each layer contributes to the model's representational capacity.
- **Singular value spectra** (via `torch.linalg.svd`) show the effective rank of each weight
  matrix. A fast-decaying spectrum indicates low effective rank; a flat spectrum suggests the
  layer uses its full capacity.
- **Dead neuron ratios** (forward hooks on ReLU/GELU) measure what fraction of neurons produce
  near-zero activations on typical inputs — a sign of under-utilised capacity.

Plots are saved to `outputs/project6/`:
- `weight_norms.png` — bar chart of Frobenius norms per layer
- `singular_values.png` — singular value spectra for the top 10 layers

### Activation Distributions (`activation_analysis.py`)

Forward hooks on all `nn.Linear` layers capture activation tensors across multiple batches.
Histograms per layer (`activation_<layer>.png`) reveal:
- Whether activations are approximately Gaussian (healthy) or heavily skewed
- Saturation effects in early vs. late layers
- How representations evolve through depth

---

## Discussion: What Results Reveal

**GPT-2** was trained on curated web text and excels at fluent generation, but its training
data skews toward internet prose. This tends to produce moderate HellaSwag scores (commonsense
completion) but weaker MMLU performance (academic knowledge) and poor TruthfulQA scores
(tendency to generate plausible-sounding falsehoods).

**Pythia-160M** benefits from The Pile's diversity (books, code, academic papers, web text),
which broadens its factual coverage. Despite being slightly larger, it may underperform GPT-2
on HellaSwag due to differences in training data curation. Its open training process makes it
easier to study and reproduce.

Both models are small by modern standards. The primary value of this evaluation is establishing
a reproducible baseline and understanding the relationship between training data, model size,
and benchmark performance — not achieving state-of-the-art results.

---

## Research Paper References

| Paper | One-line Summary |
|---|---|
| Clark et al. (2018). *Think you have Solved Question Answering? Try ARC.* | Introduces the AI2 Reasoning Challenge, a set of grade-school science questions that require genuine reasoning. |
| Zellers et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* | Proposes adversarially filtered commonsense NLI that exposes the gap between human and machine understanding. |
| Hendrycks et al. (2021). *Measuring Massive Multitask Language Understanding.* | A 57-subject academic benchmark revealing that even large models have significant knowledge gaps. |
| Lin et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* | Shows that larger models can be *less* truthful, mimicking common misconceptions from training data. |
| Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* | Introduces GPT-2 and demonstrates zero-shot task transfer from large-scale web text pre-training. |
| Biderman et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.* | Releases a family of fully reproducible models trained on The Pile, enabling controlled scaling studies. |
| Gao et al. (2021). *A Framework for Few-Shot Language Model Evaluation.* | Describes lm-evaluation-harness, the open-source evaluation framework used in this project. |
