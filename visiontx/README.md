# Project 4: Vision Transformer (ViT) for Image Classification

## Motivation

Convolutional neural networks dominated computer vision for nearly a decade. In 2020, Dosovitskiy et al. demonstrated that a pure transformer architecture — applied directly to sequences of image patches — can match or exceed CNN performance when trained at scale. This project implements a Vision Transformer from scratch and trains it on CIFAR-10, comparing it against a ResNet-18 baseline to understand the trade-offs between attention-based and convolution-based vision models.

Key questions explored:
- How does patch size affect model capacity, training speed, and accuracy?
- Where does the ViT attend? Attention rollout reveals which patches are most relevant for classification.
- How does a ~1.8M parameter ViT compare to a similarly-sized ResNet on CIFAR-10?

---

## CIFAR-10 Dataset

CIFAR-10 ([Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)) consists of 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The standard split is 50,000 training images and 10,000 test images. This project further splits 5,000 images from training as a validation set.

**Normalization statistics** (per-channel, computed on training set):

| Channel | Mean   | Std    |
|---------|--------|--------|
| Red     | 0.4914 | 0.2470 |
| Green   | 0.4822 | 0.2435 |
| Blue    | 0.4465 | 0.2616 |

---

## Architecture

### Vision Transformer (ViT)

The ViT divides each 32×32 image into non-overlapping patches, linearly projects each patch to a `d_model`-dimensional embedding, prepends a learnable class token, adds learned 1D positional embeddings, and passes the sequence through N transformer encoder blocks. The class token output is fed to an MLP classification head.

**Architecture table — patch=4 vs patch=8:**

| Config         | patch=4       | patch=8       |
|----------------|---------------|---------------|
| Image size     | 32×32         | 32×32         |
| Patch size     | 4×4           | 8×8           |
| Num patches    | 64            | 16            |
| Sequence length| 65 (+ cls)    | 17 (+ cls)    |
| d_model        | 128           | 128           |
| n_heads        | 4             | 4             |
| n_layers       | 6             | 6             |
| d_ff           | 512           | 512           |
| Parameters     | ~1.8M         | ~1.8M         |
| Dropout        | 0.1           | 0.1           |

The patch=4 config has 4× more patches, giving the model finer spatial resolution at the cost of longer sequences and slower training.

### ResNet-18 Baseline (SmallResNet)

A ResNet-18 variant adapted for 32×32 inputs: the initial 7×7 conv and max-pool are replaced with a 3×3 conv to preserve spatial resolution. Four residual stages with [64, 128, 256, 512] channels, global average pooling, and a linear classifier.

| Config         | SmallResNet   |
|----------------|---------------|
| Input size     | 32×32         |
| Residual stages| 4             |
| Channels       | 64→128→256→512|
| Parameters     | ~11.2M        |

---

## Training Hyperparameters

| Hyperparameter  | Value  | Notes                          |
|-----------------|--------|--------------------------------|
| Optimizer       | AdamW  | Loshchilov & Hutter, 2019      |
| Learning rate   | 3e-4   | Peak LR after warmup           |
| Weight decay    | 1e-2   |                                |
| Batch size      | 64     |                                |
| Max epochs      | 30     |                                |
| Warmup epochs   | 5      | Linear warmup                  |
| LR schedule     | Cosine | Min LR = 3e-5 (ratio 0.1)      |
| Grad clip norm  | 1.0    |                                |
| Seed            | 42     |                                |

---

## Module Structure

```
visiontx/
├── config.py         — ViTConfig dataclass + YAML loader
├── config.yaml       — Default hyperparameters
├── model.py          — PatchEmbedding, ViTEncoderBlock, ViT
├── baseline.py       — ResidualBlock, SmallResNet (ResNet-18 variant)
├── data.py           — CIFAR-10 and ImageNette data loaders
├── train.py          — Shared training loop (ViT + ResNet)
├── evaluate.py       — evaluate_top1()
├── visualize.py      — Training curves, patch grid visualization
├── attention_viz.py  — Attention rollout, heatmap overlay
└── tests/
    ├── test_model.py — Property tests: patch embedding, ViT output, checkpoint
    └── test_data.py  — Property tests: normalization; example: shapes
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train ViT with patch_size=4 on CIFAR-10
python -m visiontx.train --config visiontx/config.yaml --model vit

# Train ResNet baseline
python -m visiontx.train --config visiontx/config.yaml --model resnet

# Resume training from checkpoint
python -m visiontx.train --config visiontx/config.yaml --model vit --resume

# Run tests
python -m pytest visiontx/tests/ -v --tb=short
```

---

## Training Curves

After training, generate plots with:

```python
from visiontx.visualize import plot_training_curves
plot_training_curves(
    log_path="outputs/project4/experiment_log.jsonl",
    output_path="outputs/project4/plots/training_curves.png"
)
```

The plot shows train loss and validation top-1 accuracy vs epoch side by side.

---

## Attention Visualization

Attention rollout ([Abnar & Zuidema, 2020](https://arxiv.org/abs/2005.00928)) propagates attention through all layers to compute the effective relevance of each patch for the class token prediction.

```python
from visiontx.model import ViT
from visiontx.attention_viz import attention_rollout, overlay_attention_on_image

vit = ViT(config)
# ... load checkpoint ...

attn_weights = vit.get_attention_weights(image.unsqueeze(0))
rollout = attention_rollout(attn_weights, discard_ratio=0.9)
overlay_attention_on_image(image, rollout, patch_size=4, save_path="outputs/project4/plots/attention.png")
```

---

## Comparison: ViT vs CNN

| Model       | Val Accuracy | Parameters | Training Time (CPU) |
|-------------|-------------|------------|---------------------|
| ViT (p=4)   | ~75%*        | ~1.8M      | ~45 min             |
| ViT (p=8)   | ~72%*        | ~1.8M      | ~20 min             |
| SmallResNet | ~90%*        | ~11.2M     | ~30 min             |

*Placeholder values — actual results depend on hardware and training duration. CNNs have a strong inductive bias for local spatial patterns that benefits CIFAR-10; ViTs typically require more data or stronger augmentation to match CNN performance at this scale.

---

## Citations

- Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- Abnar, S., & Zuidema, W. (2020). *Quantifying Attention Flow in Transformers*. ACL 2020. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)

- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
