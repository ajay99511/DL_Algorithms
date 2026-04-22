"""
Stage comparison for Project 3: generate outputs at each alignment stage.

Compares base GPT-2, SFT checkpoint, and RLHF checkpoint on 20 fixed prompts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from finetune.config import AlignmentConfig

logger = logging.getLogger(__name__)

# 20 fixed evaluation prompts (instruction strings)
EVAL_PROMPTS: list[str] = [
    "Explain the concept of gradient descent in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and JavaScript?",
    "Summarize the key ideas of the theory of evolution.",
    "Give three tips for improving sleep quality.",
    "Describe how a neural network learns from data.",
    "What is the capital of France and what is it known for?",
    "Write a recipe for a simple chocolate cake.",
    "Explain what a transformer model is in machine learning.",
    "List five benefits of regular exercise.",
    "How does photosynthesis work?",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of climate change?",
    "Explain the difference between supervised and unsupervised learning.",
    "Give advice on how to learn a new programming language.",
    "Describe the water cycle in nature.",
    "What is the Pythagorean theorem and how is it used?",
    "Write a motivational message for someone starting a new job.",
    "Explain how the internet works at a high level.",
    "What are some strategies for effective time management?",
]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _generate_text(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Generate text from a model given a prompt string."""
    inputs = tokenizer(
        f"### Instruction:\n{prompt}\n\n### Response:\n",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_comparison(config: "AlignmentConfig") -> None:
    """
    Generate outputs on 20 fixed evaluation prompts at each stage:
    base GPT-2, SFT checkpoint, RLHF checkpoint.

    Saves structured JSON to config.comparison_file.

    Args:
        config: AlignmentConfig with checkpoint and path fields.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for run_comparison.")

    from shared.checkpointing import load_checkpoint

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: list[dict] = []

    # --- Base model ---
    logger.info("Generating base model outputs...")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    base_model = base_model.to(device)
    base_model.eval()

    base_outputs = [_generate_text(base_model, tokenizer, p) for p in EVAL_PROMPTS]
    del base_model

    # --- SFT model ---
    logger.info("Generating SFT model outputs...")
    sft_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    sft_model = sft_model.to(device)
    sft_ckpt = f"{config.checkpoint_dir}/sft_epoch_{config.sft_max_epochs}.pt"
    load_checkpoint(sft_ckpt, sft_model)
    sft_model.eval()

    sft_outputs = [_generate_text(sft_model, tokenizer, p) for p in EVAL_PROMPTS]
    del sft_model

    # --- RLHF model ---
    logger.info("Generating RLHF model outputs...")
    rlhf_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    rlhf_model = rlhf_model.to(device)
    rlhf_ckpt = f"{config.checkpoint_dir}/rlhf_final.pt"
    load_checkpoint(rlhf_ckpt, rlhf_model)
    rlhf_model.eval()

    rlhf_outputs = [_generate_text(rlhf_model, tokenizer, p) for p in EVAL_PROMPTS]
    del rlhf_model

    # Assemble results
    for prompt, base_out, sft_out, rlhf_out in zip(
        EVAL_PROMPTS, base_outputs, sft_outputs, rlhf_outputs
    ):
        results.append({
            "prompt": prompt,
            "base": base_out,
            "sft": sft_out,
            "rlhf": rlhf_out,
        })

    # Save to comparison file
    Path(config.comparison_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config.comparison_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved stage comparison to %s", config.comparison_file)
