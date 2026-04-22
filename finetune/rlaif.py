"""
RLAIF (Reinforcement Learning from AI Feedback) for Project 3.

Uses FLAN-T5-small as an AI judge to generate preference scores,
comparing them against the trained reward model.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from finetune.config import AlignmentConfig

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _parse_score(text: str) -> float | None:
    """
    Parse a numeric score (1-10) from generated text.

    Returns None if no valid score is found.
    """
    # Look for a number 1-10 in the text
    matches = re.findall(r"\b(10|[1-9])\b", text)
    if matches:
        return float(matches[0])
    return None


def run_rlaif(config: "AlignmentConfig") -> None:
    """
    Use FLAN-T5-small as an AI judge to score responses and compare
    against the trained reward model.

    Evaluates 50 examples from the Alpaca val set, logs reward distribution
    statistics, and saves comparison to config.comparison_file.

    Args:
        config: AlignmentConfig with rlaif_model and path fields.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for run_rlaif.")

    from shared.logging_utils import JSONLogger
    from finetune.data import load_alpaca
    from finetune.reward_model import RewardModel

    device = torch.device("cpu")
    json_logger = JSONLogger(config.log_path)

    # Load FLAN-T5-small as AI judge
    logger.info("Loading FLAN-T5-small judge: %s", config.rlaif_model)
    judge_tokenizer = AutoTokenizer.from_pretrained(config.rlaif_model)
    judge_model = AutoModelForSeq2SeqLM.from_pretrained(config.rlaif_model)
    judge_model = judge_model.to(device)
    judge_model.eval()

    # Load reward model for comparison
    try:
        from transformers import AutoModel  # type: ignore
        backbone = AutoModel.from_pretrained(config.base_model_name, output_hidden_states=True)
        d_model = backbone.config.hidden_size
    except Exception:
        backbone = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        d_model = backbone.config.hidden_size

    reward_model = RewardModel(backbone, d_model).to(device)
    rm_ckpt = f"{config.checkpoint_dir}/reward_model_epoch_{config.rm_max_epochs}.pt"
    from shared.checkpointing import load_checkpoint
    load_checkpoint(rm_ckpt, reward_model)
    reward_model.eval()

    # Load GPT-2 tokenizer for reward model
    gpt2_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    # Get 50 examples from Alpaca val set
    _, val_loader = load_alpaca(config)
    eval_examples: list[str] = []
    for batch in val_loader:
        input_ids, _ = batch
        for ids in input_ids:
            text = gpt2_tokenizer.decode(ids, skip_special_tokens=True)
            eval_examples.append(text)
            if len(eval_examples) >= 50:
                break
        if len(eval_examples) >= 50:
            break

    rlaif_scores: list[float] = []
    rm_scores: list[float] = []

    for i, response_text in enumerate(eval_examples):
        # RLAIF: prompt FLAN-T5 to rate the response
        judge_prompt = f"Rate this response 1-10: {response_text[:200]}"
        judge_inputs = judge_tokenizer(
            judge_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            judge_output = judge_model.generate(
                **judge_inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        judge_text = judge_tokenizer.decode(judge_output[0], skip_special_tokens=True)
        score = _parse_score(judge_text)
        if score is None:
            score = 5.0  # default neutral score
        rlaif_scores.append(score)

        # Reward model score
        rm_inputs = gpt2_tokenizer(
            response_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        with torch.no_grad():
            rm_score = reward_model(
                rm_inputs["input_ids"],
                rm_inputs["attention_mask"],
            ).item()
        rm_scores.append(rm_score)

        if (i + 1) % 10 == 0:
            logger.info("RLAIF eval: %d/50 examples processed", i + 1)

    # Compute statistics
    def _stats(scores: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
        }

    rlaif_stats = _stats(rlaif_scores)
    rm_stats = _stats(rm_scores)

    comparison = {
        "rlaif_scores": rlaif_scores,
        "rm_scores": rm_scores,
        "rlaif_stats": rlaif_stats,
        "rm_stats": rm_stats,
    }

    # Save comparison
    Path(config.comparison_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config.comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    json_logger.log({
        "type": "rlaif_comparison",
        "rlaif_stats": rlaif_stats,
        "rm_stats": rm_stats,
    })

    logger.info("RLAIF stats: %s", rlaif_stats)
    logger.info("RM stats: %s", rm_stats)
    logger.info("Saved RLAIF comparison to %s", config.comparison_file)
