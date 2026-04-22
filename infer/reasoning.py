"""Prompt formatting and chain-of-thought reasoning utilities.

References:
    # Ref: Wei et al., 2022 — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    # Ref: Kojima et al., 2022 — "Large Language Models are Zero-Shot Reasoners"
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from infer.config import ReasoningConfig


def zero_shot_prompt(question: str) -> str:
    """Format a zero-shot prompt.

    Args:
        question: The question to answer.

    Returns:
        Formatted prompt string: 'Q: {question}\\nA:'
    """
    return f"Q: {question}\nA:"


def few_shot_prompt(question: str, examples: list[dict]) -> str:
    """Format a few-shot prompt with examples.

    Args:
        question: The question to answer.
        examples: List of dicts with 'question' and 'answer' keys.

    Returns:
        Formatted prompt with examples followed by the target question.
    """
    parts: list[str] = []
    for ex in examples:
        parts.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def chain_of_thought_prompt(question: str, examples: list[dict]) -> str:
    """Format a chain-of-thought prompt.

    Each example's answer should include step-by-step reasoning.
    The prompt ends with a CoT trigger phrase to elicit reasoning.

    # Ref: Wei et al., 2022 — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

    Args:
        question: The question to answer.
        examples: List of dicts with 'question' and 'answer' keys where
                  answer includes reasoning steps.

    Returns:
        Formatted CoT prompt with examples and CoT trigger.
    """
    parts: list[str] = []
    for ex in examples:
        parts.append(f"Q: {ex['question']}\nA: Let's think step by step. {ex['answer']}")
    parts.append(f"Q: {question}\nA: Let's think step by step.")
    return "\n\n".join(parts)


def scratchpad_generate(
    model: nn.Module,
    tokenizer: Any,
    question: str,
    config: ReasoningConfig,
) -> tuple[str, str]:
    """Generate intermediate reasoning steps then a final answer.

    Uses a two-pass approach:
    1. First pass: generate reasoning/scratchpad steps from a CoT prompt.
    2. Second pass: extract the final answer from the generated text.

    Args:
        model: A causal language model (HF AutoModelForCausalLM or compatible).
        tokenizer: HuggingFace tokenizer.
        question: The question to reason about.
        config: ReasoningConfig with generation parameters.

    Returns:
        (scratchpad_steps, final_answer):
            scratchpad_steps: The intermediate reasoning text.
            final_answer: The extracted final answer.
    """
    device = torch.device("cpu")
    model.eval()
    model = model.to(device)

    # Pass 1: generate reasoning steps
    cot_prompt = f"Q: {question}\nA: Let's think step by step."
    inputs = tokenizer(
        cot_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        # Check if model supports generate() (HF models)
        if hasattr(model, "generate"):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = output_ids[0, input_ids.shape[1]:]
            scratchpad_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            # Fallback: use greedy decode from inference module
            from infer.inference import greedy_decode
            output_ids, _ = greedy_decode(model, input_ids, config.max_new_tokens, tokenizer)
            generated_ids = output_ids[0, input_ids.shape[1]:]
            scratchpad_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

    # Pass 2: extract final answer
    # Look for common answer markers
    final_answer = _extract_final_answer(scratchpad_text)

    return scratchpad_text, final_answer


def _extract_final_answer(text: str) -> str:
    """Extract the final answer from generated scratchpad text.

    Looks for common answer markers like 'The answer is', '####', or
    falls back to the last sentence.

    Args:
        text: Generated scratchpad text.

    Returns:
        Extracted final answer string.
    """
    # GSM8K uses #### as answer delimiter
    if "####" in text:
        parts = text.split("####")
        return parts[-1].strip()

    # Common answer markers
    markers = [
        "the answer is",
        "therefore,",
        "so the answer is",
        "the final answer is",
        "answer:",
    ]
    lower_text = text.lower()
    for marker in markers:
        idx = lower_text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip().split("\n")[0].strip()

    # Fallback: last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines[-1] if lines else text.strip()
