"""Tests for prompt formatting and evaluation metrics.

Tests zero_shot_prompt, few_shot_prompt, chain_of_thought_prompt,
exact_match_accuracy, and distinct_n.
"""

from __future__ import annotations

from infer.evaluate import distinct_n, exact_match_accuracy
from infer.reasoning import (
    chain_of_thought_prompt,
    few_shot_prompt,
    zero_shot_prompt,
)


# ---------------------------------------------------------------------------
# Prompt format tests
# ---------------------------------------------------------------------------

def test_zero_shot_prompt_format() -> None:
    """zero_shot_prompt must contain the question."""
    question = "What is 2 + 2?"
    prompt = zero_shot_prompt(question)
    assert question in prompt, f"Question not found in prompt: {prompt!r}"
    assert "Q:" in prompt
    assert "A:" in prompt


def test_zero_shot_prompt_structure() -> None:
    """zero_shot_prompt must follow 'Q: {question}\\nA:' format."""
    question = "How many legs does a spider have?"
    prompt = zero_shot_prompt(question)
    assert prompt == f"Q: {question}\nA:"


def test_few_shot_prompt_format() -> None:
    """few_shot_prompt must contain all example questions and answers."""
    examples = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 3 + 3?", "answer": "6"},
    ]
    question = "What is 5 + 5?"
    prompt = few_shot_prompt(question, examples)

    for ex in examples:
        assert ex["question"] in prompt, f"Example question not found: {ex['question']}"
        assert ex["answer"] in prompt, f"Example answer not found: {ex['answer']}"
    assert question in prompt, "Target question not found in prompt"


def test_few_shot_prompt_empty_examples() -> None:
    """few_shot_prompt with no examples should still include the question."""
    question = "What is 7 + 7?"
    prompt = few_shot_prompt(question, examples=[])
    assert question in prompt


def test_chain_of_thought_prompt_format() -> None:
    """chain_of_thought_prompt must contain 'Let's think step by step' CoT marker."""
    examples = [
        {"question": "If John has 3 apples and gets 2 more, how many does he have?",
         "answer": "He starts with 3 and gets 2 more. 3 + 2 = 5. The answer is 5."},
    ]
    question = "If Mary has 4 oranges and eats 1, how many remain?"
    prompt = chain_of_thought_prompt(question, examples)

    # Must contain CoT trigger phrase
    assert "Let's think step by step" in prompt, (
        f"CoT marker not found in prompt: {prompt!r}"
    )
    assert question in prompt
    for ex in examples:
        assert ex["question"] in prompt


def test_chain_of_thought_prompt_ends_with_cot_trigger() -> None:
    """The final line of chain_of_thought_prompt must end with the CoT trigger."""
    question = "What is 10 - 3?"
    prompt = chain_of_thought_prompt(question, examples=[])
    assert prompt.endswith("Let's think step by step.")


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------

def test_exact_match_accuracy_all_correct() -> None:
    """exact_match_accuracy must return 1.0 for identical lists."""
    preds = ["Paris", "London", "Berlin"]
    refs = ["Paris", "London", "Berlin"]
    assert exact_match_accuracy(preds, refs) == 1.0


def test_exact_match_accuracy_all_wrong() -> None:
    """exact_match_accuracy must return 0.0 for all-wrong predictions."""
    preds = ["Paris", "London", "Berlin"]
    refs = ["Rome", "Madrid", "Vienna"]
    assert exact_match_accuracy(preds, refs) == 0.0


def test_exact_match_accuracy_partial() -> None:
    """exact_match_accuracy must return correct fraction for partial matches."""
    preds = ["Paris", "London", "Berlin"]
    refs = ["Paris", "Madrid", "Berlin"]
    result = exact_match_accuracy(preds, refs)
    assert abs(result - 2 / 3) < 1e-9


def test_exact_match_accuracy_case_insensitive() -> None:
    """exact_match_accuracy must be case-insensitive."""
    preds = ["paris", "LONDON"]
    refs = ["Paris", "London"]
    assert exact_match_accuracy(preds, refs) == 1.0


def test_exact_match_accuracy_strips_whitespace() -> None:
    """exact_match_accuracy must strip leading/trailing whitespace."""
    preds = ["  Paris  ", "London\n"]
    refs = ["Paris", "London"]
    assert exact_match_accuracy(preds, refs) == 1.0


def test_exact_match_accuracy_empty() -> None:
    """exact_match_accuracy on empty lists must return 0.0."""
    assert exact_match_accuracy([], []) == 0.0


def test_distinct_n_range() -> None:
    """distinct_n must return a value in [0.0, 1.0]."""
    texts = ["the cat sat on the mat", "the dog ran in the park", "a bird flew over the tree"]
    for n in [1, 2, 3]:
        result = distinct_n(texts, n)
        assert 0.0 <= result <= 1.0, f"distinct_{n} out of range: {result}"


def test_distinct_1_all_unique() -> None:
    """distinct_1 must return 1.0 when all tokens are unique."""
    texts = ["alpha beta gamma delta epsilon"]
    result = distinct_n(texts, 1)
    assert result == 1.0


def test_distinct_1_all_same() -> None:
    """distinct_1 must return a low value when all tokens are identical."""
    texts = ["the the the the the"]
    result = distinct_n(texts, 1)
    assert result == 1 / 5  # 1 unique / 5 total


def test_distinct_n_empty_texts() -> None:
    """distinct_n on empty text list must return 0.0."""
    assert distinct_n([], 1) == 0.0
    assert distinct_n([""], 2) == 0.0


def test_distinct_2_bigrams() -> None:
    """distinct_2 must correctly count unique bigrams."""
    texts = ["a b c a b c"]
    # bigrams: (a,b), (b,c), (c,a), (a,b), (b,c) — 5 total, 3 unique
    result = distinct_n(texts, 2)
    assert abs(result - 3 / 5) < 1e-9
