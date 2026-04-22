# Feature: deep-learning-llm-mastery
import random
import tempfile
import os

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from shared.seed import fix_all_seeds
from shared.config import BaseConfig, save_config, load_config
from shared.checkpointing import save_checkpoint, load_checkpoint
from shared.lr_schedule import cosine_with_warmup


# ---------------------------------------------------------------------------
# Property 16: Seed Reproducibility
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=2**31))
@settings(max_examples=100)
def test_seed_reproducibility(seed):
    """Property 16: Seed Reproducibility — fix_all_seeds produces identical sequences."""
    fix_all_seeds(seed)
    torch_seq1 = torch.rand(10).tolist()
    numpy_seq1 = np.random.rand(10).tolist()
    random_seq1 = [random.random() for _ in range(10)]

    fix_all_seeds(seed)
    torch_seq2 = torch.rand(10).tolist()
    numpy_seq2 = np.random.rand(10).tolist()
    random_seq2 = [random.random() for _ in range(10)]

    assert torch_seq1 == torch_seq2, "torch.rand sequences differ across identical seeds"
    assert numpy_seq1 == numpy_seq2, "numpy.random.rand sequences differ across identical seeds"
    assert random_seq1 == random_seq2, "random.random sequences differ across identical seeds"


# ---------------------------------------------------------------------------
# Property 17: Config Round-Trip Serialization
# Validates: Requirements 7.2, 7.3
# ---------------------------------------------------------------------------

@given(
    seed=st.integers(),
    output_dir=st.text(min_size=1, max_size=50),
)
@settings(max_examples=100)
def test_config_round_trip(seed, output_dir):
    """Property 17: Config Round-Trip Serialization — serialize/deserialize preserves all fields."""
    config = BaseConfig(seed=seed, output_dir=output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.yaml")
        save_config(config, path)
        loaded = load_config(path, BaseConfig)

    assert loaded.seed == config.seed
    assert loaded.output_dir == config.output_dir
    assert loaded.log_every_n_steps == config.log_every_n_steps
    assert loaded.checkpoint_every_n_epochs == config.checkpoint_every_n_epochs


# ---------------------------------------------------------------------------
# Property 3: Checkpoint Round-Trip Fidelity
# Validates: Requirements 1.5, 2.6, 4.9
# ---------------------------------------------------------------------------

@given(hidden=st.integers(min_value=4, max_value=64))
@settings(max_examples=50, deadline=None)
def test_checkpoint_round_trip(hidden):
    """Property 3: Checkpoint Round-Trip Fidelity — saved and loaded state_dicts are identical."""
    model = nn.Linear(8, hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            step=10,
            best_metric=0.5,
        )

        # Mutate model weights to verify load restores them
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.0)

        load_checkpoint(path, model, optimizer, scheduler=None)

    loaded_state = model.state_dict()
    for key in original_state:
        assert torch.allclose(original_state[key], loaded_state[key]), (
            f"Tensor mismatch for key '{key}' after checkpoint round-trip"
        )


# ---------------------------------------------------------------------------
# Property 4: LR Schedule Monotonicity
# Validates: Requirements 1.3, 2.5, 4.10
# ---------------------------------------------------------------------------

@given(
    warmup=st.integers(min_value=1, max_value=50),
    total=st.integers(min_value=51, max_value=500),
)
@settings(max_examples=100)
def test_lr_schedule_monotonicity(warmup, total):
    """Property 4: LR Schedule Monotonicity — non-decreasing warmup, non-increasing cosine decay."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = cosine_with_warmup(optimizer, warmup_steps=warmup, total_steps=total)

    lrs = []
    for _ in range(total):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    warmup_lrs = lrs[:warmup]
    decay_lrs = lrs[warmup:]

    for i in range(len(warmup_lrs) - 1):
        assert warmup_lrs[i] <= warmup_lrs[i + 1], (
            f"LR not non-decreasing during warmup at step {i}: {warmup_lrs[i]} > {warmup_lrs[i+1]}"
        )

    for i in range(len(decay_lrs) - 1):
        assert decay_lrs[i] >= decay_lrs[i + 1], (
            f"LR not non-increasing during cosine decay at step {warmup + i}: {decay_lrs[i]} < {decay_lrs[i+1]}"
        )
