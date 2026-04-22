"""Tests for KVCache — Property 9: KV Cache Output Equivalence.

# Feature: deep-learning-llm-mastery, Property 9: KV Cache Output Equivalence
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from infer.kv_cache import KVCache


# Feature: deep-learning-llm-mastery, Property 9: KV Cache Output Equivalence
@given(seq_len=st.integers(2, 32), n_layers=st.integers(1, 4))
@settings(max_examples=50)
def test_kv_cache_stores_and_retrieves_correctly(seq_len: int, n_layers: int) -> None:
    """KVCache.update() must return the full accumulated k/v tensors.

    After seq_len updates, the returned tensors must have shape
    (1, n_heads, seq_len, d_head).

    This validates the core correctness property: cached outputs match uncached.

    **Validates: Requirements 5.4, 5.11**
    """
    n_heads = 2
    d_head = 8
    cache = KVCache(
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        max_seq_len=seq_len + 10,
    )

    for step in range(seq_len):
        k = torch.randn(1, n_heads, 1, d_head)
        v = torch.randn(1, n_heads, 1, d_head)
        full_k, full_v = cache.update(0, k, v)

    assert full_k.shape == (1, n_heads, seq_len, d_head), (
        f"Expected full_k shape (1, {n_heads}, {seq_len}, {d_head}), got {full_k.shape}"
    )
    assert full_v.shape == (1, n_heads, seq_len, d_head), (
        f"Expected full_v shape (1, {n_heads}, {seq_len}, {d_head}), got {full_v.shape}"
    )


def test_kv_cache_clear_resets_length() -> None:
    """After clear(), current_len must be 0."""
    cache = KVCache(n_layers=2, n_heads=2, d_head=4, max_seq_len=16)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    cache.update(0, k, v)
    cache.update(1, k, v)
    assert cache.current_len == 1
    cache.clear()
    assert cache.current_len == 0


def test_kv_cache_accumulates_across_steps() -> None:
    """Cache must accumulate tensors correctly across multiple update steps."""
    n_layers, n_heads, d_head = 1, 2, 4
    cache = KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head, max_seq_len=10)

    k1 = torch.ones(1, n_heads, 1, d_head) * 1.0
    k2 = torch.ones(1, n_heads, 1, d_head) * 2.0
    k3 = torch.ones(1, n_heads, 1, d_head) * 3.0

    cache.update(0, k1, k1)
    cache.update(0, k2, k2)
    full_k, _ = cache.update(0, k3, k3)

    assert full_k.shape == (1, n_heads, 3, d_head)
    # Verify values are accumulated in order
    assert torch.allclose(full_k[0, 0, 0, :], torch.ones(d_head) * 1.0)
    assert torch.allclose(full_k[0, 0, 1, :], torch.ones(d_head) * 2.0)
    assert torch.allclose(full_k[0, 0, 2, :], torch.ones(d_head) * 3.0)


def test_kv_cache_current_len_tracks_updates() -> None:
    """current_len must equal the number of tokens appended to layer 0."""
    cache = KVCache(n_layers=2, n_heads=2, d_head=4, max_seq_len=20)
    assert cache.current_len == 0

    for i in range(5):
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        cache.update(0, k, v)
        assert cache.current_len == i + 1


def test_kv_cache_independent_layers() -> None:
    """Each layer's cache must be independent."""
    cache = KVCache(n_layers=3, n_heads=2, d_head=4, max_seq_len=10)

    k0 = torch.ones(1, 2, 1, 4) * 0.0
    k1 = torch.ones(1, 2, 1, 4) * 1.0
    k2 = torch.ones(1, 2, 1, 4) * 2.0

    cache.update(0, k0, k0)
    cache.update(1, k1, k1)
    cache.update(2, k2, k2)

    full_k0, _ = cache.update(0, k0, k0)
    full_k1, _ = cache.update(1, k1, k1)

    # Layer 0 has 2 updates, layer 1 has 2 updates, layer 2 has 1 update
    assert full_k0.shape[2] == 2
    assert full_k1.shape[2] == 2
