"""
RLHF (PPO-style) training for Project 3.

Implements a simplified REINFORCE-style policy gradient update suitable for CPU.
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shared.logging_utils import JSONLogger
from shared.checkpointing import load_checkpoint
from finetune.reward_model import RewardModel

if TYPE_CHECKING:
    from finetune.config import AlignmentConfig

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


def _generate_responses(
    policy: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int = 64,
) -> Tensor:
    """
    Greedy generation of responses from the policy model.

    Args:
        policy: Causal LM model.
        input_ids: (B, T) prompt token IDs.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        generated: (B, T + max_new_tokens) token IDs.
    """
    policy.eval()
    with torch.no_grad():
        generated = policy.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=policy.config.eos_token_id,
        )
    return generated


def _compute_token_kl(
    policy: nn.Module,
    ref_policy: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """
    Compute token-level KL divergence KL(policy || ref_policy).

    Returns mean KL per sequence: (B,)
    """
    with torch.no_grad():
        ref_logits = ref_policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits  # (B, T, V)

    policy_logits = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits  # (B, T, V)

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)   # (B, T, V)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)          # (B, T, V)
    policy_probs = policy_log_probs.exp()                       # (B, T, V)

    # KL(p || q) = sum_v p(v) * (log p(v) - log q(v))
    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)  # (B, T)
    kl_per_seq = kl_per_token.mean(dim=-1)  # (B,)
    return kl_per_seq


def ppo_step(
    policy: nn.Module,
    ref_policy: nn.Module,
    reward_model: RewardModel,
    batch: dict[str, Tensor],
    config: "AlignmentConfig",
    logger: JSONLogger,
) -> dict[str, float]:
    """
    One PPO update step (REINFORCE-style for CPU simplicity).

    Steps:
    1. Generate responses from policy (greedy, max 64 tokens)
    2. Score with reward model → clip to ±reward_clip_bound (log warning if clipped)
    3. Compute KL(policy || ref_policy) token-level
    4. adjusted_reward = reward - kl_coeff * KL
    5. Policy gradient update (REINFORCE-style)

    # CPU-only: on a CUDA-enabled machine with BF16 you would use:
    # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

    Args:
        policy: The policy model being trained.
        ref_policy: Frozen reference policy (SFT model).
        reward_model: Trained reward model.
        batch: Dict with "input_ids" and "attention_mask" tensors.
        config: AlignmentConfig with ppo_* fields.
        logger: JSONLogger for logging metrics.

    Returns:
        Dict with keys: reward, kl, policy_loss.
    """
    device = torch.device("cpu")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # 1. Generate responses
    generated_ids = _generate_responses(policy, input_ids, max_new_tokens=64)
    gen_len = generated_ids.shape[1]
    gen_mask = torch.ones(generated_ids.shape[0], gen_len, dtype=torch.long, device=device)

    # 2. Score with reward model
    reward_model.eval()
    with torch.no_grad():
        rewards = reward_model(generated_ids, gen_mask)  # (B,)

    # Clip rewards
    clipped = rewards.clamp(-config.reward_clip_bound, config.reward_clip_bound)
    if (rewards.abs() > config.reward_clip_bound).any():
        warnings.warn(
            f"Rewards clipped: {rewards.tolist()} → {clipped.tolist()}",
            stacklevel=2,
        )
    rewards = clipped

    # 3. Compute KL divergence
    kl = _compute_token_kl(policy, ref_policy, generated_ids, gen_mask)  # (B,)

    # 4. Adjusted reward
    adjusted_reward = rewards - config.kl_coeff * kl  # (B,)

    # 5. Policy gradient update (REINFORCE)
    policy.train()
    policy_outputs = policy(
        input_ids=generated_ids,
        attention_mask=gen_mask,
        labels=generated_ids,
    )
    # Use negative log-likelihood as the policy loss, weighted by adjusted reward
    policy_loss = -(adjusted_reward.detach() * (-policy_outputs.loss)).mean()

    policy_loss.backward()

    metrics = {
        "reward": rewards.mean().item(),
        "kl": kl.mean().item(),
        "policy_loss": policy_loss.item(),
    }
    return metrics


def run_rlhf(config: "AlignmentConfig") -> None:
    """
    Run the full RLHF training loop.

    Loads SFT checkpoint as policy and frozen ref_policy, loads trained
    reward model, then runs ppo_steps PPO steps.

    Args:
        config: AlignmentConfig with ppo_* and checkpoint fields.
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers is required for run_rlhf.")

    from shared.seed import fix_all_seeds
    from finetune.data import load_alpaca

    fix_all_seeds(config.seed)
    device = torch.device("cpu")

    json_logger = JSONLogger(config.log_path)

    # Load policy (SFT checkpoint)
    policy = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    policy = policy.to(device)

    sft_ckpt = f"{config.checkpoint_dir}/sft_epoch_{config.sft_max_epochs}.pt"
    load_checkpoint(sft_ckpt, policy)

    # Load frozen reference policy
    ref_policy = AutoModelForCausalLM.from_pretrained(config.base_model_name)
    ref_policy = ref_policy.to(device)
    load_checkpoint(sft_ckpt, ref_policy)
    for param in ref_policy.parameters():
        param.requires_grad_(False)
    ref_policy.eval()

    # Load reward model
    from finetune.reward_model import RewardModel
    try:
        from transformers import AutoModel  # type: ignore
        backbone = AutoModel.from_pretrained(config.base_model_name, output_hidden_states=True)
        d_model = backbone.config.hidden_size
    except Exception:
        backbone = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        d_model = backbone.config.hidden_size

    reward_model = RewardModel(backbone, d_model).to(device)
    rm_ckpt = f"{config.checkpoint_dir}/reward_model_epoch_{config.rm_max_epochs}.pt"
    load_checkpoint(rm_ckpt, reward_model)
    reward_model.eval()

    # Data — use a small subset for PPO
    train_loader, _ = load_alpaca(config)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.ppo_lr)

    step = 0
    data_iter = iter(train_loader)

    while step < config.ppo_steps:
        try:
            batch_tensors = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch_tensors = next(data_iter)

        input_ids, attention_mask = batch_tensors
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        optimizer.zero_grad()
        metrics = ppo_step(policy, ref_policy, reward_model, batch, config, json_logger)
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % config.log_every_n_steps == 0:
            json_logger.log({
                "type": "ppo_step",
                "step": step,
                **metrics,
            })
            logger.info(
                "PPO step %d/%d — reward: %.4f, kl: %.4f, policy_loss: %.4f",
                step, config.ppo_steps,
                metrics["reward"], metrics["kl"], metrics["policy_loss"],
            )

    # Save final RLHF checkpoint
    from shared.checkpointing import save_checkpoint
    ckpt_path = f"{config.checkpoint_dir}/rlhf_final.pt"
    save_checkpoint(ckpt_path, policy, optimizer, None, epoch=0, step=step, best_metric=0.0)
    logger.info("Saved RLHF checkpoint: %s", ckpt_path)
