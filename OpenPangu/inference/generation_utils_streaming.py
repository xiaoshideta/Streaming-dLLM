# coding=utf-8
# Open Pangu diffusion decoding with Streaming-dLLM temporal commit scheduling.
# Based on Huawei's Open Pangu generation utilities and adapted for Ascend NPU/910B.
#
# Added alg options:
#   "streaming" -- temporal commit scheduling used by the Open Pangu adapter.

from collections.abc import Iterable
from typing import Optional

import torch
try:
    import torch_npu
except ImportError:
    pass
import torch.distributions as dists
from torch.nn import functional as F

from transformers.cache_utils import DynamicCache

# ---------- Utilities from the original Open Pangu generation file ----------

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                  margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


class BlockDynamicCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__(_distributed_cache_data)
        self.skip_cache_update = False

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.skip_cache_update:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return key_cache, value_cache
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

# ---------- Main decoding function ----------

@torch.no_grad()
def diffusion_generate(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    num_small_blocks: Optional[int] = 1,
    **kwargs,
):
    block_length     = kwargs.pop("block_length", 32)
    attention_mask   = kwargs.pop("attention_mask", None)
    alg              = kwargs.get("alg", "origin")
    temperature      = kwargs.get("temperature", 0.0)
    mask_token_id    = kwargs.get("mask_token_id", None)
    eos_token_id     = kwargs.get("eos_token_id", None)

    # Streaming temporal commit scheduling.
    streaming_rerank_lambda = kwargs.get("streaming_rerank_lambda", 0.05)
    streaming_extra_conf_floor = kwargs.get("streaming_extra_conf_floor", 0.75)
    streaming_extra_ratio = kwargs.get("streaming_extra_ratio", 0.1)
    streaming_extra_max = kwargs.get("streaming_extra_max", 1)
    history_gate_min_streak = kwargs.get("history_gate_min_streak", 1)
    history_gate_confidence_escape = kwargs.get("history_gate_confidence_escape", 0.9)

    if mask_token_id is None:
        raise ValueError("mask_token_id must be provided")
    if eos_token_id is None:
        raise ValueError("eos_token_id must be provided")
    if inputs is None:
        raise ValueError("inputs must be provided")
    if attention_mask is None:
        raise ValueError("attention_mask must be provided")

    input_ids = inputs

    if type(kwargs.get("max_new_tokens", None)) is int:
        max_length = kwargs.get("max_new_tokens") + input_ids.shape[-1]
    elif kwargs.get("max_length", None) is None:
        raise ValueError("Pass max_new_tokens or max_length")
    else:
        max_length = kwargs.get("max_length")

    prompt_length = input_ids.shape[1]
    if (max_length - prompt_length) % block_length != 0:
        raise ValueError(
            f"Token length ({max_length - prompt_length}) "
            f"not divisible by block_length ({block_length})."
        )

    num_blocks   = (max_length - prompt_length) // block_length
    device       = model.device
    x            = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

    past_key_values = BlockDynamicCache()
    causal_mask = torch.tril(
        torch.ones(max_length, max_length, device=device, dtype=torch.bool)
    )[None, None, :, :]

    padding_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
    position_ids = padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(padding_mask == 0, 1)
    padding_mask = torch.logical_and(
        padding_mask.unsqueeze(1).unsqueeze(-2),
        padding_mask.unsqueeze(1).unsqueeze(-1),
    )
    attention_mask = padding_mask & causal_mask

    # History gate state (per sample).
    batch_size = input_ids.shape[0]
    stable_streak = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Prefill
    if prompt_length > 0:
        cur_x = x[:, :prompt_length]
        output = model(cur_x,
                       attention_mask=attention_mask[:, :, :prompt_length, :prompt_length],
                       position_ids=position_ids[:, :prompt_length],
                       past_key_values=past_key_values,
                       use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1:]
        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, prompt_length:prompt_length + 1] = x0

    # Block 循环
    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end   = prompt_length + (num_block + 1) * block_length
        cur_x       = x[:, block_start:block_end]
        cur_attn_mask    = attention_mask[:, :, block_start:block_end, :block_end]
        cur_padding_mask = padding_mask[:, :, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        small_block_length = block_length // num_small_blocks
        if block_length % num_small_blocks != 0:
            raise ValueError(
                f"block_length ({block_length}) must be divisible by num_small_blocks ({num_small_blocks})."
            )

        past_key_values.skip_cache_update = True

        for small_block_idx in range(num_small_blocks):
            small_block_start = small_block_idx * small_block_length
            small_block_end   = small_block_start + small_block_length

            while True:
                sub_mask_index = (cur_x[:, small_block_start:small_block_end] == mask_token_id)
                if sub_mask_index.sum() == 0:
                    break

                # 标准 forward
                output = model(cur_x,
                               attention_mask=cur_padding_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True)
                logits = output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_sb = logits[:, small_block_start:small_block_end]
                confidence, x0 = sample_tokens(
                    logits_sb, temperature=temperature, top_p=top_p, top_k=top_k,
                    neg_entropy=(alg in ("entropy", "streaming")),
                    margin_confidence=(alg == "topk_margin"),
                )

                confidence = torch.where(sub_mask_index, confidence, -torch.inf)

                if alg == "streaming":
                    probs_sb = torch.softmax(logits_sb, dim=-1)
                    x0_prob = torch.gather(probs_sb, -1, x0.unsqueeze(-1)).squeeze(-1)

                    # logits score: use top1-top2 margin as rerank signal
                    top2_vals, _ = torch.topk(probs_sb, k=2, dim=-1)
                    margin = top2_vals[..., 0] - top2_vals[..., 1]
                    streaming_score = x0_prob + streaming_rerank_lambda * margin

                    # history gate
                    stable_step = (x0_prob > threshold).any(dim=1)
                    stable_streak = torch.where(
                        stable_step,
                        stable_streak + 1,
                        torch.zeros_like(stable_streak),
                    )
                    gate_open = (stable_streak >= history_gate_min_streak).unsqueeze(1)
                    gate_escape = x0_prob >= history_gate_confidence_escape
                    gate_mask = gate_open | gate_escape

                    # baseline primary commit
                    masked_conf = torch.where(sub_mask_index, x0_prob, torch.full_like(x0_prob, -torch.inf))
                    top1_idx = torch.argmax(masked_conf, dim=1, keepdim=True)
                    transfer_index = torch.zeros_like(sub_mask_index)
                    transfer_index.scatter_(1, top1_idx, True)
                    transfer_index = transfer_index & sub_mask_index

                    # gated confident commit
                    base_mask = sub_mask_index & (x0_prob > threshold) & gate_mask
                    transfer_index |= base_mask

                    # capped extra commit
                    remaining_mask = sub_mask_index & (~transfer_index)
                    if streaming_extra_max > 0:
                        conf_mask = x0_prob >= streaming_extra_conf_floor
                        candidate_mask = remaining_mask & conf_mask & gate_mask
                        max_extra = min(int(streaming_extra_max), small_block_length)
                        ratio_extra = max(0, int(sub_mask_index.shape[1] * float(streaming_extra_ratio)))
                        extra_budget = min(max_extra, ratio_extra if ratio_extra > 0 else max_extra)
                        if extra_budget > 0:
                            candidate_scores = torch.where(
                                candidate_mask,
                                streaming_score,
                                torch.full_like(streaming_score, -torch.inf),
                            )
                            topk_vals, topk_idx = torch.topk(
                                candidate_scores,
                                k=min(extra_budget, candidate_scores.shape[1]),
                                dim=1,
                            )
                            valid = torch.isfinite(topk_vals)
                            extra_mask = torch.zeros_like(sub_mask_index)
                            extra_mask.scatter_(1, topk_idx, valid)
                            transfer_index |= extra_mask
                else:
                    transfer_index = (
                        F.one_hot(torch.max(confidence, dim=1)[1], num_classes=small_block_length) == 1
                    )
                    if alg == "confidence_threshold":
                        transfer_index |= (confidence > threshold)

                cur_x[:, small_block_start:small_block_end][transfer_index] = x0[transfer_index]

            if eos_token_id and (x[:, prompt_length:] == eos_token_id).any(dim=1).all():
                return x

        # KV cache update.
        past_key_values.skip_cache_update = False
        output = model(cur_x,
                       attention_mask=cur_attn_mask,
                       position_ids=cur_position_ids,
                       past_key_values=past_key_values,
                       use_cache=True)
        past_key_values = output.past_key_values
        if num_block < num_blocks - 1:
            logits = output.logits[:, -1:]
            confidence, x0 = sample_tokens(logits, temperature=temperature,
                                           top_p=top_p, top_k=top_k)
            x[:, block_end:block_end + 1] = x0

    return x
