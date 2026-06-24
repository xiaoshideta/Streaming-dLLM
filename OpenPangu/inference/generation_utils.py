# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream



from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union

import torch
try:
    import torch_npu
except ImportError as e:
    pass
import torch.distributions as dists
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

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
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


class BlockDynamicCache(DynamicCache):
    """
    When `skip_cache_update` is True, this class does NOT update the cached key and value states.
    Instead, it concatenates the current states with the original cached states along the sequence dimension
    and returns the result. 

    Example:

        ```python
        >>> past_key_values = BlockDynamicCache()
        >>> past_key_values.skip_cache_update = True
        >>> outputs.past_key_values
        ```
    """
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        """
        Initialize a BlockDynamicCache instance.

        skip_cache_update is False by default.
        """
        super().__init__(_distributed_cache_data)
        self.skip_cache_update = False
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Behavior depends on the `skip_cache_update` flag:
        - If `skip_cache_update` is True:
            * Does NOT update the stored cache.
            * Concatenates the current `key_states` and `value_states` 
              with the original cached states along the sequence dimension.
            * Returns the concatenated result.
        - If `skip_cache_update` is False:
            * Uses the parent class update logic to update the cache.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The updated key and value states after concatenation or update.
                When `skip_cache_update=True`, returns the concatenated tensor without modifying cache.
                When `skip_cache_update=False`, returns the result from the parent class.
        """
        if self.skip_cache_update:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return key_cache, value_cache
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    

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
    block_length=kwargs.pop("block_length", 32)
    attention_mask = kwargs.pop("attention_mask", None)
    alg = kwargs.get("alg", 'origin')
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)

    if mask_token_id is None:
        raise ValueError("mask_token_id must be provided")

    if eos_token_id is None:
        raise ValueError("eos_token_id must be provided")

    if inputs is None:
        raise ValueError("inputs must be provided")

    if attention_mask is None:
        raise ValueError("attention_mask must be provided")


    input_ids = inputs

    if type(kwargs.get('max_new_tokens', None)) is int:
        max_length = kwargs.get('max_new_tokens') + input_ids.shape[-1]
    elif kwargs.get('max_length', None) is None:
        raise ValueError("Pass max_new_tokens or max_length")

    prompt_length = input_ids.shape[1]
    if (max_length - prompt_length) % block_length != 0:
        raise ValueError(
            f"The token length ({max_length - prompt_length}) "
            f"cannot be evenly divided by the block length ({block_length})."
        )

    num_blocks = (max_length - prompt_length) // block_length
    device = model.device
    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)

    # Initialize cache for the prompt
    past_key_values = BlockDynamicCache()

    causal_mask = torch.tril(torch.ones(max_length, max_length, device=device, dtype=torch.bool))[None, None, :, :]
    
    padding_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
    position_ids = padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(padding_mask == 0, 1)
    # [B, N] --> [B, 1, N, N]
    padding_mask = torch.logical_and(
        padding_mask.unsqueeze(1).unsqueeze(-2),
        padding_mask.unsqueeze(1).unsqueeze(-1),
    )
    attention_mask = padding_mask & causal_mask

    
    # Prefill stage
    if prompt_length > 0:
        cur_x = x[:, :prompt_length]
        cur_attn_mask = attention_mask[:, :, :prompt_length, :prompt_length]
        cur_position_ids = position_ids[:, :prompt_length]
        output = model(cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True
                        )
        past_key_values = output.past_key_values
    
        logits = output.logits[:, -1:]
        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, prompt_length:prompt_length + 1] = x0
        
    # Process each block
    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length
        cur_x = x[:, block_start:block_end]
        cur_attn_mask = attention_mask[:, :, block_start:block_end, :block_end]
        cur_padding_mask = padding_mask[:, :, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]    
        # Use cache for generation
        small_block_length = block_length // num_small_blocks

        if block_length % num_small_blocks != 0:
            raise ValueError(
                f"block_length ({block_length}) must be divisible by num_small_blocks ({num_small_blocks})."
            )

        # Just concatenates current key value states, do not update key value cache
        past_key_values.skip_cache_update = True
        for small_block_idx in range(num_small_blocks):
            small_block_start = small_block_idx * small_block_length
            small_block_end = small_block_start + small_block_length

            while True:
                sub_mask_index = (cur_x[:, small_block_start:small_block_end] == mask_token_id)
                if sub_mask_index.sum() == 0:
                    break
                
                output = model(cur_x,
                        attention_mask=cur_padding_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True)
                logits = output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = logits[:, small_block_start:small_block_end]

                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k,
                                            neg_entropy=(alg == 'entropy'), margin_confidence=(alg == 'topk_margin'))
                confidence = torch.where(sub_mask_index, confidence, -torch.inf)
                transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=small_block_length) == 1)
                if alg == 'confidence_threshold':
                    transfer_index |= (confidence > threshold)
                cur_x[:, small_block_start:small_block_end][transfer_index] = x0[transfer_index]

            if eos_token_id and (x[:, prompt_length:] == eos_token_id).any(dim=1).all():
                return x
            
        # Store kv cache
        past_key_values.skip_cache_update = False
        output = model(cur_x, 
                        attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                        )
        past_key_values = output.past_key_values
        if num_block < num_blocks - 1:
            logits = output.logits[:, -1:]
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, block_end:block_end + 1] = x0

    return x










