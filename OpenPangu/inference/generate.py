# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import types
import torch
try:
    import torch_npu
except ImportError as e:
    pass
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_utils import diffusion_generate

model_local_path = "path_to_openPangu-7B-Diffusion-Base"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_local_path, 
    use_fast=False, 
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_local_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="npu",
    local_files_only=True
)

model.diffusion_generate = types.MethodType(diffusion_generate, model)

mask_token_id = 45830
eos_token_id = tokenizer.eos_token_id

prompts = ["introduce the china", "hello",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"]
input_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids.to(model.device)
# Create attention mask: Mark positions with non-padding tokens as True(attended), and padding tokens as False(ignored).
attention_mask = input_ids.ne(tokenizer.pad_token_id)

output = model.diffusion_generate(
    input_ids,
    block_length=32,
    attention_mask=attention_mask,
    temperature=0.0,
    max_new_tokens=128,
    alg="entropy",
    mask_token_id=mask_token_id,
    eos_token_id=eos_token_id,
    num_small_blocks=4
)
generation = tokenizer.batch_decode(output[:, input_ids.shape[1]:].tolist())
generation = [x.split(tokenizer.eos_token)[0].strip() for x in generation]
print(generation)