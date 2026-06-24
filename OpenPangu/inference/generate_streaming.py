# coding=utf-8
import argparse
import os
import types

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils_streaming import diffusion_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Run Open Pangu with Streaming decoding on Ascend NPU.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("OPENPANGU_MODEL_PATH", "path_to_openPangu-7B-Diffusion-Base"),
        help="Path to the downloaded openPangu-7B-Diffusion-Base model directory.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text. Can be passed multiple times for batch inference.",
    )
    parser.add_argument("--device-map", default="npu", help='Device map for transformers, e.g. "npu" or "auto".')
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--num-small-blocks", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--mask-token-id", type=int, default=45830)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--alg", default="streaming", choices=["streaming", "entropy", "confidence_threshold", "topk_margin"])
    parser.add_argument("--streaming-rerank-lambda", type=float, default=0.05)
    parser.add_argument("--streaming-extra-conf-floor", type=float, default=0.75)
    parser.add_argument("--streaming-extra-ratio", type=float, default=0.10)
    parser.add_argument("--streaming-extra-max", type=int, default=1)
    parser.add_argument("--history-gate-min-streak", type=int, default=1)
    parser.add_argument("--history-gate-confidence-escape", type=float, default=0.90)
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = args.prompt or [
        "introduce the china",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device_map,
        local_files_only=True,
    )
    model.eval()
    model.diffusion_generate = types.MethodType(diffusion_generate, model)

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    eos_token_id = args.eos_token_id if args.eos_token_id is not None else tokenizer.eos_token_id

    output = model.diffusion_generate(
        input_ids,
        block_length=args.block_length,
        attention_mask=attention_mask,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p if args.top_p < 1.0 else None,
        top_k=args.top_k,
        alg=args.alg,
        threshold=args.threshold,
        mask_token_id=args.mask_token_id,
        eos_token_id=eos_token_id,
        num_small_blocks=args.num_small_blocks,
        streaming_rerank_lambda=args.streaming_rerank_lambda,
        streaming_extra_conf_floor=args.streaming_extra_conf_floor,
        streaming_extra_ratio=args.streaming_extra_ratio,
        streaming_extra_max=args.streaming_extra_max,
        history_gate_min_streak=args.history_gate_min_streak,
        history_gate_confidence_escape=args.history_gate_confidence_escape,
    )

    generated = tokenizer.batch_decode(output[:, input_ids.shape[1]:].tolist(), skip_special_tokens=False)
    if tokenizer.eos_token is not None:
        generated = [text.split(tokenizer.eos_token)[0].strip() for text in generated]
    else:
        generated = [text.strip() for text in generated]
    for text in generated:
        print(text)


if __name__ == "__main__":
    main()
