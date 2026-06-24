# OpenPangu

This directory contains a lightweight Open Pangu adapter for Streaming-dLLM. Open Pangu uses context-causal block diffusion, so the spatial redundancy reduction in Streaming-dLLM becomes a block-topology-aware special case, while the temporal decoding module is adapted for Open Pangu decoding on Ascend 910B.

## Model Preparation

Download `openPangu-7B-Diffusion-Base` from:

```text
https://ai.gitcode.com/ascend-tribe/openPangu-7B-Diffusion-Base
```

The downloaded model directory should contain the checkpoint shards, tokenizer files, model config, and Open Pangu remote-code files. This adapter keeps the required code files in this folder, but it does not vendor the 7B checkpoint shards.

The original Open Pangu environment targets Ascend NPU:

- Python 3.10
- CANN 8.1/8.2
- torch 2.6.0
- torch-npu 2.6.0
- transformers 4.53.2

## Usage

Run the Streaming entry point with the downloaded model path:

```shell
cd OpenPangu/inference
python generate_streaming.py \
  --model-path /path/to/openPangu-7B-Diffusion-Base \
  --prompt "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
```

The default decoding mode is `--alg streaming`. The original Open Pangu baseline script is kept as `inference/generate.py`, and the original decoding utilities are kept as `inference/generation_utils.py`.

## Results

Each cell reports accuracy and throughput in tokens per second (TPS).

| Benchmark | Open Pangu | Ours |
| --- | --- | --- |
| GSM8K | 69.29<br>11.8 TPS (1×) | **75.82**<br>18.3 TPS (1.6×) |
| MATH | 41.14<br>9.7 TPS (1×) | **41.46**<br>13.1 TPS (1.4×) |
| HumanEval | 47.56<br>10.4 TPS (1×) | **48.17**<br>14.6 TPS (1.4×) |
| MMLU-Pro | 51.65<br>16.6 TPS (1×) | **51.65**<br>25.4 TPS (1.5×) |
| BBH | 51.33<br>13.1 TPS (1×) | **51.66**<br>20.1 TPS (1.5×) |
| CMMLU | **75.46**<br>18.2 TPS (1×) | 74.72<br>27.9 TPS (1.5×) |

`generation_utils_streaming.py` implements the adapted Streaming decoding branch. The integration is intentionally minimal so it can be used as an Open Pangu extension without rewriting the original model code.
