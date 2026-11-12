# Appendix: Complete 234-Point Benchmark Tables, Compiler Fixes, and Compatibility Notes (FLUX.2 Klein on TPU v7x)

This document preserves all 234 measured hardware benchmark points (`116` direct `TorchTPU` points + `118` `vLLM-Omni` OpenAI HTTP points), framework/compiler blockers and fixes, disaggregated CPU PNG/WebP encoding ablations, and setup caveats. For headline summaries and reproduction commands, see [`README.md`](./README.md).

---

## 1. Setup Differences, Framework Blockers, and `TorchTPU` Fixes

### A. Measurement Scope Between `vLLM-Omni` and Direct `TorchTPU`
- **`vLLM-Omni` (`vllm serve --omni`, `v0.32.0`, `bfloat16`):** Full OpenAI `/v1/images/generations` HTTP server. Latency and throughput include HTTP JSON request parsing, `Qwen3-4B` (`FLUX.2-klein-4B`), `Qwen3-8B` (`Flux-2-Klein-9B`), or `CLIP+T5` (`FLUX.1-schnell`) prompt encoding, DiT denoising, `AutoencoderKLFlux2` / `AutoencoderKL` VAE decode, RGB PNG compression, and Base64 JSON response serialization.
- **Direct `TorchTPU` (`scripts/run_flux2_klein_torchtpu_fp8_kv_bench.py`):** Loads the exact BFL `.safetensors` DiT weights and calibrated `float32` `input_scale` / `weight_scale` tensors (`black-forest-labs/FLUX.2-klein-4b-fp8` and `black-forest-labs/FLUX.2-klein-9b-kv-fp8`) plus `AutoencoderKLFlux2` (`32` latent channels) onto each TensorCore via `torchrun --nproc_per_node=8` (`tpu_dist`), timing compiled full-depth DiT denoising + compiled `AutoencoderKLFlux2` decode from pre-computed `512`-token text embeddings (excluding per-request `Qwen3` LLM prompt encoding and Base64 PNG HTTP serialization).

### B. What Did Not Work / Compatibility Limits & Fixes
1. **`sglang-torchtpu-diffusion` (`v0.2.0`) lacks `Flux2KleinPipeline` ([`results/sglang_compatibility_probe.json`](./results/sglang_compatibility_probe.json)):** Registers `FluxPipeline` (16-channel VAE + CLIP/T5) and `ZImagePipeline`, but does not yet register `Flux2KleinPipeline` (32-channel `AutoencoderKLFlux2` + `Qwen3ForCausalLM` joint embeddings + SwiGLU single/double blocks) or `QwenImage21Pipeline`.
2. **`vLLM-Omni` (`v0.32.0`) untiled VAE window limit at `1536×1536` and `2048×2048`:** `vLLM-Omni` `v0.32.0` served `512×512` and `1024×1024` natively, but untiled `AutoencoderKLFlux2` mid-block spatial attention exceeded XLA window bounds at `>=1536×1536`.
3. **Channels-last `NHWC` `.view()` failure in `@torch.compile` `AutoencoderKLFlux2` `GroupNorm`:** `torch_tpu` 2D convolutions emit `NHWC`-strided tensors, causing standard `GroupNorm` `.view()` calls (including 3D mid-block attention norms) to raise stride errors under `@torch.compile`. Wrapping `GroupNorm.forward` with `.contiguous().reshape(B, G, -1)` compiled the full 32-channel decoder and enabled untiled `1536×1536` and `2048×2048` (`4 MP`) execution in direct `TorchTPU`.
4. **`U=8` head divisibility on `FLUX.2-klein-4B` (`24` heads):** `Flux-2-Klein-9B` (`32` heads, `32 % 8 == 0`) supports `DP=1 × U=8` (`0.531 s` at `1024²`), whereas `FLUX.2-klein-4B` (`24` heads, `24 % 8 != 0`) cannot evenly shard heads across `U=8` and tops out at `U=4` (`0.471 s`) and `U=2` (`0.564 s`).
5. **Dual-core `/dev/vfio/<chip>` lock contention across independent OS servers:** Because each physical `v7x` chip exposes 2 TensorCores behind one `/dev/vfio/<chip>` device, separate `vllm serve` OS processes must align to whole chips (`U >= 2`: `DP=4 × U=2`, `DP=2 × U=4`, `DP=1 × U=8`). Running `DP=8 × U=1` (`1` TensorCore per replica) requires a unified `torchrun --nproc_per_node=8` `tpu_dist` slicebuilder group.
6. **GKE multi-host `TPU_VISIBLE_CHIPS=1,2,3,0` ordering:** On `2×2×2` GKE `v7x` pods, worker 0 receives twisted-torus chip ordering (`1,2,3,0`), which deadlocks single-host sub-slice ICI rings unless normalized to `0,1,2,3` alongside `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1` ([`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py)).
7. **Single-stream vs. 8-way wave CPU PNG offloading:** For a single `1024×1024` PNG image, remote HTTP RTT is slightly slower than uncontended local TPU-host CPU (`142.12 ms` vs. `112.63 ms`) due to RGB payload transfer. Remote offloading wins during concurrent `DP=8` wave completion (`170.74 ms` 8-way parallel wall time vs. `928.52 ms` serial on-host, `5.44×` faster for PNG and `7.14×` for WebP).

---

## 2. Direct `TorchTPU` `BF16` vs. Calibrated `FP8` (`4B`) & Hybrid `KV-FP8` (`9B`) Matrices

Raw logs: [`results/torchtpu_fp8_kv_host0.jsonl`](./results/torchtpu_fp8_kv_host0.jsonl), [`results/torchtpu_fp8_kv_host1.jsonl`](./results/torchtpu_fp8_kv_host1.jsonl), and [`results/torchtpu_fp8_kv_multihost_dp16.jsonl`](./results/torchtpu_fp8_kv_multihost_dp16.jsonl).

### A. `FLUX.2-klein-4B` — `BF16` vs. Calibrated `FP8` Resolution, Step & Batch Matrix (`1-Host DP=8 × U=1`)

| Suite | Precision | Resolution | Steps | Batch | `1-TC` `p50` / `p99` (s) | Single-TC IPM | Host `DP=8` Wave (s) | Host `DP=8` IPM | Scaling Eff. | HBM / TC (GiB) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resolution_step_matrix` | `BF16` | `512x512` | 1 | 1 | 0.051 / 0.052 | 1,175.42 | 0.052 | 9,230.77 | 98.2% | 14.77 |
| `resolution_step_matrix` | `BF16` | `512x512` | 4 | 1 | 0.101 / 0.101 | 592.57 | 0.102 | 4,705.88 | 99.3% | 14.77 |
| `resolution_step_matrix` | `BF16` | `512x512` | 8 | 1 | 0.168 / 0.169 | 356.51 | 0.169 | 2,840.24 | 99.6% | 14.77 |
| `resolution_step_matrix` | `BF16` | `512x512` | 20 | 1 | 0.370 / 0.370 | 162.11 | — | — | — | 14.77 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 1 | 1 | 0.226 / 0.227 | 265.00 | 0.228 | 2,105.26 | 99.3% | 14.77 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 4 | 1 | 0.456 / 0.456 | 131.62 | 0.458 | 1,048.03 | 99.5% | 14.77 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 8 | 1 | 0.761 / 0.762 | 78.80 | 0.764 | 628.27 | 99.7% | 14.77 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 20 | 1 | 1.679 / 1.679 | 35.74 | — | — | — | 14.77 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 1 | 1 | 0.561 / 0.562 | 106.90 | 0.563 | 852.58 | 99.7% | 14.77 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 4 | 1 | 1.200 / 1.200 | 50.00 | 1.202 | 399.33 | 99.8% | 14.77 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 8 | 1 | 2.051 / 2.052 | 29.25 | 2.054 | 233.69 | 99.9% | 14.77 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 1 | 1 | 1.203 / 1.204 | 49.86 | 1.206 | 398.01 | 99.8% | 14.77 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 4 | 1 | 2.743 / 2.743 | 21.88 | 2.744 | 174.93 | 99.9% | 14.77 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 8 | 1 | 4.793 / 4.793 | 12.52 | 4.796 | 100.08 | 99.9% | 14.77 |
| `batch_size_scaling` | `BF16` | `512x512` | 4 | 2 | 0.169 / 0.170 | 708.64 | 0.171 | 5,614.04 | 99.0% | 14.77 |
| `batch_size_scaling` | `BF16` | `512x512` | 4 | 4 | 0.326 / 0.327 | 735.63 | 0.328 | 5,853.66 | 99.5% | 14.77 |
| `batch_size_scaling` | `BF16` | `1024x1024` | 4 | 2 | 0.712 / 0.712 | 168.58 | 0.713 | 1,346.42 | 99.8% | 14.77 |
| `batch_size_scaling` | `BF16` | `1024x1024` | 4 | 4 | 1.396 / 1.396 | 171.98 | 1.398 | 1,373.39 | 99.8% | 14.77 |
| `resolution_step_matrix` | `FP8` | `512x512` | 1 | 1 | 0.046 / 0.047 | 1,294.65 | 0.048 | **10,000.00** | 96.6% | 11.34 |
| `resolution_step_matrix` | `FP8` | `512x512` | 4 | 1 | 0.085 / 0.085 | 704.19 | 0.086 | 5,581.40 | 99.1% | 11.34 |
| `resolution_step_matrix` | `FP8` | `512x512` | 8 | 1 | 0.136 / 0.136 | 441.71 | 0.137 | 3,503.65 | 99.2% | 11.34 |
| `resolution_step_matrix` | `FP8` | `512x512` | 20 | 1 | 0.289 / 0.289 | 207.54 | — | — | — | 11.34 |
| `resolution_step_matrix` | `FP8` | `1024x1024` | 1 | 1 | 0.211 / 0.211 | 284.79 | 0.213 | 2,253.52 | 98.9% | 11.34 |
| `resolution_step_matrix` | `FP8` | `1024x1024` | 4 | 1 | **0.394** / 0.394 | 152.41 | 0.395 | 1,215.19 | 99.7% | 11.34 |
| `resolution_step_matrix` | `FP8` | `1024x1024` | 8 | 1 | 0.638 / 0.639 | 94.03 | 0.639 | 751.17 | 99.9% | 11.34 |
| `resolution_step_matrix` | `FP8` | `1024x1024` | 20 | 1 | 1.370 / 1.370 | 43.81 | — | — | — | 11.34 |
| `resolution_step_matrix` | `FP8` | `1536x1536` | 1 | 1 | 0.530 / 0.531 | 113.14 | 0.532 | 902.26 | 99.7% | 11.34 |
| `resolution_step_matrix` | `FP8` | `1536x1536` | 4 | 1 | 1.075 / 1.076 | 55.79 | 1.077 | 445.68 | 99.8% | 11.34 |
| `resolution_step_matrix` | `FP8` | `1536x1536` | 8 | 1 | 1.803 / 1.803 | 33.28 | 1.805 | 265.93 | 99.9% | 11.34 |
| `resolution_step_matrix` | `FP8` | `2048x2048` | 1 | 1 | 1.152 / 1.152 | 52.08 | 1.155 | 415.58 | 99.7% | 11.34 |
| `resolution_step_matrix` | `FP8` | `2048x2048` | 4 | 1 | 2.537 / 2.537 | 23.65 | 2.540 | 188.98 | 99.9% | 11.34 |
| `resolution_step_matrix` | `FP8` | `2048x2048` | 8 | 1 | 4.384 / 4.384 | 13.69 | 4.387 | 109.41 | 99.9% | 11.34 |
| `batch_size_scaling` | `FP8` | `512x512` | 4 | 2 | 0.133 / 0.133 | 904.10 | 0.134 | 7,164.18 | 99.1% | 11.34 |
| `batch_size_scaling` | `FP8` | `512x512` | 4 | 4 | 0.251 / 0.251 | 955.75 | 0.252 | **7,619.05** | 99.6% | 11.34 |
| `batch_size_scaling` | `FP8` | `1024x1024` | 4 | 2 | 0.603 / 0.603 | 199.15 | 0.604 | 1,589.40 | 99.8% | 11.34 |
| `batch_size_scaling` | `FP8` | `1024x1024` | 4 | 4 | 1.172 / 1.173 | 204.72 | 1.174 | **1,635.43** | 99.9% | 11.34 |

### B. `FLUX.2-klein-9B` — `BF16` vs. Hybrid `KV-FP8` Resolution, Step & Batch Matrix (`1-Host DP=8 × U=1`)

| Suite | Precision | Resolution | Steps | Batch | `1-TC` `p50` / `p99` (s) | Single-TC IPM | Host `DP=8` Wave (s) | Host `DP=8` IPM | Scaling Eff. | HBM / TC (GiB) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resolution_step_matrix` | `BF16` | `512x512` | 1 | 1 | 0.069 / 0.069 | 872.06 | 0.070 | 6,857.14 | 98.3% | 31.93 |
| `resolution_step_matrix` | `BF16` | `512x512` | 4 | 1 | 0.173 / 0.174 | 346.45 | 0.175 | 2,742.86 | 99.0% | 31.93 |
| `resolution_step_matrix` | `BF16` | `512x512` | 8 | 1 | 0.312 / 0.313 | 192.08 | 0.314 | 1,528.66 | 99.5% | 31.93 |
| `resolution_step_matrix` | `BF16` | `512x512` | 20 | 1 | 0.731 / 0.731 | 82.08 | — | — | — | 31.93 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 1 | 1 | 0.300 / 0.300 | 199.96 | 0.301 | 1,594.68 | 99.7% | 31.93 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 4 | 1 | 0.748 / 0.748 | 80.21 | 0.749 | 640.85 | 99.9% | 31.93 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 8 | 1 | 1.345 / 1.346 | 44.59 | 1.347 | 356.35 | 99.9% | 31.93 |
| `resolution_step_matrix` | `BF16` | `1024x1024` | 20 | 1 | 3.139 / 3.139 | 19.11 | — | — | — | 31.93 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 1 | 1 | 0.749 / 0.749 | 80.12 | 0.750 | 640.00 | 99.9% | 31.93 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 4 | 1 | 1.949 / 1.949 | 30.79 | 1.950 | 246.15 | 99.9% | 31.93 |
| `resolution_step_matrix` | `BF16` | `1536x1536` | 8 | 1 | 3.549 / 3.549 | 16.91 | 3.550 | 135.21 | 100.0% | 31.93 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 1 | 1 | 1.638 / 1.639 | 36.62 | 1.639 | 292.86 | 100.0% | 31.93 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 4 | 1 | 4.480 / 4.481 | 13.39 | 4.481 | 107.12 | 100.0% | 31.93 |
| `resolution_step_matrix` | `BF16` | `2048x2048` | 8 | 1 | 8.271 / 8.271 | 7.25 | 8.271 | 58.03 | 100.0% | 31.93 |
| `batch_size_scaling` | `BF16` | `512x512` | 4 | 2 | 0.315 / 0.315 | 380.63 | 0.316 | 3,037.97 | 99.8% | 31.93 |
| `batch_size_scaling` | `BF16` | `512x512` | 4 | 4 | 0.628 / 0.629 | 382.03 | 0.629 | 3,052.46 | 99.9% | 31.93 |
| `batch_size_scaling` | `BF16` | `1024x1024` | 4 | 2 | 1.293 / 1.294 | 92.77 | 1.294 | 741.89 | 100.0% | 31.93 |
| `batch_size_scaling` | `BF16` | `1024x1024` | 4 | 4 | 2.562 / 2.563 | 93.67 | 2.563 | 749.12 | 100.0% | 31.93 |
| `resolution_step_matrix` | `KV-FP8` | `512x512` | 1 | 1 | 0.058 / 0.059 | 1,027.44 | 0.059 | **8,135.59** | 99.0% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `512x512` | 4 | 1 | 0.132 / 0.133 | 453.61 | 0.133 | 3,609.02 | 99.5% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `512x512` | 8 | 1 | 0.231 / 0.231 | 260.15 | 0.232 | 2,068.97 | 99.4% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `512x512` | 20 | 1 | 0.526 / 0.527 | 114.09 | — | — | — | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1024x1024` | 1 | 1 | 0.264 / 0.265 | 226.85 | 0.266 | 1,804.51 | 99.4% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1024x1024` | 4 | 1 | **0.608** / 0.609 | 98.66 | 0.609 | 788.18 | 99.9% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1024x1024` | 8 | 1 | 1.066 / 1.066 | 56.30 | 1.067 | 449.86 | 99.9% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1024x1024` | 20 | 1 | 2.439 / 2.439 | 24.60 | — | — | — | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1536x1536` | 1 | 1 | 0.674 / 0.674 | 89.02 | 0.675 | 711.11 | 99.9% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1536x1536` | 4 | 1 | 1.649 / 1.655 | 36.39 | 1.649 | 291.09 | 100.0% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `1536x1536` | 8 | 1 | 2.948 / 2.949 | 20.35 | 2.949 | 162.77 | 100.0% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `2048x2048` | 1 | 1 | 1.511 / 1.511 | 39.71 | 1.512 | 317.46 | 99.9% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `2048x2048` | 4 | 1 | 3.970 / 3.972 | 15.11 | 3.972 | 120.85 | 100.0% | 24.21 |
| `resolution_step_matrix` | `KV-FP8` | `2048x2048` | 8 | 1 | 7.247 / 7.248 | 8.28 | 7.249 | 66.22 | 100.0% | 24.21 |
| `batch_size_scaling` | `KV-FP8` | `512x512` | 4 | 2 | 0.226 / 0.226 | 531.56 | 0.227 | 4,229.07 | 99.4% | 24.21 |
| `batch_size_scaling` | `KV-FP8` | `512x512` | 4 | 4 | 0.440 / 0.441 | 544.84 | 0.442 | **4,343.89** | 99.7% | 24.21 |
| `batch_size_scaling` | `KV-FP8` | `1024x1024` | 4 | 2 | 0.998 / 0.999 | 120.23 | 0.999 | 960.96 | 99.9% | 24.21 |
| `batch_size_scaling` | `KV-FP8` | `1024x1024` | 4 | 4 | 1.983 / 1.983 | 121.04 | 1.984 | **967.74** | 99.9% | 24.21 |

### C. Intra-Chip `DP=4 × U=2` ICI, `1-Host DP=8` Concurrency/Poisson, and `2-Host DP=16` Slice Scaling (`4` Steps)

| Suite & Topology | Model (`Precision`) | Resolution | Load (`CON` / `RPS`) | `p50` / `p99` Latency (s) | Throughput (IPM) | DiT / Total HBM per TC |
| :--- | :--- | :---: | :--- | ---: | ---: | ---: |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-4B` (`FP8`) | `1024x1024` | `B=1` | **0.269** / 0.269 | 891.92 | `0.88 / 4.97 GiB` |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-4B` (`FP8`) | `1536x1536` | `B=1` | 0.696 / 0.696 | 345.03 | `0.88 / 4.97 GiB` |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-4B` (`FP8`) | `2048x2048` | `B=1` | 1.580 / 1.581 | 151.91 | `0.88 / 4.97 GiB` |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-9B` (`KV-FP8`) | `1024x1024` | `B=1` | **0.354** / 0.354 | 678.84 | `2.18 / 10.13 GiB` |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-9B` (`KV-FP8`) | `1536x1536` | `B=1` | 0.949 / 0.955 | 252.77 | `2.18 / 10.13 GiB` |
| `1-Host DP=4 × U=2` (Intra-Chip ICI) | `FLUX.2-klein-9B` (`KV-FP8`) | `2048x2048` | `B=1` | 2.233 / 2.234 | 107.47 | `2.18 / 10.13 GiB` |
| `1-Host DP=8 × U=1` Closed-Loop | `FLUX.2-klein-4B` (`FP8`) | `1024x1024` | `CON=1 / 8 / 64` | `0.395` / `0.398` / `3.139` (`p50`) | `151.6` / `1,193.3` / `1,211.9` | `11.34 GiB` |
| `1-Host DP=8 × U=1` Closed-Loop | `FLUX.2-klein-9B` (`KV-FP8`) | `1024x1024` | `CON=1 / 8 / 64` | `0.609` / `0.612` / `4.848` (`p50`) | `98.4` / `779.2` / `787.3` | `24.21 GiB` |
| `1-Host DP=8 × U=1` Poisson | `FLUX.2-klein-4B` (`FP8`) | `1024x1024` | `1.0 .. 8.0 RPS` | `0.395` / `0.397` (`p50`/`p99`) | `77.7` → `342.7` (`5.71 RPS`) | `11.34 GiB` |
| `1-Host DP=8 × U=1` Poisson | `FLUX.2-klein-9B` (`KV-FP8`) | `1024x1024` | `1.0 .. 8.0 RPS` | `0.609` / `0.610` (`p50`/`p99`) | `75.1` → `330.1` (`5.50 RPS`) | `24.21 GiB` |
| `2-Host DP=16 × U=1` (`16 TC`) | `FLUX.2-klein-4B` (`FP8`) | `512x512` | `CON=16 / 64` | `0.086/0.091` · `0.329/0.346` | `10,592.0` / **10,994.6** | `11.34 GiB` |
| `2-Host DP=16 × U=1` (`16 TC`) | `FLUX.2-klein-4B` (`FP8`) | `1024x1024` | `CON=16 / 64` | `0.396/0.401` · `1.567/1.584` | `2,393.0` / **2,415.7** | `11.34 GiB` |
| `2-Host DP=16 × U=1` (`16 TC`) | `FLUX.2-klein-9B` (`KV-FP8`) | `512x512` | `CON=16 / 64` | `0.134/0.138` · `0.519/0.536` | `6,895.5` / **7,138.4** | `24.21 GiB` |
| `2-Host DP=16 × U=1` (`16 TC`) | `FLUX.2-klein-9B` (`KV-FP8`) | `1024x1024` | `CON=16 / 64` | `0.609/0.614` · `2.419/2.437` | `1,562.8` / **1,572.9** | `24.21 GiB` |

---

## 3. Complete `vLLM-Omni` OpenAI `/v1/images/generations` HTTP Tables (`bfloat16`)

Raw logs: [`results/vllm_omni_host0_results.jsonl`](./results/vllm_omni_host0_results.jsonl), [`results/vllm_omni_host1_results.jsonl`](./results/vllm_omni_host1_results.jsonl), and [`results/vllm_omni_multihost_results.jsonl`](./results/vllm_omni_multihost_results.jsonl).

### A. Closed-Loop Concurrency Sweeps Across All Four Models (`1` Host & `2` Hosts)

| Model | Topology | Resolution | Steps | Concurrency | `p50` / `p95` / `p99` (s) | Throughput (IPM) |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `512x512` | 1 | `1 / 4 / 8 / 16` | `0.189` / `0.176` / `0.271` / `0.527` (`p50`) | `297.5` / `1,142.9` / `1,399.1` / `1,456.9` |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `512x512` | 4 | `1 / 4 / 8 / 16` | `0.303` / `0.359` / `0.550` / `1.174` (`p50`) | `184.5` / `610.4` / `688.6` / `722.9` |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 1 | `1 / 2 / 4 / 8 / 16` | `0.352` / `0.364` / `0.384` / `0.449` / `0.947` (`p50`) | `158.0` / `295.5` / `532.9` / `762.0` / `832.8` |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 4 | `1 / 2 / 4 / 8 / 16 / 32` | `0.564` / `0.572` / `0.594` / `0.873` / `1.812` / `3.641` (`p50`) | `100.6` / `191.4` / `363.8` / `452.7` / `473.9` / **497.2** |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 8 | `1 / 2 / 4 / 8 / 16` | `0.834` / `0.864` / `0.869` / `1.425` / `2.928` (`p50`) | `69.4` / `131.2` / `255.7` / `299.7` / `308.5` |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 20 | `1 / 2 / 4 / 8 / 16` | `1.677` / `1.677` / `1.709` / `1.752` / `3.353` (`p50`) | `35.0` / `69.2` / `134.4` / `142.7` / `144.8` |
| `FLUX.2-klein-4B` | `1-Host DP=2 x U=4 (8 TC)` | `1024x1024` | 4 | `1 / 2 / 4 / 8` | **0.471** / `0.486` / `0.710` / `1.439` (`p50`) | `116.7` / `206.9` / `274.1` / `296.8` |
| `FLUX.2-klein-4B` | `2-Host DP=8 x U=2 (16 TC)` | `512x512` | 4 | `8 / 16 / 32` | `0.319` / `0.555` / `1.108` (`p50`) | `1,198.7` / `1,373.1` / **1,394.8** |
| `FLUX.2-klein-4B` | `2-Host DP=8 x U=2 (16 TC)` | `1024x1024` | 4 | `1 / 4 / 8 / 16 / 32 / 64` | `0.577` / `0.594` / `0.624` / `0.818` / `1.764` / `3.580` (`p50`) | `97.9` / `357.9` / `660.1` / `892.1` / `941.0` / **957.2** |
| `Flux-2-Klein-9B` | `1-Host DP=4 x U=2 (8 TC)` | `512x512` | 4 | `1 / 4 / 8 / 16` | `0.353` / `0.374` / `0.714` / `1.358` (`p50`) | `162.8` / `535.6` / `574.0` / `575.3` |
| `Flux-2-Klein-9B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 1 | `1 / 4 / 8 / 16` | `0.418` / `0.443` / `0.570` / `1.188` (`p50`) | `132.8` / `470.3` / `644.7` / `676.8` |
| `Flux-2-Klein-9B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 4 | `1 / 2 / 4 / 8 / 16 / 32` | `0.802` / `0.810` / `0.827` / `1.339` / `2.701` / `5.467` (`p50`) | `72.0` / `137.9` / `269.0` / `318.8` / `330.0` / **336.7** |
| `Flux-2-Klein-9B` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 8 | `1 / 4 / 8` | `1.295` / `1.338` / `2.315` (`p50`) | `45.0` / `170.8` / `190.4` |
| `Flux-2-Klein-9B` | `1-Host DP=2 x U=4 (8 TC)` | `1024x1024` | 4 | `1 / 2 / 4` | `0.599` / `0.616` / `0.953` (`p50`) | `95.1` / `180.9` / `221.7` |
| `Flux-2-Klein-9B` | `1-Host DP=1 x U=8 (8 TC)` | `1024x1024` | 4 | `1 / 2` | **0.531** / `0.863` (`p50`) | `106.1` / `121.6` |
| `Flux-2-Klein-9B` | `2-Host DP=8 x U=2 (16 TC)` | `512x512` | 4 | `8 / 16 / 32` | `0.388` / `0.779` / `1.338` (`p50`) | `1,051.3` / `1,138.8` / **1,215.3** |
| `Flux-2-Klein-9B` | `2-Host DP=8 x U=2 (16 TC)` | `1024x1024` | 4 | `1 / 4 / 8 / 16 / 32 / 64` | `0.815` / `0.832` / `0.873` / `1.262` / `2.639` / `5.429` (`p50`) | `70.5` / `265.2` / `492.7` / `627.0` / `655.0` / **664.0** |
| `FLUX.1-schnell` | `1-Host DP=4 x U=2 (8 TC)` | `512x512` | 4 | `1 / 4 / 8` | `0.515` / `0.558` / `1.094` (`p50`) | `108.1` / `394.1` / `360.0` |
| `FLUX.1-schnell` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 4 | `1 / 4 / 8 / 16` | `1.401` / `1.457` / `2.552` / `5.171` (`p50`) | `42.0` / `157.5` / `167.4` / `179.3` |
| `Qwen-Image-2.1-Turbo` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 4 | `1 / 4 / 8` | `1.499` / `1.929` / `2.132` (`p50`) | `38.9` / `22.0` *(cold bucket)* / `181.8` |
| `Qwen-Image-2.1-Turbo` | `1-Host DP=4 x U=2 (8 TC)` | `1024x1024` | 8 | `1 / 4 / 8 / 16` | `1.170` / `1.202` / `2.063` / `4.185` (`p50`) | `49.5` / `185.9` / `210.6` / `212.5` |

### B. `vLLM-Omni` Open-Loop Poisson Arrival Sweeps (`1024×1024`, `4` Steps)

| Model | Topology | Target RPS | Achieved RPS | Achieved IPM | `p50` / `p95` / `p99` Latency (s) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `FLUX.2-klein-4B` | `1-Host DP=4 x U=2` | `1.0 / 2.0 / 4.0 / 8.0` | `1.25 / 3.18 / 4.13 / 7.34` | `74.8 / 191.0 / 247.8 / 440.2` | `0.568` / `0.594` / `0.581` / `0.838` (`p50`); `1.374` (`p99 @ 8 RPS`) |
| `FLUX.2-klein-4B` | `2-Host DP=8 x U=2` | `1.0 / 2.0 / 4.0 / 8.0` | `1.26 / 3.20 / 4.14 / 7.38` | `75.3 / 191.7 / 248.4 / 442.6` | `0.566` / `0.577` / `0.579` / **0.574** (`p50`); **0.775** (`p99 @ 8 RPS`) |
| `Flux-2-Klein-9B` | `1-Host DP=4 x U=2` | `1.0 / 2.0 / 4.0` | `1.20 / 2.91 / 3.89` | `72.0 / 174.8 / 233.3` | `0.792` / `0.802` / `0.818` (`p50`); `1.257` (`p99 @ 4 RPS`) |
| `Flux-2-Klein-9B` | `2-Host DP=8 x U=2` | `1.0 / 2.0 / 4.0 / 8.0` | `1.20 / 2.90 / 3.88 / 6.97` | `71.9 / 173.7 / 233.0 / 418.5` | `0.798` / `0.810` / `0.810` / **0.817** (`p50`); **1.245** (`p99 @ 8 RPS`) |

---

## 4. Disaggregated CPU `PNG` & `WebP` Image Encode Offload (`2 × 44`-vCPU Intel Xeon Platinum 8481C)

Raw data: [`results/image_encode_offload_host0.json`](./results/image_encode_offload_host0.json).

| Resolution | Format | Encoded Size | Single-Stream Local CPU (ms) | Single-Stream Remote RTT (ms) | `8`-Image Serial On-Host (ms) | `8`-Image Disagg Parallel Wall (ms) | `8`-Image Wave Speedup | Async TPU Host Cost (ms) |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512x512` | `PNG` | 768.8 KiB | 31.15 | 35.82 | 224.76 | 49.43 | 4.55× | 2.47 |
| `512x512` | `WEBP` | 227.9 KiB | 34.56 | 35.01 | 275.18 | 44.11 | 6.24× | 2.21 |
| `1024x1024` | `PNG` | 3,074.1 KiB | 112.63 | 142.12 | 928.52 | **170.74** | **5.44×** | 2.50 |
| `1024x1024` | `WEBP` | 907.7 KiB | 138.37 | 136.87 | 1,103.59 | **154.64** | **7.14×** | 2.50 |
| `1536x1536` | `PNG` | 6,915.9 KiB | 259.19 | 321.98 | 2,066.99 | 385.13 | 5.37× | 2.50 |
| `1536x1536` | `WEBP` | 2,041.0 KiB | 307.49 | 307.37 | 2,495.87 | 351.80 | 7.09× | 2.50 |
| `2048x2048` | `PNG` | 12,294.3 KiB | 449.85 | 571.75 | 3,637.74 | 662.20 | 5.49× | 2.50 |
| `2048x2048` | `WEBP` | 3,611.2 KiB | 554.72 | 545.78 | 4,446.50 | 625.89 | 7.10× | 2.50 |
