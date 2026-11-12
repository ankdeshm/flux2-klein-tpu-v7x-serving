# FLUX.2 Klein (4B & 9B) Serving on Google Cloud TPU v7x

End-to-end `vLLM-Omni` OpenAI HTTP image-generation serving and direct `TorchTPU` (`bfloat16`, calibrated `fp8`, and hybrid `KV-fp8`) benchmarks for Black Forest Labs `FLUX.2 Klein` (`4B` and `9B`) on Google Cloud TPU v7x (Ironwood).

**Summary:** On the end-to-end `vllm serve --omni` OpenAI HTTP stack (`1024×1024`, 4 steps, `bfloat16` including `Qwen3` prompt encoding and Base64 PNG serialization), **`FLUX.2-klein-4B` achieves 0.564 s `p50` (`0.471 s` at `U=4`) and 497.2 images/min per host (`957.2 images/min` across 2 hosts) — 2.48× lower latency and 2.77× higher per-host throughput than `FLUX.1-schnell` (`1.401 s`, `179.3 images/min`)**, while `Flux-2-Klein-9B` reaches **0.531 s `p50` (`U=8`) and 664.0 images/min across 2 hosts (`1.88×` higher per-host throughput than `FLUX.1-schnell`)**. In direct compiled `TorchTPU` `DiT + 32-ch VAE` execution (`torch._scaled_mm`), calibrated `fp8` (`4B`) and hybrid `KV-fp8` (`9B`) lower `1024×1024` 4-step latency to **0.269 s (`4B`) and 0.354 s (`9B`) with intra-chip `U=2` (`1.16–1.23×` faster than `bfloat16` on 1 TensorCore)** and reach **2,415.7 images/min (`4B`) and 1,572.9 images/min (`9B`)** across a 2-host `DP=16` slice.

| `FLUX.2-klein-4B` (`1024²`, 4 steps) | `FLUX.2-klein-9B` (`1024²`, 4 steps) | `Qwen-Image-2.1-Turbo` (`1024²`, 8 steps) | `FLUX.1-schnell` (`1024²`, 4 steps) |
| :---: | :---: | :---: | :---: |
| <img src="results/samples/flux_2_klein_4b_1024x1024_4step.png" width="190" alt="FLUX.2-klein-4B 1024x1024 4-step"/> | <img src="results/samples/flux_2_klein_9b_1024x1024_4step.png" width="190" alt="FLUX.2-klein-9B 1024x1024 4-step"/> | <img src="results/samples/qwen_image_2_1_turbo_1024x1024_8step.png" width="190" alt="Qwen-Image-2.1-Turbo 1024x1024 8-step"/> | <img src="results/samples/flux_1_schnell_1024x1024_4step.png" width="190" alt="FLUX.1-schnell 1024x1024 4-step"/> |

---

## 1. Model, Framework, and Hardware

- **Models:**
  - **`FLUX.2-klein-4B` (`~4B` DiT + `Qwen3-4B` text encoder + `32`-channel `AutoencoderKLFlux2` VAE):** `black-forest-labs/FLUX.2-klein-4B` (`bfloat16`, `14.77 GiB` peak HBM/TC) and `black-forest-labs/FLUX.2-klein-4b-fp8` (calibrated `float8_e4m3fn` with `float32` `input_scale`/`weight_scale`, `11.34 GiB` peak HBM/TC; `5` double-stream + `20` single-stream blocks, `d = 3072`, `24` heads × `128`).
  - **`FLUX.2-klein-9B` (`~9B` DiT + `Qwen3-8B` text encoder + `32`-channel `AutoencoderKLFlux2` VAE):** `Comfy-Org/Flux-2-Klein-9B` (`bfloat16`, `31.93 GiB` peak HBM/TC) and `black-forest-labs/FLUX.2-klein-9b-kv-fp8` (hybrid `KV-fp8` keeping `double_blocks.0..1` image-attention QKV/proj and text MLP in `bfloat16` and remaining blocks in calibrated `float8_e4m3fn`, `24.21 GiB` peak HBM/TC; `8` double + `24` single blocks, `d = 4096`, `32` heads × `128`).
  - **Reference Baselines (`bfloat16` in `vLLM-Omni`):** `black-forest-labs/FLUX.1-schnell` (`12B` DiT + CLIP/T5 + 16-ch VAE) and `Qwen/Qwen-Image-2-1-Turbo` (8-step distilled image generator).
- **Framework:** `PyTorch 2.11+tpu` (`torch_tpu`, `libtpu` 0.0.50, `torch.compile(dynamic=False)`, `torch._scaled_mm`), `vLLM-Omni` `v0.32.0` (`vllm serve --omni` OpenAI `/v1/images/generations`), and C3 Intel Xeon Platinum 8481C (`44` vCPUs) HTTP Keep-Alive PNG/WebP offload (`multimodal-cpu-offload-svc:8090`).
- **Hardware:** Google Cloud TPU v7x (Ironwood) on GKE (`192 GiB` HBM3e per physical chip = `2` TensorCores × `96 GiB` HBM3e):
  - Single host: `4` physical chips = `8` TensorCores (`768 GiB` HBM3e)
  - Two-host slice: `2x2x2` ICI torus = `8` physical chips = `16` TensorCores (`1.536 TB` HBM3e)

---

## 2. Main Results

### A. End-to-End OpenAI `/v1/images/generations` HTTP Serving (`vLLM-Omni v0.32.0`, `bfloat16`, `1024×1024`)

Includes HTTP JSON parsing, `Qwen3`/`T5` text encoding, DiT denoising, VAE decode, PNG compression, and Base64 serialization (**IPM** = images/min):

| Model (`Steps`) | Sharding Topology | Single-Req `p50` (`CON=1`) | Peak 1-Host IPM (`8 TC`) | Peak 2-Host IPM (`16 TC`) | Comparison |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `FLUX.1-schnell` (`4` steps) | `DP=4 × U=2` (`1` chip/replica) | 1.401 s | 179.25 (`CON=16`) | — | Prior-gen `12B` `vLLM-Omni` reference (`1.0×`) |
| `Qwen-Image-2.1-Turbo` (`8` steps) | `DP=4 × U=2` (`1` chip/replica) | 1.170 s | 212.46 (`CON=16`) | — | `1.19×` higher 1-host IPM than `FLUX.1-schnell` |
| `FLUX.2-klein-4B` (`4` steps) | `DP=2 × U=4` (`2` chips/replica) | **0.471 s** | 296.80 (`CON=8`) | — | **2.97× lower** `CON=1` latency than `FLUX.1-schnell` |
| **`FLUX.2-klein-4B` (`4` steps)** | **`DP=4 / DP=8 × U=2`** | **0.564 s** | **497.15** (`452.7 @ CON=8`) | **957.17** (`CON=64`) | **2.48× lower** latency & **2.77× higher** 1-host IPM vs. `FLUX.1-schnell`; **1.93×** 2-host scaling |
| `Flux-2-Klein-9B` (`4` steps) | `DP=1 × U=8` (`4` chips/replica) | **0.531 s** | 121.60 (`CON=2`) | — | **2.64× lower** `CON=1` latency than `FLUX.1-schnell` |
| **`Flux-2-Klein-9B` (`4` steps)** | **`DP=4 / DP=8 × U=2`** | **0.802 s** | **336.74** (`CON=32`) | **664.01** (`CON=64`) | **1.75× lower** latency & **1.88× higher** 1-host IPM vs. `FLUX.1-schnell`; **1.97×** 2-host scaling |

### B. Direct Compiled `TorchTPU` `DiT + 32-ch VAE` (`bfloat16` vs. Calibrated `FP8` / Hybrid `KV-FP8`, `4` Steps)

Executes compiled full-depth DiT denoising + compiled `AutoencoderKLFlux2` decode from pre-computed `512`-token text embeddings (excluding `Qwen3` LLM text encoding and Base64 PNG HTTP serialization):

| Model & Resolution | `1-TC BF16` `p50` | `1-TC FP8 / KV-FP8` `p50` | Intra-Chip `U=2` `p50` (`Quant`) | `1-Host DP=8` IPM (`B=1` / `B=4`) | `2-Host DP=16` IPM (`16 TC`) | Comparison (`FP8 / KV-FP8` vs. `BF16`) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `FLUX.2-klein-4B` (`512×512`) | 0.101 s | 0.085 s | — | 5,581.4 / 7,619.1 | **10,994.6** (`183.2 img/s`) | **1.19× faster** `1-TC`; **1.97×** 2-host vs. 1-host (`B=1`) |
| **`FLUX.2-klein-4B` (`1024×1024`)** | 0.456 s | **0.394 s** | **0.269 s** (`4.97 GiB`/TC) | 1,215.2 / **1,635.4** | **2,415.7** (`40.3 img/s`) | **1.16× faster** `1-TC` (`1.70×` w/ `U=2`); `14.77 → 11.34 GiB` HBM |
| `FLUX.2-klein-4B` (`2048×2048`, 4 MP) | 2.743 s | 2.537 s | **1.580 s** | 189.0 / — | — | **1.08× faster** `1-TC` (`1.74×` w/ `U=2`, untiled VAE) |
| **`FLUX.2-klein-9B` (`1024×1024`)** | 0.748 s | **0.608 s** | **0.354 s** (`10.13 GiB`/TC) | 788.2 / **967.7** | **1,572.9** (`26.2 img/s`) | **1.23× faster** `1-TC` (`2.11×` w/ `U=2`); `31.93 → 24.21 GiB` HBM |
| `FLUX.2-klein-9B` (`2048×2048`, 4 MP) | 4.480 s | 3.970 s | **2.233 s** | 120.9 / — | — | **1.13× faster** `1-TC` (`2.01×` w/ `U=2`, untiled VAE) |

*Notes:* Offloading concurrent 8-image `1024×1024` PNG / WebP encoding to C3 Xeon pods reduces wave encode wall time from `928.5 / 1,103.6 ms` (serial on-host) to **`170.7 / 154.6 ms` (`5.44× / 7.14×` faster)**. Raw logs and 16 generated PNG samples are in [`results/`](./results/).

---

## 3. Reproducing the Main Results

Scripts: [`scripts/prepare_flux2_klein_9b_diffusers.py`](./scripts/prepare_flux2_klein_9b_diffusers.py), [`scripts/run_flux2_klein_vllm_omni_serving.py`](./scripts/run_flux2_klein_vllm_omni_serving.py), and [`scripts/run_flux2_klein_torchtpu_fp8_kv_bench.py`](./scripts/run_flux2_klein_torchtpu_fp8_kv_bench.py) (or GKE manifest [`k8s/v7x-flux2-klein-serving.yaml`](./k8s/v7x-flux2-klein-serving.yaml)):

```bash
# 1. Stage FLUX.2-klein-4B and assemble Flux-2-Klein-9B Diffusers layout in /dev/shm
python3 scripts/prepare_flux2_klein_9b_diffusers.py --models-root /models/models --shm-root /dev/shm/flux2_models

# 2. Run vLLM-Omni OpenAI HTTP serving sweeps (1-host DP=4xU=2, DP=2xU=4, DP=1xU=8, and baselines)
python3 scripts/run_flux2_klein_vllm_omni_serving.py --suite host0-main --output-dir ./results
python3 scripts/run_flux2_klein_vllm_omni_serving.py --suite host1-baselines --output-dir ./results

# 3. Run Direct TorchTPU BF16 vs. calibrated FP8 (4B) & hybrid KV-FP8 (9B) sweeps (512^2..2048^2)
python3 scripts/run_flux2_klein_torchtpu_fp8_kv_bench.py --models klein_4b,klein_9b --output-dir ./results
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 234-point tables (`512²..2048²` × `1/4/8/20` steps × `B=1..4`, closed-loop `CON=1..64`, Poisson `1..8 RPS`, PNG/WebP CPU offload tables), framework compatibility findings (`sglang-torchtpu-diffusion` `v0.2.0` unregistered `Flux2KleinPipeline`, `vLLM-Omni` `>=1536²` untiled VAE limit, `U=8` 24-head divisibility limit on `4B`), and `GroupNorm` / `TPU_VISIBLE_CHIPS` fixes.
- **[`results/`](./results/):** [`flux2_klein_master_summary.json`](./results/flux2_klein_master_summary.json), per-host `.jsonl` logs, [`image_encode_offload_host0.json`](./results/image_encode_offload_host0.json), [`sglang_compatibility_probe.json`](./results/sglang_compatibility_probe.json), and [`samples/`](./results/samples/).
